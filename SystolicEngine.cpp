#include "SystolicEngine.h"

// PE Output Stationary đã bao gồm tín hiệu Reset nội bộ
template<typename W_T, typename IN_T, typename OUT_T>
struct PE_OS {
    W_T weight_reg;   
    IN_T act_reg;     
    OUT_T psum_reg;   

    void compute(IN_T act_in, W_T weight_in, bool reset) {
        #pragma HLS INLINE
        OUT_T mult = (OUT_T)act_in * (OUT_T)weight_in;
        #pragma HLS BIND_OP variable=mult op=mul impl=dsp
        
        // Nếu có cờ reset đến, khởi tạo lại bộ cộng dồn với giá trị nhân hiện tại
        psum_reg = reset ? mult : (OUT_T)(psum_reg + mult);
        
        act_reg = act_in;
        weight_reg = weight_in;
    }
};

void systolic_engine(
    hls::stream<systolic_data_t> &staggered_pixels_in, 
    hls::stream<weight_mat_t> &staggered_weights_in, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<SystolicConfig> &config_stream  
) {
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE axis port=staggered_pixels_in
    #pragma HLS INTERFACE axis port=staggered_weights_in
    #pragma HLS INTERFACE axis port=psums_out
    
    SystolicConfig cfg = config_stream.read();
    int Cin = cfg.Cin;
    int Cout = cfg.Cout;
    int tiles_per_ch = cfg.tiles_per_ch;
    ap_uint<2> sel = mode_stream.read();

    if (sel != 1 && sel != 2) return; 

    bool mode_3x3 = (sel == 1);
    int k_max = mode_3x3 ? 9 : 1;
    int cout_blocks = (Cout + 15) / 16;
    if (cout_blocks == 0) cout_blocks = 1;

    static PE_OS<weight_t, pixel_t, psum_t> pe_array[ARRAY_SIZE][ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=pe_array complete dim=0

    // Khai báo các thanh ghi tạo Lệch nhịp (Skew Registers)
    static pixel_t p_skew[ARRAY_SIZE][ARRAY_SIZE];
    static weight_t w_skew[ARRAY_SIZE][ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=p_skew complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w_skew complete dim=0

    // Cần cộng thêm (ARRAY_SIZE * 2) để chờ mảng Skew đẩy dữ liệu đi hết mạng 2D
    int compute_iters = Cin * k_max + (ARRAY_SIZE * 2);

    tile_loop: for (int t = 0; t < tiles_per_ch; t++) {
        cout_loop: for (int cb = 0; cb < cout_blocks; cb++) {
            
            // ==========================================================
            // PHA 1: TÍNH TOÁN (ĐÃ CÓ SKEW LOGIC)
            // ==========================================================

            static systolic_data_t pixel_buffer[2304]; 
            #pragma HLS BIND_STORAGE variable=pixel_buffer type=ram_2p impl=bram

            compute_loop: for (int iter = 0; iter < compute_iters; iter++) {
                #pragma HLS PIPELINE II=1
                
                systolic_data_t p_vec = 0;
                weight_mat_t w_vec = 0;

                // LOGIC BYPASS & BUFFERING PIXEL
                if (iter < Cin * k_max) {
                    if (cb == 0) {
                        // Lần đầu tính toán khối này: Đọc từ stream và lưu vào Buffer
                        p_vec = staggered_pixels_in.read();
                        pixel_buffer[iter] = p_vec;
                    } else {
                        // Các Cout_block sau: Đọc lại từ Buffer nội bộ (Không làm cạn stream)
                        p_vec = pixel_buffer[iter];
                    }
                    // Trọng số thì luôn được bơm mới từ stream
                    w_vec = staggered_weights_in.read();
                }

                // 1. Cập nhật mảng Skew (Giữ nguyên đoạn code cũ của bạn)
                skew_update: for (int i = 0; i < ARRAY_SIZE; i++) {
                    #pragma HLS UNROLL
                    for (int d = ARRAY_SIZE - 1; d > 0; d--) {
                        #pragma HLS UNROLL
                        p_skew[i][d] = p_skew[i][d-1];
                        w_skew[i][d] = w_skew[i][d-1];
                    }
                    p_skew[i][0] = (pixel_t)p_vec.range(i*8+7, i*8);
                    w_skew[i][0] = (weight_t)w_vec.range(i*8+7, i*8);
                }

                // 2. Thực thi Mạng PE 2D
                update_rows: for (int r = ARRAY_SIZE - 1; r >= 0; r--) {
                    #pragma HLS UNROLL
                    update_cols: for (int c = ARRAY_SIZE - 1; c >= 0; c--) {
                        #pragma HLS UNROLL
                        
                        // Lấy Activation và Weight đã được làm lệch nhịp
                        pixel_t act_in = (c == 0) ? p_skew[r][r] : pe_array[r][c-1].act_reg;
                        weight_t w_in  = (r == 0) ? w_skew[c][c] : pe_array[r-1][c].weight_reg;
                        
                        // Tín hiệu Reset giờ đây được đối chiếu theo tọa độ chéo (r + c)
                        // PE tại [r][c] sẽ bắt đầu nhận dữ liệu hợp lệ đầu tiên ở chu kỳ (r + c)
                        bool local_reset = (iter == r + c);
                        
                        pe_array[r][c].compute(act_in, w_in, local_reset);
                    }
                }
            }

            // ==========================================================
            // PHA 2: XẢ DỮ LIỆU (DRAIN PHASE) - Không đổi
            // ==========================================================
            drain_loop: for (int r = 0; r < ARRAY_SIZE; r++) {
                #pragma HLS PIPELINE II=1
                psum_block_t out_block;
                #pragma HLS ARRAY_PARTITION variable=out_block.data complete
                
                drain_cols: for (int c = 0; c < ARRAY_SIZE; c++) {
                    #pragma HLS UNROLL
                    out_block.data[c] = pe_array[ARRAY_SIZE - 1][c].psum_reg;
                    for (int shift_r = ARRAY_SIZE - 1; shift_r > 0; shift_r--) {
                        #pragma HLS UNROLL
                        pe_array[shift_r][c].psum_reg = pe_array[shift_r - 1][c].psum_reg;
                    }
                }
                psums_out.write(out_block);
            }

        }
    }
}