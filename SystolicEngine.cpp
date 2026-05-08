#include "SystolicEngine.h"

void systolic_engine(
    hls::stream<SysWindow> &pixels_in,
    hls::stream<weight_mat_t> &weights_in, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<SystolicConfig> &config_stream  
) {
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE axis port=pixels_in
    #pragma HLS INTERFACE axis port=weights_in
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

    // Tổng số chu kỳ của toàn bộ khối Conv layer này
    int total_iters = tiles_per_ch * Cin * k_max * cout_blocks;

    // Bộ đệm tĩnh lưu Partial Sums
    psum_block_t acc_buffer[MAX_COUT_BLOCKS];
    #pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=1

    // Các biến chỉ mục cho vòng lặp phẳng
    int t = 0;
    int cin = 0;
    int k = 0;
    int cb = 0;

    SysWindow win;
    
    // Khai báo thanh ghi dùng cho Data Forwarding (Chống Data Hazard)
    psum_block_t prev_acc;
    int prev_cb = -1;

    // --- SINGLE FLATTENED LOOP ---
    systolic_flattened_loop: for (int iter = 0; iter < total_iters; iter++) {
        #pragma HLS PIPELINE II=1
        
        // Ép HLS compiler bỏ qua cảnh báo False Dependence do ta đã có logic Forwarding tự xử lý
        #pragma HLS DEPENDENCE variable=acc_buffer type=inter false
        
        // 1. ĐỌC PIXEL: Chỉ đọc SysWindow mới khi bắt đầu quét 1 điểm ảnh kernel mới
        if (k == 0 && cb == 0) {
            win = pixels_in.read();
        }

        // Đọc 16 weights
        weight_mat_t w_vec = weights_in.read();
        pixel_t p_in = win.data[k];
        
        // 2. ĐỌC ACCUMULATOR & DATA FORWARDING
        psum_block_t current_acc;
        
        if (cin == 0 && k == 0) {
            // Khởi tạo Partial Sum = 0 cho channel đầu tiên
            init_vector: for(int c = 0; c < 16; c++) {
                #pragma HLS UNROLL
                current_acc.data[c] = 0;
            }
        } else {
            // Tránh Read-After-Write (RAW) hazard khi cout_blocks = 1
            if (cb == prev_cb) {
                current_acc = prev_acc; 
            } else {
                current_acc = acc_buffer[cb];
            }
        }

        // 3. TÍNH TOÁN 16 MACs SONG SONG (Sử dụng 16 DSP slices)
        psum_block_t next_acc;
        
        vector_mac: for (int c = 0; c < 16; c++) {
            #pragma HLS UNROLL
            weight_t w = w_vec.range(c * 8 + 7, c * 8);
            psum_t product = (psum_t)p_in * (psum_t)w;
            #pragma HLS BIND_OP variable=product op=mul impl=dsp
            
            next_acc.data[c] = current_acc.data[c] + product;
        }

        // 4. LƯU KẾT QUẢ VÀO BRAM & CẬP NHẬT THANH GHI FORWARDING
        acc_buffer[cb] = next_acc;
        prev_acc = next_acc;
        prev_cb = cb;

        // 5. XUẤT OUTPUT (Chỉ xuất khi đã cộng dồn đủ toàn bộ Input Channels và vùng Kernel)
        if (cin == Cin - 1 && k == k_max - 1) {
            psums_out.write(next_acc);
        }

        // 6. CẬP NHẬT CHỈ MỤC CHO CHU KỲ TIẾP THEO (Mô phỏng vòng lặp lồng)
        cb++;
        if (cb == cout_blocks) {
            cb = 0;
            k++;
            if (k == k_max) {
                k = 0;
                cin++;
                if (cin == Cin) {
                    cin = 0;
                    t++;
                }
            }
        }
    }
}