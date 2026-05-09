#include "DataRouter.h"

void weight_demux(
    hls::stream<weight_mat_t> &in_weight_stream, 
    hls::stream<ap_uint<2>>& mode_stream,                    
    hls::stream<weight_mat_t> &out_systolic_w,   
    hls::stream<Tile4x4> &out_winograd_w,        
    hls::stream<DemuxWeightConfig>& config_stream                
) {
    #pragma HLS INLINE off
    DemuxWeightConfig cfg = config_stream.read();
    int num_weight_vectors = cfg.num_weight_vectors;
    ap_uint<2> sel_val = mode_stream.read();

    if (sel_val == 0) { 
        demux_w_wino_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            weight_mat_t w_vec = in_weight_stream.read();
            Tile4x4 temp_w_tile;
            unpack_row: for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                unpack_col: for (int c = 0; c < 4; c++) {
                    #pragma HLS UNROLL
                    int idx = r * 4 + c;
                    temp_w_tile.data[r][c] = (ap_int<8>)w_vec.range(idx * 8 + 7, idx * 8); 
                }
            }
            out_winograd_w.write(temp_w_tile);
        }
    } else { 
        demux_w_sys_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic_w.write(in_weight_stream.read());
        }
    }
}

void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<systolic_data_t> &out_systolic,
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<DemuxDataConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    DemuxDataConfig cfg = config_stream.read();
    ap_uint<2> sel_val  = mode_stream.read();

    if (sel_val == 0) { // Chế độ Winograd
        int total_iters = cfg.num_tiles * cfg.Cin;
        demux_wino_loop: for (int iter = 0; iter < total_iters; iter++) {
            #pragma HLS PIPELINE II=1
            out_winograd.write(in_stream.read());
        }
    } 
    else { // Chế độ Systolic Array
        // TỐI ƯU: Gộp 16 pixels vào 1 vector 128-bit ngay từ đầu. 
        // Chỉ cần chia mảng theo chiều Kernel (9-way), cực kỳ tiết kiệm BRAM.
        static systolic_data_t packed_buffer[MAX_CIN][KERNEL_SIZE];
        #pragma HLS ARRAY_PARTITION variable=packed_buffer complete dim=2
        #pragma HLS BIND_STORAGE variable=packed_buffer type=ram_2p impl=bram

        int sys_tiles = (cfg.num_tiles + 15) / 16;
        
        demux_sys_tile_loop: for (int t = 0; t < sys_tiles; t++) {
            int spatial_points = (t == sys_tiles - 1 && cfg.num_tiles % 16 != 0) ? (cfg.num_tiles % 16) : 16;
            
            // --- PHA 1: ĐỌC VÀ ĐÓNG GÓI TRỰC TIẾP (Read-Modify-Write) ---
            read_spatial_loop: for (int p = 0; p < spatial_points; p++) {
                read_channel_loop: for (int c = 0; c < cfg.Cin; c++) {
                    #pragma HLS PIPELINE II=1
                    // Bỏ qua cảnh báo phụ thuộc vòng lặp vì p là outer loop, c đảm bảo khoảng cách an toàn
                    #pragma HLS DEPENDENCE variable=packed_buffer type=inter false
                    
                    Tile4x4 tile = in_stream.read();
                    
                    pixel_t temp_window[KERNEL_SIZE];
                    #pragma HLS ARRAY_PARTITION variable=temp_window complete
                    
                    if (sel_val == 1) { 
                        int idx = 0;
                        for(int r=0; r<3; r++) {
                            for(int col=0; col<3; col++) temp_window[idx++] = tile.data[r][col]; 
                        }
                    } else if (sel_val == 2) { 
                        temp_window[0] = tile.data[0][0]; 
                        for(int z=1; z<KERNEL_SIZE; z++) temp_window[z] = 0;
                    }

                    for(int k=0; k<KERNEL_SIZE; k++) {
                        #pragma HLS UNROLL
                        // Mẹo: Khi p=0, gán toàn bộ 128-bit bằng 0 để dọn rác và xử lý zero-padding tự nhiên.
                        // Các p tiếp theo chỉ ghi đè vào byte tương ứng.
                        systolic_data_t val = (p == 0) ? (systolic_data_t)0 : packed_buffer[c][k];
                        val.range(p * 8 + 7, p * 8) = temp_window[k];
                        packed_buffer[c][k] = val;
                    }
                }
            }
            
            // --- PHA 2: XẢ RA SYSTOLIC ENGINE (Đơn giản hóa) ---
            int k_limit = (sel_val == 1) ? 9 : 1;
            write_channel_loop: for (int c = 0; c < cfg.Cin; c++) {
                write_kernel_loop: for (int k = 0; k < k_limit; k++) {
                    #pragma HLS PIPELINE II=1
                    // Dữ liệu đã được gộp sẵn và zero-padded ở Pha 1, chỉ việc đẩy ra stream
                    out_systolic.write(packed_buffer[c][k]);
                }
            }
        }
    }
}