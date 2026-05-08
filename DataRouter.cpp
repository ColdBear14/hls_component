#include "DataRouter.h"

void weight_demux(
    hls::stream<weight_mat_t> &in_weight_stream, 
    hls::stream<ap_uint<2>>& mode_stream,                    
    hls::stream<weight_mat_t> &out_systolic_w,   
    hls::stream<WeightBundle> &out_winograd_w,  // Đổi Tile4x4 -> WeightBundle
    hls::stream<DemuxWeightConfig>& config_stream                
) {
    #pragma HLS INLINE off

    DemuxWeightConfig cfg = config_stream.read();
    int num_weight_vectors = cfg.num_weight_vectors;
    ap_uint<2> sel_val = mode_stream.read();

    if (sel_val == 0) { // Chế độ Winograd Engine
        // Đọc COUT_UNROLL (4) vector mỗi lần để đóng gói thành 1 Bundle
        demux_w_wino_loop: for (int i = 0; i < num_weight_vectors; i += COUT_UNROLL) {
            #pragma HLS PIPELINE II=4 
            
            WeightBundle bundle;

            // Gom 4 vector 128-bit từ stream đầu vào
            bundle_pack_loop: for (int u = 0; u < COUT_UNROLL; u++) {
                #pragma HLS UNROLL
                weight_mat_t w_vec = in_weight_stream.read();
                
                unpack_row: for (int r = 0; r < 4; r++) {
                    #pragma HLS UNROLL
                    unpack_col: for (int c = 0; c < 4; c++) {
                        #pragma HLS UNROLL
                        int idx = r * 4 + c;
                        bundle.weights[u].data[r][c] = (ap_int<8>)w_vec.range(idx * 8 + 7, idx * 8); 
                    }
                }
            }
            out_winograd_w.write(bundle);
        }
    }
    else { // Chế độ Systolic Array
        demux_w_sys_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic_w.write(in_weight_stream.read());
        }
    }
}

// Hàm data_demux giữ nguyên logic cũ
void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<SysWindow> &out_systolic,
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<DemuxDataConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    DemuxDataConfig cfg = config_stream.read();
    int num_tiles = cfg.num_tiles;
    int Cin = cfg.Cin;
    ap_uint<2> sel_val = mode_stream.read();
    
    int total_iters = num_tiles * Cin;

    demux_flat_loop: for (int iter = 0; iter < total_iters; iter++) {
        #pragma HLS PIPELINE II=1
        
        Tile4x4 tile = in_stream.read();
        
        if (sel_val == 0) {
            out_winograd.write(tile);
        }
        else {
            SysWindow sys_win;
            #pragma HLS ARRAY_PARTITION variable=sys_win.data complete
            
            if(sel_val == 1) { 
                int idx = 0;
                extract_3x3_row: for(int r=0; r<3; r++) {
                    #pragma HLS UNROLL
                    extract_3x3_col: for(int c=0; c<3; c++) {
                        #pragma HLS UNROLL
                        sys_win.data[idx++] = tile.data[r][c]; 
                    }
                }
            } 
            else if (sel_val == 2) { 
                sys_win.data[0] = tile.data[0][0]; 
                for(int z=1; z<9; z++) {
                    #pragma HLS UNROLL
                    sys_win.data[z] = 0;
                }
            }
            out_systolic.write(sys_win);
        }
    }
}