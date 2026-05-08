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

    if (sel_val == 0) { // Chế độ Winograd Engine
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
    }
    else { // Chế độ Systolic Array
        demux_w_sys_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic_w.write(in_weight_stream.read());
        }
    }
}

void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<SysWindow> &out_systolic,
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<DemuxDataConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    DemuxDataConfig cfg = config_stream.read();
    int total_iters     = cfg.num_tiles * cfg.Cin;
    ap_uint<2> sel_val  = mode_stream.read();

    demux_flat_loop: for (int iter = 0; iter < total_iters; iter++) {
        #pragma HLS PIPELINE II=1
        
        Tile4x4 tile = in_stream.read();
        
        if (sel_val == 0) {
            out_winograd.write(tile);
        }
        else {
            SysWindow sys_win;
            #pragma HLS ARRAY_PARTITION variable=sys_win.data complete
            
            if (sel_val == 1) { 
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
                for(int z=1; z<KERNEL_SIZE; z++) {
                    #pragma HLS UNROLL
                    sys_win.data[z] = 0;
                }
            }
            out_systolic.write(sys_win);
        }
    }
}