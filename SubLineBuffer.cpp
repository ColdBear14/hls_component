#include "SubLineBuffer.h"

void sub_line_buffer_top(
    hls::stream<pixel_t> &residual_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<FuseConfig>& residual_config_stream
) {
    #pragma HLS INLINE off

    FuseConfig residual_config = residual_config_stream.read();

    if (!residual_config.has_residual) return;

    int channels = residual_config.channel_limit;
    int width = residual_config.feature_map_width;
    int height = residual_config.feature_map_height;
    ap_uint<2> mode = residual_config.mode;

    if (mode == 0) {
        // ================================
        // Winograd path (unchanged)
        // ================================
        int tiles_x = width / 2;
        int tiles_y = height / 2;
        int total_tiles = tiles_x * tiles_y;

        wino_tile_loop: for (int t = 0; t < total_tiles; t++) {
            fuse_vec_in_t pkt;
            int w = 0;
            
            wino_ch_loop: for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE II=4 
                
                pkt.data[w*4 + 0] = residual_in.read();
                pkt.data[w*4 + 1] = residual_in.read();
                pkt.data[w*4 + 2] = residual_in.read();
                pkt.data[w*4 + 3] = residual_in.read();
                
                w++;
                bool is_last_ch = (c == channels - 1);
                
                if (w == 4 || is_last_ch) {
                    if (is_last_ch && w < 4) {
                        for (int fill = w * 4; fill < FUSE_PARALLEL_SIZE; fill++) {
                            #pragma HLS UNROLL
                            pkt.data[fill] = 0;
                        }
                    }
                    
                    fuse_out.write(pkt);
                    w = 0;
                }
            }
        }
    } else {
        // Systolic path (CORRECTED)
        int cout_blocks = (channels + 15) / 16;
        int total_pixels = width * height;
        int sys_tiles = (total_pixels + 15) / 16;

        // FIX 1: Removed 'static' to prevent inter-invocation WAW/RAW dependencies.
        pixel_t tile_buffer[16][16];
        #pragma HLS BIND_STORAGE variable=tile_buffer type=ram_2p impl=bram
        #pragma HLS ARRAY_PARTITION variable=tile_buffer complete dim=2
        
        // FIX 2: Explicitly tell HLS to ignore false loop-carried dependencies on this buffer.
        #pragma HLS DEPENDENCE variable=tile_buffer type=inter false

        sys_tile_loop: for (int t = 0; t < sys_tiles; t++) {
            int spatial_points = (t == sys_tiles - 1 && total_pixels % 16 != 0) ?
                                 (total_pixels % 16) : 16;

            sys_cb_loop: for (int cb = 0; cb < cout_blocks; cb++) {
                
                // --- Read all 16x16 values for this (tile, cout block) ---
                read_spatial: for (int r = 0; r < 16; r++) {
                    
                    // FIX 3: Pipeline the INNER loop instead of unrolling it.
                    read_ch: for (int c = 0; c < 16; c++) {
                        #pragma HLS PIPELINE II=1
                        
                        pixel_t in_val = residual_in.read(); 
                        
                        if (r < spatial_points) {
                            tile_buffer[r][c] = in_val;
                        } else {
                            // For padded spatial points, fill with zero
                            tile_buffer[r][c] = 0;
                        }
                    }
                }

                // --- Output packets in REVERSE spatial order (15 -> 0) ---
                output_rev_spatial: for (int r_out = 0; r_out < 16; r_out++) {
                    #pragma HLS PIPELINE II=1
                    fuse_vec_in_t pkt;
                    int src_r = 15 - r_out;   // reverse index

                    for (int c = 0; c < 16; c++) {
                        #pragma HLS UNROLL
                        if (src_r < spatial_points) {
                            pkt.data[c] = tile_buffer[src_r][c];
                        } else {
                            pkt.data[c] = 0;
                        }
                    }
                    fuse_out.write(pkt);
                }
            }
        }
    }
}