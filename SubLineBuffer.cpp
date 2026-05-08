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
        // CHẾ ĐỘ WINOGRAD (Nhóm 4 pixels / 1 kênh)
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
        // CHẾ ĐỘ SYSTOLIC (Nhóm 16 kênh / 1 pixel)
        int cout_blocks = (channels + 15) / 16;
        int total_pixels = width * height;
        
        int total_reads = total_pixels * cout_blocks * 16; 

        fuse_vec_in_t pkt;
        int pack_idx = 0;

        sys_packing_loop: for (int i = 0; i < total_reads; i++) {
            #pragma HLS PIPELINE II=1
            
            pkt.data[pack_idx] = residual_in.read();
            pack_idx++;

            if (pack_idx == FUSE_PARALLEL_SIZE) {
                fuse_out.write(pkt);
                pack_idx = 0;
            }
        }
    }
}