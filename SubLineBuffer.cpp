#include "SubLineBuffer.h"

void sub_line_buffer_top(
    hls::stream<axi_word_t> &residual_in,
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

    // BIẾN UNPACKER 128-bit
    ap_uint<4> read_idx = 0; 
    axi_word_t res_word = 0;

    if (mode == 0) {
        // CHẾ ĐỘ WINOGRAD (Nhóm 4 pixels / 1 kênh)
        int tiles_x = width / 2;
        int tiles_y = height / 2;
        int total_tiles = tiles_x * tiles_y;

        wino_tile_loop: for (int t = 0; t < total_tiles; t++) {
            fuse_vec_in_t pkt;
            int w = 0;
            
            wino_ch_loop: for (int c = 0; c < channels; c++) {
                // TỐI ƯU 1: ÉP PIPELINE II=1 thay vì II=4
                #pragma HLS PIPELINE II=1 
                
                // Đọc 4 pixels từ 128-bit stream
                if (read_idx == 0) {
                    res_word = residual_in.read();
                }
                
                for (int i = 0; i < 4; i++) {
                    pkt.data[w*4 + i] = (ap_int<8>)res_word.range(i * 8 + 7, i * 8);
                }
                res_word >>= 32;
                read_idx += 4;
                
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
            
            if (read_idx == 0) {
                res_word = residual_in.read();
            }
            
            // TỐI ƯU 3: Áp dụng Shift Register tương tự cho khối Systolic
            pkt.data[pack_idx] = (ap_int<8>)res_word.range(7, 0);
            res_word >>= 8;
            read_idx++;
            
            pack_idx++;

            if (pack_idx == FUSE_PARALLEL_SIZE) {
                fuse_out.write(pkt);
                pack_idx = 0;
            }
        }
    }
}