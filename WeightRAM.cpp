#include "WeightRAM.h"

// 1. Hàm nạp Weight từ AXI vào BRAM
void load_weights(
    hls::stream<axi_word_t> &in_weights,
    weight_t bram[NUM_BANKS][BANK_DEPTH],
    int num_weights_per_tile
) {
    // 1 Word 128-bit chứa đúng 16 weights. 
    for (int w = 0; w < num_weights_per_tile; w++) {
        #pragma HLS PIPELINE II=1
        axi_word_t word = in_weights.read();
        for (int i = 0; i < NUM_BANKS; i++) {
            #pragma HLS UNROLL
            bram[i][w] = (weight_t)word.range(i * 8 + 7, i * 8);
        }
    }
}

// 2. Hàm đẩy Weight từ BRAM ra Engine (Có cơ chế Reuse)
void feed_weights(
    weight_t bram[NUM_BANKS][BANK_DEPTH],
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles, 
    int num_weights_per_tile
) {
    reuse_loop: for (int t = 0; t < num_spatial_tiles; t++) {
        feed_loop: for (int w = 0; w < num_weights_per_tile; w++) {
            #pragma HLS PIPELINE II=1
            weight_mat_t temp_mat;
            
            for (int i = 0; i < NUM_BANKS; i++) {
                #pragma HLS UNROLL
                temp_mat.range(i * 8 + 7, i * 8) = bram[i][w];
            }
            out_weight_stream.write(temp_mat);
        }
    }
}

// HÀM CORE: Đóng gói vùng Dataflow. 
void weight_controller_core(
    hls::stream<axi_word_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles,
    int num_weights_per_tile
) {
    #pragma HLS DATAFLOW
    weight_t local_bram[NUM_BANKS][BANK_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=local_bram complete dim=1
    #pragma HLS BIND_STORAGE variable=local_bram type=ram_2p impl=uram

    load_weights(in_weights, local_bram, num_weights_per_tile);
    feed_weights(local_bram, out_weight_stream, num_spatial_tiles, num_weights_per_tile);
}

// 3. TOP MODULE
void weight_controller_top(
    hls::stream<axi_word_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    hls::stream<WeightRamConfig> &config_stream
) {
    WeightRamConfig cfg = config_stream.read();
    weight_controller_core(in_weights, out_weight_stream, cfg.num_spatial_tiles, cfg.num_weights_per_tile);
}