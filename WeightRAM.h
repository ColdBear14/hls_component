#ifndef WEIGHTRAM_H
#define WEIGHTRAM_H

#include "global.h"

void load_weights(
    hls::stream<axi_word_t> &in_weights,
    weight_t bram[NUM_BANKS][BANK_DEPTH],
    int num_weights_per_tile
);

void feed_weights(
    weight_t bram[NUM_BANKS][BANK_DEPTH],
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles, 
    int num_weights_per_tile
);

void weight_controller_core(
    hls::stream<axi_word_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles,
    int num_weights_per_tile
);

// 3. TOP MODULE
void weight_controller_top(
    hls::stream<axi_word_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    hls::stream<WeightRamConfig> &config_stream
);

#endif