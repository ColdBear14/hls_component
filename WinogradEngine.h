#ifndef WINOGRADENGINE_H
#define WINOGRADENGINE_H

#include "global.h"

void input_transform(Tile4x4 d, Tile4x4 &U);

void ewmm(Tile4x4 U, Tile4x4 V, Tile4x4 &M);

void output_transform(Tile4x4 M, Tile2x2 &Y);

void winograd_engine_top(
    hls::stream<Tile4x4> &in_tile_stream,  
    hls::stream<WeightBundle> &weight_v_stream, 
    hls::stream<Tile2x2> &out_tile_stream,  
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<WinogradConfig> &config_stream // Thay thế 3 luồng num_tiles, cin, cout
);

#endif