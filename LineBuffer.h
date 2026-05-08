#ifndef LINEBUFFER_H
#define LINEBUFFER_H

#include "global.h"

void line_buffer(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<Tile4x4> &out_tile_stream, 
    hls::stream<ap_uint<2>>& mode_stream,             
    hls::stream<LineBufferConfig>& config_stream
);

#endif