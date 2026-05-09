#ifndef SYSTOLICENGINE_H
#define SYSTOLICENGINE_H

#include "global.h"

using namespace hls;

void systolic_engine(
    hls::stream<systolic_data_t> &staggered_pixels_in, 
    hls::stream<weight_mat_t> &staggered_weights_in, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<SystolicConfig> &config_stream  
);

#endif