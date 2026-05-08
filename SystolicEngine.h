#ifndef SYSTOLICENGINE_H
#define SYSTOLICENGINE_H

#include "global.h"

using namespace hls;

void systolic_engine(
    hls::stream<SysWindow> &pixels_in,
    hls::stream<weight_mat_t> &weights_in, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<SystolicConfig> &config_stream // Thay thế 3 luồng int cấu hình
);

#endif