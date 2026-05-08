#ifndef SubLineBuffer_H
#define SubLineBuffer_H

#include "global.h"

void sub_line_buffer_top(
    hls::stream<pixel_t> &residual_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<FuseConfig>& config_stream
);

#endif // SubLineBuffer_H