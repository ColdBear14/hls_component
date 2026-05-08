#ifndef FUSE_H
#define FUSE_H

#include "global.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in,
    hls::stream<axi_stream_out_t>& fuse_out,
    hls::stream<ap_int<8>>& bias_array,
    hls::stream<FuseConfig>& config_stream
);

void compute_to_fuse_serializer(
    hls::stream<psum_block_t>& systolic_in,
    hls::stream<Tile2x2>& winograd_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<ap_uint<2>>& mode_stream,
    hls::stream<SerializerConfig>& config_stream
);

#endif // FUSE_H