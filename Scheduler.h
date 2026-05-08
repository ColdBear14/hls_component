#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "global.h"

void controller_top(
    const LayerDescriptor& descriptor,
    bool start,
    
    // Engine Mode Select Streams
    hls::stream<ap_uint<2>>& mode_sys_stream,
    hls::stream<ap_uint<2>>& mode_wino_stream,
    hls::stream<ap_uint<2>>& mode_lb_stream,
    hls::stream<ap_uint<2>>& mode_ddemux_stream,
    hls::stream<ap_uint<2>>& mode_wdemux_stream,
    hls::stream<ap_uint<2>>& mode_serializer_stream,

    // Module Configuration Streams
    hls::stream<LineBufferConfig>& lb_cfg_stream,
    hls::stream<WeightRamConfig>& wr_cfg_stream,
    hls::stream<DemuxDataConfig>& ddemux_cfg_stream,
    hls::stream<DemuxWeightConfig>& wdemux_cfg_stream,
    hls::stream<SystolicConfig>& sys_cfg_stream,
    hls::stream<WinogradConfig>& wino_cfg_stream,
    hls::stream<SerializerConfig>& serializer_cfg_stream,
    hls::stream<FuseConfig>& fuse_cfg_stream,
    hls::stream<FuseConfig>& slb_cfg_stream
);

#endif // SCHEDULER_H