#ifndef DATAROUTER_H
#define DATAROUTER_H

#include "global.h"

void weight_demux(
    hls::stream<weight_mat_t> &in_weight_stream, 
    hls::stream<ap_uint<2>>& mode_stream,                    
    hls::stream<weight_mat_t> &out_systolic_w,   
    hls::stream<Tile4x4> &out_winograd_w,        
    hls::stream<DemuxWeightConfig>& config_stream                
);

void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<SysWindow> &out_systolic,  
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<DemuxDataConfig>& config_stream
);

#endif // DATAROUTER_H