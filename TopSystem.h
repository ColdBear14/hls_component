#ifndef TOPSYSTEM_H
#define TOPSYSTEM_H

#include "global.h"

#include "Scheduler.h"
#include "LineBuffer.h"
#include "DataRouter.h"
#include "SystolicEngine.h"
#include "WinogradEngine.h"
#include "Fuse.h"
#include "WeightRAM.h"
#include "SubLineBuffer.h"

void cnn_accelerator_top(
    hls::stream<pixel_t>& pixels_in_stream,       
    hls::stream<pixel_t>& residual_in_stream,     
    hls::stream<weight_t>& weights_in_stream,     
    hls::stream<axi_dma_out_t>& fuse_out_stream, 
    ap_int<8> bias_array[MAX_CHANNELS],
    
    // --- Giao tiếp AXI-Lite ---
    LayerDescriptor descriptor,
    bool start_accel
);

#endif // TOPSYSTEM_H