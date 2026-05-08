#ifndef HELPER_H
#define HELPER_H

#include "global.h"

#include "DataRouter.h"
#include "WinogradEngine.h"
#include "LineBuffer.h"
#include "SubLineBuffer.h"
#include "WeightRAM.h"
#include "SystolicEngine.h"
#include "Fuse.h"
#include "Scheduler.h"
#include "TopSystem.h"

using namespace std;

void golden_winograd(Tile4x4 d, Tile3x3 g, Tile2x2 &y_ref);
void weight_transform(Tile3x3 g, Tile4x4 &V);

void fill_pixel_stream(hls::stream<pixel_t>& stream, int width, int height, int Cin);
void fill_weight_stream(hls::stream<weight_t>& stream, int kernel_size, int Cin, int Cout);
void fill_Wino_weight_stream(weight_t* weight_mem, int kernel_size, int Cin, int Cout, hls::stream<weight_t>& output);

void standard_convolution_golden(
    int W, int H, int Cin, int Cout, int Tiles_X, int Tiles_Y,
    pixel_t* img,  
    weight_t* weight, // Changed to flat pointer
    data_t* output_gold // Changed to flat pointer
);

void golden_conv_full(
    int W, int H, int Cin, int Cout, int K, int stride, int pad,
    pixel_t* img, weight_t* weight, ap_int<8>* bias, pixel_t* residual,
    bool has_residual, int requant_shift, pixel_t* final_out
);

#endif