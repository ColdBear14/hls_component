#pragma once
#ifndef GLOBAL_H
#define GLOBAL_H

#include <ap_int.h>
#include <hls_stream.h>

// ==========================================================
// 1. SYSTEM LIMITS & CONSTANTS
// ==========================================================
#define MAX_CHANNELS         512
#define MAX_CIN              256    
#define MAX_COUT             256    
#define MAX_COUT_BLOCKS      16     // 256 / 16
#define MAX_LINE_BUFFER_SIZE 32000
#define ARRAY_SIZE           16     // Systolic Array Size (16x16)
#define MAX_COUT_TILE        64

// Configuration: Weight RAM
#define NUM_BANKS            16
#define BANK_DEPTH           8192 

// Configuration: Winograd & Fuse
#define COUT_UNROLL          4      // Parallel channels in Winograd
#define FUSE_PARALLEL_SIZE   16     // Parallel elements in Fuse block

// ==========================================================
// 2. BASIC DATA TYPES
// ==========================================================
typedef ap_uint<128> axi_word_t;      
typedef ap_int<8>    pixel_t;       
typedef ap_int<8>    weight_t;      
typedef ap_int<32>   data_t;        
typedef ap_int<32>   psum_t;        
typedef ap_uint<128> weight_mat_t;  

// ==========================================================
// 3. ENGINE SPECIFIC STRUCTURES
// ==========================================================
struct psum_block_t { psum_t data[ARRAY_SIZE]; };
struct SysWindow    { pixel_t data[9]; };

struct Tile4x4      { data_t data[4][4]; };
struct Tile3x3      { data_t data[3][3]; };
struct Tile2x2      { data_t data[2][2]; };

struct WeightBundle { Tile4x4 weights[COUT_UNROLL]; };
struct WindowVec16  { pixel_t p[4][4][16]; };
struct fuse_vec_in_t{ ap_int<32> data[FUSE_PARALLEL_SIZE]; };

struct axi_stream_out_t {
    axi_word_t data;       
    ap_uint<16> keep;      
    ap_uint<1> last;       
};

// ==========================================================
// 4. LAYER DESCRIPTOR & CONFIGURATIONS
// ==========================================================
struct LayerDescriptor {
    ap_uint<2>  type;              
    ap_uint<16> W;                 
    ap_uint<16> H;                 
    ap_uint<16> Cin;               
    ap_uint<16> Cout;              
    ap_uint<2>  kernel_size;       
    ap_uint<2>  stride;            
    ap_uint<2>  pad;               
    ap_uint<2>  has_residual;      
    ap_uint<8>  requant_shift_val; 
};

struct LineBufferConfig  { int W, H, Cin, pad; };
struct WeightRamConfig   { int num_spatial_tiles, num_weights_per_tile; };
struct DemuxDataConfig   { int num_tiles, Cin; };
struct DemuxWeightConfig { int num_weight_vectors; };
struct SystolicConfig    { int Cin, Cout, tiles_per_ch; };
struct WinogradConfig    { int num_tiles, Cin, Cout; };
struct SerializerConfig  { int total_elements; };

struct FuseConfig {
    bool has_residual;
    int  channel_limit;
    int  feature_map_width;
    int  feature_map_height;
    ap_uint<8> requant_shift_val;
    ap_uint<2> mode;
    int  total_packets;
};

#endif // GLOBAL_H