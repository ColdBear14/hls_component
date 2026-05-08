#ifndef GLOBAL_H
#define GLOBAL_H

#include <ap_int.h>
#include <hls_stream.h>

// ==========================================================
// 1. SYSTEM CONFIGURATION
// ==========================================================
#define MAX_CHANNELS         512
#define MAX_WIDTH            642    // 640 + 2 (padding)
#define MAX_CIN              256    // Max in_channels in SPPF
#define MAX_COUT             256    // Max out_channels in deep Conv
#define MAX_COUT_BLOCKS      16     // 256 / 16
#define MAX_TILES            1024 
#define MAX_LINE_BUFFER_SIZE 32000

// ==========================================================
// 2. BASIC DATA TYPES & LIMITS
// ==========================================================
typedef ap_int<8>  pixel_t;     // Pixel / activation 8-bit
typedef ap_int<8>  weight_t;    // Weight 8-bit
typedef ap_int<32> data_t;      // Intermediate / accumulator data
typedef ap_int<32> psum_t;      // Partial sum for MAC blocks

#define MAX_INT8 127
#define MIN_INT8 -128
#define LEAKY_RELU_MULT  13
#define LEAKY_RELU_SHIFT 7

// ==========================================================
// 3. SYSTOLIC ARRAY & WEIGHT RAM
// ==========================================================
const int ARRAY_SIZE  = 16;   // 16x16 PE array
const int KERNEL_SIZE = 9;
const int NUM_BANKS   = 16;
const int BANK_DEPTH  = 8096; // 512 channels * 16 tiles max

typedef ap_uint<128> weight_mat_t;

struct psum_block_t { psum_t data[ARRAY_SIZE]; };
struct SysWindow    { pixel_t data[KERNEL_SIZE]; };

// ==========================================================
// 4. WINOGRAD TRANSFORM STRUCTURES
// ==========================================================
struct Tile16x16 { data_t data[16][16]; };
struct Tile4x4   { data_t data[4][4]; };
struct Tile3x3   { data_t data[3][3]; };
struct Tile2x2   { data_t data[2][2]; };

// ==========================================================
// 5. FUSE MODULE
// ==========================================================
#define FUSE_PARALLEL_SIZE 16

struct fuse_vec_in_t  { ap_int<32> data[FUSE_PARALLEL_SIZE]; };

struct axi_dma_out_t {
    ap_int<8>   data[FUSE_PARALLEL_SIZE]; // TDATA 128-bit (16 pixel)
    ap_uint<16> keep;                     // TKEEP 16-bit (1 bit cho mỗi byte)
    ap_uint<1>  last;                     // TLAST 1-bit (Cờ báo kết thúc)
};

// ==========================================================
// 6. LAYER DESCRIPTOR & CONFIG STRUCTS
// ==========================================================
struct LayerDescriptor {
    ap_uint<2>  type;              // 0: Conv2D
    ap_uint<16> W;                 // Feature map width
    ap_uint<16> H;                 // Feature map height
    ap_uint<16> Cin;               // Input channels
    ap_uint<16> Cout;              // Output channels
    ap_uint<2>  kernel_size;       // 1: 1x1, 3: 3x3
    ap_uint<2>  stride;            // Stride (1 or 2)
    ap_uint<2>  pad;               // Padding (0: no, 1: pad 1 pixel)
    ap_uint<2>  has_residual;      // Skip connection flag
    ap_uint<8>  requant_shift_val; // Requantize shift value
};

struct LineBufferConfig  { int W; int H; int Cin; int pad; };
struct WeightRamConfig   { int num_spatial_tiles; int num_weights_per_tile; };
struct DemuxDataConfig   { int num_tiles; int Cin; };
struct DemuxWeightConfig { int num_weight_vectors; };
struct SystolicConfig    { int Cin; int Cout; int tiles_per_ch; };
struct WinogradConfig    { int num_tiles; int Cin; int Cout; };
struct SerializerConfig  { int total_elements; };

struct FuseConfig {
    bool       has_residual;
    int        channel_limit;
    int        feature_map_width;
    int        feature_map_height;
    ap_uint<8> requant_shift_val;
    ap_uint<2> mode;
    int        total_packets;
};

#endif // GLOBAL_H