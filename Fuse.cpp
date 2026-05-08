#include "Fuse.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in,
    hls::stream<fuse_vec_out_t>& fuse_out,
    hls::stream<ap_int<8>>& bias_array,
    hls::stream<FuseConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    FuseConfig config = config_stream.read();
    int total_packets = config.total_packets; 
    int channels = config.channel_limit;
    ap_int<8> shift_val = config.requant_shift_val;

    // 1. KHỞI TẠO LOCAL BIAS CACHE
    ap_int<8> local_bias[MAX_CHANNELS];
    #pragma HLS BIND_STORAGE variable=local_bias type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=local_bias cyclic factor=16 dim=1

    // 2. NẠP BIAS TỪ DRAM VÀO BRAM NỘI BỘ 
    load_bias_loop: for (int i = 0; i < channels; i++) {
        #pragma HLS PIPELINE II=1
        local_bias[i] = bias_array.read();
    }
    ap_uint<16> c = 0; 

    process_stream: for (int p = 0; p < total_packets; p++) {
        #pragma HLS PIPELINE II=1
        
        fuse_vec_in_t in_pkt = conv_in.read();
        fuse_vec_in_t res_pkt;
        
        if (config.has_residual) {
            res_pkt = residual_in.read();
        }

        fuse_vec_out_t out_pkt;

        process_elements: for (int i = 0; i < FUSE_PARALLEL_SIZE; i++) {
            #pragma HLS UNROLL 
            ap_uint<16> current_c; 

            if(config.mode == 0) {
                current_c = (c + (i / 4)) % channels;
            } else {
                current_c = (c + i) % channels;
            }

            // --- BƯỚC 1: Đọc Psum (Int32) ---
            ap_int<32> acc = in_pkt.data[i];

            // --- BƯỚC 2: Cộng Bias ---
            ap_int<32> with_bias = acc + local_bias[current_c];

            // --- Bước 3: Activation (Leaky ReLU 0.1015625) ---
            ap_int<32> relu_mult = with_bias * 13;
            #pragma HLS BIND_OP variable=relu_mult op=mul impl=dsp
            ap_int<32> activated = (with_bias < 0) ? (ap_int<32>)(relu_mult >> 7) : with_bias;

            // --- BƯỚC 4: Requantize (Scale & Shift) ---
            ap_int<32> shifted = activated >> shift_val;

            ap_int<8> requantized = (shifted > 127) ? (ap_int<8>)127 : 
                                    (shifted < -128) ? (ap_int<8>)-128 : (ap_int<8>)shifted;

            // --- BƯỚC 5: Cộng Residual ---
            ap_int<8> final_pixel = requantized;
            
            if (config.has_residual) {
                ap_int<9> add_result = (ap_int<9>)requantized + (ap_int<9>)res_pkt.data[i];
                
                final_pixel = (add_result > 127) ? (ap_int<8>)127 : 
                              (add_result < -128) ? (ap_int<8>)-128 : (ap_int<8>)add_result;
            }

            // --- BƯỚC 6: Channel Masking / Slice ---
            if (current_c < config.channel_limit) {
                out_pkt.data[i] = final_pixel;
            } else {
                out_pkt.data[i] = 0;
            }
        }
        
        fuse_out.write(out_pkt);

        if (config.mode == 0) {
            c += (FUSE_PARALLEL_SIZE / 4);
        } else {
            c += FUSE_PARALLEL_SIZE;
        }        
        while (c >= channels) {
            c -= channels;
        }
    }
}

void compute_to_fuse_serializer(
    hls::stream<psum_block_t>& systolic_in,
    hls::stream<Tile2x2>& winograd_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<ap_uint<2>>& mode_stream,
    hls::stream<SerializerConfig>& config_stream
) {
    #pragma HLS INLINE off

    SerializerConfig config = config_stream.read();
    unsigned int total_elements = config.total_elements;
    ap_uint<2> sel_val = mode_stream.read();

    if (sel_val == 0) {
        fuse_vec_in_t out_pkt;
        #pragma HLS ARRAY_PARTITION variable=out_pkt.data type=complete dim=1

        winograd_pack_loop: for (unsigned int i = 0; i < total_elements; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 
            
            Tile2x2 tile = winograd_in.read(); 
            int w = i % 4; 

            out_pkt.data[w*4 + 0] = tile.data[0][0];
            out_pkt.data[w*4 + 1] = tile.data[0][1];
            out_pkt.data[w*4 + 2] = tile.data[1][0];
            out_pkt.data[w*4 + 3] = tile.data[1][1];

            if (w == 3 || i == (total_elements - 1)) {
                fuse_out.write(out_pkt);
            }
        }
    } else {
        systolic_pack_loop: for (unsigned int i = 0; i < total_elements; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 
            
            psum_block_t block = systolic_in.read(); 
            fuse_vec_in_t out_pkt;
            #pragma HLS ARRAY_PARTITION variable=out_pkt.data type=complete dim=1
            
            systolic_unpack: for (int j = 0; j < FUSE_PARALLEL_SIZE; j++) {
                #pragma HLS UNROLL
                out_pkt.data[j] = block.data[j];
            }
            
            fuse_out.write(out_pkt);
        }
    }
}