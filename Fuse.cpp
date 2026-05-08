#include "Fuse.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in,
    hls::stream<axi_stream_out_t>& fuse_out,     
    hls::stream<ap_int<8>>& bias_array,
    hls::stream<FuseConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    FuseConfig config = config_stream.read();
    int total_packets = config.total_packets; 
    int channels = config.channel_limit;
    ap_int<8> shift_val = config.requant_shift_val;

    int hw_channel_boundary = ((channels + 15) / 16) * 16;

    // 1. INITIALIZE LOCAL BIAS CACHE
    ap_int<8> local_bias[MAX_CHANNELS];
    #pragma HLS BIND_STORAGE variable=local_bias type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=local_bias cyclic factor=16 dim=1

    // 2. LOAD BIAS FROM DRAM
    load_bias_loop: for (int i = 0; i < channels; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512
        local_bias[i] = bias_array.read();
    }
    
    pad_bias_loop: for (int i = channels; i < hw_channel_boundary; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=0 max=15
        local_bias[i] = 0;
    }

    ap_uint<16> c = 0; 

    process_stream: for (int p = 0; p < total_packets; p++) {
        #pragma HLS PIPELINE II=1
        
        fuse_vec_in_t in_pkt = conv_in.read();
        fuse_vec_in_t res_pkt;
        
        if (config.has_residual) {
            res_pkt = residual_in.read();
        }

        axi_word_t out_word = 0;

        process_elements: for (int i = 0; i < FUSE_PARALLEL_SIZE; i++) {
            #pragma HLS UNROLL 
            ap_uint<16> current_c; 

            // HLS OPTIMIZATION: Avoid modulo (%) in unrolled loops
            if(config.mode == 0) {
                ap_uint<16> temp_c = c + (i / 4);
                current_c = (temp_c >= channels) ? (ap_uint<16>)(temp_c - channels) : temp_c;
            } else {
                current_c = c + i; 
            }

            // Step 1: Read Psum
            ap_int<32> acc = in_pkt.data[i];

            // Step 2: Add Bias
            ap_int<32> with_bias = acc + local_bias[current_c];

            // Step 3: Activation (Leaky ReLU ~0.1)
            ap_int<32> relu_mult = with_bias * 13;
            #pragma HLS BIND_OP variable=relu_mult op=mul impl=dsp
            ap_int<32> activated = (with_bias < 0) ? (ap_int<32>)(relu_mult >> 7) : with_bias;

            // Step 4: Requantize
            ap_int<32> shifted = activated >> shift_val;
            ap_int<8> requantized = (shifted > 127) ? (ap_int<8>)127 : 
                                    (shifted < -128) ? (ap_int<8>)-128 : (ap_int<8>)shifted;

            // Step 5: Add Residual
            ap_int<8> final_pixel = requantized;
            if (config.has_residual) {
                ap_int<9> add_result = (ap_int<9>)requantized + (ap_int<9>)((ap_int<8>)res_pkt.data[i]);
                final_pixel = (add_result > 127) ? (ap_int<8>)127 : 
                              (add_result < -128) ? (ap_int<8>)-128 : (ap_int<8>)add_result;
            }

            // Step 6: Channel Masking & Packing
            if (current_c < channels) {
                out_word.range(i * 8 + 7, i * 8) = (ap_uint<8>)final_pixel;
            } else {
                out_word.range(i * 8 + 7, i * 8) = 0;
            }
        }
        
        axi_stream_out_t dma_pkt;
        dma_pkt.data = out_word;
        dma_pkt.keep = 0xFFFF;  // 16 bytes (128-bit) đều hợp lệ -> 16 bit 1
        dma_pkt.last = (p == (total_packets - 1)) ? 1 : 0; // TLAST = 1 ở packet cuối
        
        fuse_out.write(dma_pkt);

        // Update Channel Index
        if (config.mode == 0) {
            c += (FUSE_PARALLEL_SIZE / 4);
            while (c >= channels) c -= channels;
        } else {
            c += FUSE_PARALLEL_SIZE;
            while (c >= hw_channel_boundary) c -= hw_channel_boundary;
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