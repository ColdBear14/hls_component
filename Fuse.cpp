#include "Fuse.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in,
    hls::stream<axi_dma_out_t>& fuse_out,
    hls::stream<ap_int<8>>& bias_array,
    hls::stream<FuseConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    FuseConfig config = config_stream.read();
    int total_packets = config.total_packets; 
    int channels      = config.channel_limit;
    ap_int<8> shift_val = config.requant_shift_val;

    // --- 1. LOCAL BIAS CACHE ---
    ap_int<8> local_bias[MAX_CHANNELS];
    #pragma HLS BIND_STORAGE variable=local_bias type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=local_bias cyclic factor=16 dim=1

    // --- 2. LOAD BIAS FROM DRAM ---
    load_bias_loop: for (int i = 0; i < channels; i++) {
        #pragma HLS PIPELINE II=1
        local_bias[i] = bias_array.read();
    }
    
    ap_uint<16> c = 0; 

    process_stream: for (int p = 0; p < total_packets; p++) {
        #pragma HLS PIPELINE II=1
        
        fuse_vec_in_t in_pkt = conv_in.read();
        fuse_vec_in_t res_pkt;
        axi_dma_out_t out_pkt;
        
        if (config.has_residual) res_pkt = residual_in.read();

        out_pkt.keep = 0xFFFF;
        out_pkt.last = (p == total_packets - 1) ? 1 : 0;

        process_elements: for (int i = 0; i < FUSE_PARALLEL_SIZE; i++) {
            #pragma HLS UNROLL 
            
            ap_uint<16> current_c = (config.mode == 0) ? 
                                    (c + (i / 4)) % channels : 
                                    (c + i) % channels;

            // Step 1 & 2: Read Psum & Add Bias
            ap_int<32> acc       = in_pkt.data[i];
            ap_int<32> with_bias = acc + local_bias[current_c];

            // Step 3: Activation (Leaky ReLU)
            ap_int<32> relu_mult = with_bias * LEAKY_RELU_MULT;
            #pragma HLS BIND_OP variable=relu_mult op=mul impl=dsp
            
            ap_int<32> activated = (with_bias < 0) ? (ap_int<32>)(relu_mult >> LEAKY_RELU_SHIFT) : with_bias;

            // Step 4: Requantize (Scale & Shift)
            ap_int<32> shifted = activated >> shift_val;
            ap_int<8> requantized = (shifted > MAX_INT8) ? (ap_int<8>)MAX_INT8 : 
                                    (shifted < MIN_INT8) ? (ap_int<8>)MIN_INT8 : (ap_int<8>)shifted;

            // Step 5: Add Residual
            ap_int<8> final_pixel = requantized;
            
            if (config.has_residual) {
                ap_int<9> add_result = (ap_int<9>)requantized + (ap_int<9>)res_pkt.data[i];
                final_pixel = (add_result > MAX_INT8) ? (ap_int<8>)MAX_INT8 : 
                              (add_result < MIN_INT8) ? (ap_int<8>)MIN_INT8 : (ap_int<8>)add_result;
            }

            // Step 6: Channel Masking
            out_pkt.data[i] = (current_c < config.channel_limit) ? final_pixel : (ap_int<8>)0;
        }
        
        fuse_out.write(out_pkt);

        // Update channel counter
        c += (config.mode == 0) ? (FUSE_PARALLEL_SIZE / 4) : FUSE_PARALLEL_SIZE;
        if (c >= channels) c -= channels;
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