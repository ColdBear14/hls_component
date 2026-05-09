#include "TopSystem.h"

void read_axilite_bias(ap_int<8> bias_array[MAX_CHANNELS], hls::stream<ap_int<8>>& out_stream, int channels) {
    read_bias_loop: for (int i = 0; i < channels; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(bias_array[i]);
    }
}

void cnn_accelerator_top(
    hls::stream<pixel_t>&        pixels_in_stream,
    hls::stream<pixel_t>&        residual_in_stream,     
    hls::stream<weight_t>&       weights_in_stream,
    hls::stream<axi_dma_out_t>& fuse_out_stream,
    ap_int<8>                    bias_array[MAX_CHANNELS],
    LayerDescriptor              descriptor,
    bool                         start_accel
) {
    // --- Interfaces ---
    #pragma HLS INTERFACE axis port=pixels_in_stream
    #pragma HLS INTERFACE axis port=residual_in_stream
    #pragma HLS INTERFACE axis port=weights_in_stream
    #pragma HLS INTERFACE axis port=fuse_out_stream

    #pragma HLS INTERFACE s_axilite port=bias_array  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=descriptor  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=start_accel bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return      bundle=CTRL

    #pragma HLS DATAFLOW

    // --- 1. CONFIG STREAMS ---
    hls::stream<ap_uint<2>> mode_lb_stream, mode_ddemux_stream, mode_wdemux_stream;
    hls::stream<ap_uint<2>> mode_sys_stream, mode_wino_stream, mode_serializer_stream;
    
    #pragma HLS stream variable=mode_lb_stream         depth=2
    #pragma HLS stream variable=mode_ddemux_stream     depth=2
    #pragma HLS stream variable=mode_wdemux_stream     depth=2
    #pragma HLS stream variable=mode_sys_stream        depth=2
    #pragma HLS stream variable=mode_wino_stream       depth=2
    #pragma HLS stream variable=mode_serializer_stream depth=2
    
    hls::stream<LineBufferConfig>  lb_config_stream;
    hls::stream<WeightRamConfig>   wr_config_stream;
    hls::stream<DemuxDataConfig>   ddemux_config_stream;
    hls::stream<DemuxWeightConfig> wdemux_config_stream;
    hls::stream<SystolicConfig>    sys_config_stream;
    hls::stream<WinogradConfig>    wino_config_stream;
    hls::stream<SerializerConfig>  serializer_config_stream;
    hls::stream<FuseConfig>        fuse_config_stream, slb_config_stream;

    #pragma HLS stream variable=lb_config_stream depth=2
    #pragma HLS stream variable=wr_config_stream depth=2

    // --- 2. DATA PATH STREAMS ---
    hls::stream<Tile4x4>             lb_to_router_stream("lb_to_router");
    hls::stream<systolic_data_t>     router_to_sys_data("router_to_sys_data");
    hls::stream<Tile4x4>             router_to_wino_data("router_to_wino_data");
    hls::stream<psum_block_t>        sys_to_mux_stream("sys_to_mux_stream");
    hls::stream<Tile2x2>             wino_to_mux_stream("wino_to_mux_stream");
    hls::stream<fuse_vec_in_t>       mux_to_fuse_stream("mux_to_fuse_stream");
    hls::stream<fuse_vec_in_t>       residual_to_fuse_stream("residual_to_fuse_stream");
    hls::stream<ap_int<8>>           internal_bias_stream("internal_bias_stream");

    #pragma HLS stream variable=router_to_sys_data      depth=4
    #pragma HLS stream variable=wino_to_mux_stream      depth=4
    #pragma HLS stream variable=internal_bias_stream    depth=4
    #pragma HLS stream variable=residual_to_fuse_stream depth=4

    
    // --- 3. WEIGHT PATH STREAMS ---
    hls::stream<weight_mat_t> weight_ram_to_router_stream("weight_ram_to_router");
    hls::stream<weight_mat_t> router_to_sys_weight("router_to_sys_weight");
    hls::stream<Tile4x4>      router_to_wino_weight("router_to_wino_weight");

    #pragma HLS stream variable=weight_ram_to_router_stream depth=4
    #pragma HLS stream variable=router_to_sys_weight        depth=4

    #pragma HLS stream variable=router_to_wino_weight   depth=4
    #pragma HLS BIND_STORAGE variable=router_to_wino_weight type=fifo impl=srl

    #pragma HLS stream variable=lb_to_router_stream     depth=4
    #pragma HLS BIND_STORAGE variable=lb_to_router_stream type=fifo impl=srl

    #pragma HLS stream variable=router_to_wino_data     depth=4
    #pragma HLS BIND_STORAGE variable=router_to_wino_data type=fifo impl=srl

    #pragma HLS stream variable=sys_to_mux_stream       depth=4
    #pragma HLS BIND_STORAGE variable=sys_to_mux_stream type=fifo impl=srl

    #pragma HLS stream variable=mux_to_fuse_stream      depth=4
    #pragma HLS BIND_STORAGE variable=mux_to_fuse_stream type=fifo impl=srl



    // --- 4. MODULE INSTANTIATIONS (DATAFLOW) ---
    read_axilite_bias(bias_array, internal_bias_stream, descriptor.Cout);

    controller_top(descriptor, start_accel, mode_sys_stream, mode_wino_stream, mode_lb_stream, 
                   mode_ddemux_stream, mode_wdemux_stream, mode_serializer_stream,
                   lb_config_stream, wr_config_stream, ddemux_config_stream, wdemux_config_stream,
                   sys_config_stream, wino_config_stream, serializer_config_stream, 
                   fuse_config_stream, slb_config_stream);

    line_buffer(pixels_in_stream, lb_to_router_stream, mode_lb_stream, lb_config_stream);

    weight_controller_top(weights_in_stream, weight_ram_to_router_stream, wr_config_stream);

    data_demux(lb_to_router_stream, mode_ddemux_stream, router_to_sys_data, 
               router_to_wino_data, ddemux_config_stream);

    weight_demux(weight_ram_to_router_stream, mode_wdemux_stream, router_to_sys_weight, 
                 router_to_wino_weight, wdemux_config_stream);

    winograd_engine_top(router_to_wino_data, router_to_wino_weight, wino_to_mux_stream, 
                        mode_wino_stream, wino_config_stream);

    systolic_engine(router_to_sys_data, router_to_sys_weight, sys_to_mux_stream, 
                    mode_sys_stream, sys_config_stream);

    sub_line_buffer_top(residual_in_stream, residual_to_fuse_stream, slb_config_stream);
    
    compute_to_fuse_serializer(sys_to_mux_stream, wino_to_mux_stream, mux_to_fuse_stream,
                               mode_serializer_stream, serializer_config_stream);

    fuse_post_conv(mux_to_fuse_stream, residual_to_fuse_stream, fuse_out_stream, 
                   internal_bias_stream, fuse_config_stream);
}