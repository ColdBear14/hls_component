#include "TopSystem.h"

void read_axilite_bias(ap_int<8> bias_array[MAX_CHANNELS], hls::stream<ap_int<8>>& out_stream, int channels) {
    read_bias_loop: for (int i = 0; i < channels; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512
        out_stream.write(bias_array[i]);
    }
}

void cnn_accelerator_top(
    hls::stream<axi_word_t>& pixels_in_stream,
    hls::stream<axi_word_t>& residual_in_stream,
    hls::stream<axi_word_t>& weights_in_stream,
    hls::stream<axi_stream_out_t>& fuse_out_stream,
    ap_int<8> bias_array[MAX_CHANNELS],
    LayerDescriptor descriptor,
    bool start_accel
) {
    #pragma HLS INTERFACE axis port=pixels_in_stream
    #pragma HLS INTERFACE axis port=residual_in_stream
    #pragma HLS INTERFACE axis port=weights_in_stream
    #pragma HLS INTERFACE axis port=fuse_out_stream
    
    #pragma HLS INTERFACE s_axilite port=bias_array bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=descriptor bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=start_accel bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return      bundle=CTRL

    #pragma HLS DATAFLOW

    // --- STREAM CONFIGURATIONS ---
    hls::stream<ap_uint<2>> mode_streams[6];
    #pragma HLS STREAM variable=mode_streams depth=2

    hls::stream<LineBufferConfig>  lb_cfg;
    hls::stream<WeightRamConfig>   wr_cfg;
    hls::stream<DemuxDataConfig>   ddemux_cfg;
    hls::stream<DemuxWeightConfig> wdemux_cfg;
    hls::stream<SystolicConfig>    sys_cfg;
    hls::stream<WinogradConfig>    wino_cfg;
    hls::stream<SerializerConfig>  ser_cfg;
    hls::stream<FuseConfig>        fuse_cfg;
    hls::stream<FuseConfig>        slb_cfg;

    // --- DATA PATH STREAMS ---
    hls::stream<Tile4x4>       lb_to_router("lb_to_router");
    hls::stream<SysWindow>     router_to_sys_data("router_to_sys_data");
    hls::stream<Tile4x4>       router_to_wino_data("router_to_wino_data");
    hls::stream<psum_block_t>  sys_to_mux("sys_to_mux");
    hls::stream<Tile2x2>       wino_to_mux("wino_to_mux");
    hls::stream<fuse_vec_in_t> mux_to_fuse("mux_to_fuse");
    hls::stream<fuse_vec_in_t> res_to_fuse("res_to_fuse");
    hls::stream<ap_int<8>>     internal_bias("internal_bias");

    #pragma HLS STREAM variable=lb_to_router       depth=16
    #pragma HLS STREAM variable=router_to_wino_data depth=16
    #pragma HLS STREAM variable=router_to_sys_data  depth=16
    #pragma HLS STREAM variable=sys_to_mux          depth=16
    #pragma HLS STREAM variable=wino_to_mux         depth=16
    #pragma HLS STREAM variable=mux_to_fuse         depth=16
    #pragma HLS STREAM variable=internal_bias       depth=16
    #pragma HLS STREAM variable=res_to_fuse         depth=64

    // --- WEIGHT PATH STREAMS ---
    hls::stream<weight_mat_t> ram_to_router_w("ram_to_router_w");
    hls::stream<weight_mat_t> router_to_sys_w("router_to_sys_w");
    hls::stream<WeightBundle> router_to_wino_w("router_to_wino_w");

    #pragma HLS STREAM variable=ram_to_router_w  depth=16
    #pragma HLS STREAM variable=router_to_sys_w  depth=16
    #pragma HLS STREAM variable=router_to_wino_w depth=16

    // Module Executions
    read_axilite_bias(bias_array, internal_bias, descriptor.Cout);

    controller_top(
        descriptor, start_accel,
        mode_streams[0], mode_streams[1], mode_streams[2], 
        mode_streams[3], mode_streams[4], mode_streams[5],
        lb_cfg, wr_cfg, ddemux_cfg, wdemux_cfg,
        sys_cfg, wino_cfg, ser_cfg, fuse_cfg, slb_cfg
    );

    line_buffer(pixels_in_stream, lb_to_router, mode_streams[2], lb_cfg);
    weight_controller_top(weights_in_stream, ram_to_router_w, wr_cfg);
    
    data_demux(lb_to_router, mode_streams[3], router_to_sys_data, router_to_wino_data, ddemux_cfg);
    weight_demux(ram_to_router_w, mode_streams[4], router_to_sys_w, router_to_wino_w, wdemux_cfg);

    winograd_engine_top(router_to_wino_data, router_to_wino_w, wino_to_mux, mode_streams[1], wino_cfg);
    systolic_engine(router_to_sys_data, router_to_sys_w, sys_to_mux, mode_streams[0], sys_cfg);
    
    sub_line_buffer_top(residual_in_stream, res_to_fuse, slb_cfg);
    compute_to_fuse_serializer(sys_to_mux, wino_to_mux, mux_to_fuse, mode_streams[5], ser_cfg);
    fuse_post_conv(mux_to_fuse, res_to_fuse, fuse_out_stream, internal_bias, fuse_cfg);
}