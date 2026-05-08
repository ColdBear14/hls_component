#include "Scheduler.h"

void controller_top(
    const LayerDescriptor& descriptor,
    bool start,
    
    hls::stream<ap_uint<2>>& mode_sys_stream,
    hls::stream<ap_uint<2>>& mode_wino_stream,
    hls::stream<ap_uint<2>>& mode_lb_stream,
    hls::stream<ap_uint<2>>& mode_ddemux_stream,
    hls::stream<ap_uint<2>>& mode_wdemux_stream,
    hls::stream<ap_uint<2>>& mode_serializer_stream,

    hls::stream<LineBufferConfig>& lb_cfg_stream,
    hls::stream<WeightRamConfig>& wr_cfg_stream,
    hls::stream<DemuxDataConfig>& ddemux_cfg_stream,
    hls::stream<DemuxWeightConfig>& wdemux_cfg_stream,
    hls::stream<SystolicConfig>& sys_cfg_stream,
    hls::stream<WinogradConfig>& wino_cfg_stream,
    hls::stream<SerializerConfig>& serializer_cfg_stream,
    hls::stream<FuseConfig>& fuse_cfg_stream,
    hls::stream<FuseConfig>& slb_cfg_stream
) {
    
    #pragma HLS INTERFACE s_axilite port=descriptor bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=start bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    if (start) {
        int W      = descriptor.W;
        int H      = descriptor.H;
        int pad    = descriptor.pad;
        int K      = descriptor.kernel_size;
        int stride = descriptor.stride;
        int Cin    = descriptor.Cin;
        int Cout   = descriptor.Cout;

        ap_uint<2> mode;

        if (K == 3 && stride == 1) {
            mode = 0; // Winograd 3x3 s1
        } else if (K == 3 && stride == 2) {
            mode = 1; // Systolic 3x3 s2
        } else if (K == 1 && stride == 1) {
            mode = 2; // Systolic 1x1 s1
        }

        mode_sys_stream.write(mode);
        mode_wino_stream.write(mode);
        mode_lb_stream.write(mode);
        mode_ddemux_stream.write(mode);
        mode_wdemux_stream.write(mode);
        mode_serializer_stream.write(mode);

        int padded_W = W + 2 * pad;
        int padded_H = H + 2 * pad;
        int tiles_X, tiles_Y;
    
        if (mode == 2) {
            tiles_X = padded_W;
            tiles_Y = padded_H;
        } else if (mode == 1) { 
            tiles_X = (padded_W - 3) / 2 + 1;
            tiles_Y = (padded_H - 3) / 2 + 1;
        } else { 
            tiles_X = (padded_W - 2) / 2; 
            tiles_Y = (padded_H - 2) / 2;
        }

        int total_tiles_per_channel = tiles_X * tiles_Y;
        int cout_blocks = (Cout + 15) / 16;
        
        int wdemux_weights_val, wr_weights_val, packets_val;
        int serializer_reads_val;

        // --- BƯỚC 1: TÍNH TOÁN TỔNG WEIGHTS YÊU CẦU CHO TỪNG MODE ---
        if (mode == 0) {
            wr_weights_val       = Cin * Cout;                              
            packets_val          = (total_tiles_per_channel * Cout + 3) / 4; 
            serializer_reads_val = total_tiles_per_channel * Cout;
        } else {
            int k_max            = (mode == 1) ? 9 : 1;
            wr_weights_val       = k_max * Cin * cout_blocks;
            packets_val          = total_tiles_per_channel * cout_blocks; 
            serializer_reads_val = packets_val;
        }
        
        // --- BƯỚC 2: LOGIC PHÂN PHA (PHASE-TILING) CHO BRAM ---
        wdemux_weights_val = total_tiles_per_channel * wr_weights_val;

        int out_w = (mode == 0) ? (tiles_X * 2) : tiles_X;
        int out_h = (mode == 0) ? (tiles_Y * 2) : tiles_Y;

        // --- BƯỚC 3: ĐÓNG GÓI VÀ GHI VÀO LUỒNG CẤU HÌNH ---
        lb_cfg_stream.write({W, H, Cin, pad});
        
        wr_cfg_stream.write({total_tiles_per_channel, wr_weights_val});
        
        ddemux_cfg_stream.write({total_tiles_per_channel, Cin});
        wdemux_cfg_stream.write({wdemux_weights_val});
        
        sys_cfg_stream.write({Cin, Cout, total_tiles_per_channel});
        wino_cfg_stream.write({total_tiles_per_channel, Cin, Cout});
        serializer_cfg_stream.write({serializer_reads_val});   
             
        FuseConfig f_cfg;
        f_cfg.has_residual       = descriptor.has_residual;
        f_cfg.channel_limit      = Cout;
        f_cfg.feature_map_width  = out_w;
        f_cfg.feature_map_height = out_h;
        f_cfg.requant_shift_val  = descriptor.requant_shift_val;
        f_cfg.mode               = mode;
        f_cfg.total_packets      = packets_val;

        fuse_cfg_stream.write(f_cfg);
        slb_cfg_stream.write(f_cfg);
    }
}