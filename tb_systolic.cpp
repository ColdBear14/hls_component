#include "tb_systolic.h"

void run_systolic_test_generic(
    std::string test_name, 
    int Cin, 
    int Cout, 
    int tiles, 
    pixel_t p_val, 
    weight_t w_val, 
    int mode_sel
) {
    hls::stream<SysWindow> pixels_in("pixels_in");
    hls::stream<weight_mat_t> weights_in("weights_in");
    hls::stream<psum_block_t> psums_out("psums_out");
    hls::stream<ap_uint<2>> mode_stream("mode_stream");
    
    // MỚI: Sử dụng stream cấu trúc
    hls::stream<SystolicConfig> config_stream("config_stream");

    std::cout << ">>> TEST CASE: " << test_name << " (Mode=" << mode_sel << ", Cin=" << Cin << ", Tiles=" << tiles << ")" << std::endl;

    // 1. Ghi cấu hình
    config_stream.write({Cin, Cout, tiles});
    mode_stream.write(mode_sel);

    int k_max = (mode_sel == 1) ? 9 : 1; 
    int cout_blocks = (Cout + 15) / 16;
    
    long long expected_val = (long long)Cin * k_max * p_val * w_val;

    // 2. Nạp dữ liệu vào Stream
    for (int t = 0; t < tiles; t++) {
        for (int ci = 0; ci < Cin; ci++) {
            SysWindow win;
            for (int k = 0; k < 9; k++) win.data[k] = p_val;
            pixels_in.write(win);

            for (int k = 0; k < k_max; k++) {
                for (int cb = 0; cb < cout_blocks; cb++) {
                    weight_mat_t w_vec = 0;
                    for (int c = 0; c < 16; c++) {
                        w_vec.range(c * 8 + 7, c * 8) = (ap_uint<8>)w_val;
                    }
                    weights_in.write(w_vec);
                }
            }
        }
    }

    // 3. Gọi Engine
    systolic_engine(pixels_in, weights_in, psums_out, mode_stream, config_stream);

    // 4. Kiểm tra kết quả
    int total_blocks = 0;
    bool match = true;
    while (!psums_out.empty()) {
        psum_block_t res = psums_out.read();
        for (int i = 0; i < 16; i++) {
            if (res.data[i] != expected_val) {
                match = false;
            }
        }
        total_blocks++;
    }

    // 5. Báo cáo
    if (match && total_blocks == (tiles * cout_blocks)) {
        std::cout << "[PASS] Ket qua khop: " << expected_val << " | Blocks: " << total_blocks << std::endl;
    } else {
        std::cout << "[FAIL] Sai lech ket qua hoac thieu tile! (Expected: " << expected_val << ")" << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;
}

void SystolicEngine_TB(){
    run_systolic_test_generic("Input/Weight Zeros", 2, 16, 1, 0, 0, 1);
    run_systolic_test_generic("Max Values (255x127)", 1, 16, 1, 255, 127, 1);
    run_systolic_test_generic("Continuous Stream (50 Tiles)", 4, 32, 50, 2, 3, 1);
    run_systolic_test_generic("Large Accumulation (Cin=256)", 256, 16, 1, 200, 100, 1);
    run_systolic_test_generic("1x1 Convolution Test", 4, 16, 10, 5, 10, 2);
}