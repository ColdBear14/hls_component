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
    // 1. CẬP NHẬT KIỂU DỮ LIỆU: pixels_in đổi thành weight_mat_t
    hls::stream<weight_mat_t> pixels_in("pixels_in");
    hls::stream<weight_mat_t> weights_in("weights_in");
    hls::stream<psum_block_t> psums_out("psums_out");
    hls::stream<ap_uint<2>> mode_stream("mode_stream");
    hls::stream<SystolicConfig> config_stream("config_stream");

    std::cout << ">>> TEST CASE: " << test_name << " (Mode=" << mode_sel << ", Cin=" << Cin << ", Tiles=" << tiles << ")" << std::endl;

    // Ghi cấu hình
    config_stream.write({Cin, Cout, tiles});
    mode_stream.write(mode_sel);

    int k_max = (mode_sel == 1) ? 9 : 1; 
    int cout_blocks = (Cout + 15) / 16;
    if (cout_blocks == 0) cout_blocks = 1;
    
    // Giá trị kỳ vọng tính toán (Ép kiểu long long để không bị tràn số học C++)
    long long expected_val = (long long)Cin * k_max * p_val * w_val;

    // 2. NẠP DỮ LIỆU ĐÚNG THEO CHU KỲ CỦA ENGINE 2D OS
    for (int t = 0; t < tiles; t++) {
        for (int cb = 0; cb < cout_blocks; cb++) {
            
            // Engine 2D OS đọc đúng (Cin * k_max) vector cho mỗi Block
            for (int iter = 0; iter < Cin * k_max; iter++) {
                weight_mat_t p_vec = 0;
                weight_mat_t w_vec = 0;
                
                // Đóng gói 16 pixel và 16 weight vào vector 128-bit
                for (int i = 0; i < 16; i++) {
                    p_vec.range(i * 8 + 7, i * 8) = (ap_uint<8>)p_val;
                    w_vec.range(i * 8 + 7, i * 8) = (ap_uint<8>)w_val;
                }
                
                pixels_in.write(p_vec);
                weights_in.write(w_vec);
            }
        }
    }

    // 3. Gọi Engine
    systolic_engine(pixels_in, weights_in, psums_out, mode_stream, config_stream);

    // 4. KIỂM TRA KẾT QUẢ VỚI KHUNG XUẤT 16x16
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

    // 5. Báo cáo (Output của OS Engine = tiles * cout_blocks * 16 hàng)
    int expected_total_blocks = tiles * cout_blocks * ARRAY_SIZE;

    if (match && total_blocks == expected_total_blocks) {
        std::cout << "[PASS] Ket qua khop: " << expected_val 
                  << " | Tong so hang Psum: " << total_blocks << std::endl;
    } else {
        std::cout << "[FAIL] Sai lech!" << std::endl;
        std::cout << "       -> Expected Val: " << expected_val << std::endl;
        std::cout << "       -> Expected Blocks: " << expected_total_blocks 
                  << " | Got: " << total_blocks << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;
}

void SystolicEngine_TB() {
    // LƯU Ý BẢO MẬT ÉP KIỂU: 
    // Kiểu pixel_t và weight_t của bạn là ap_int<8> (có dấu, range: -128 đến 127).
    // Test case cũ truyền vào 255 và 200 sẽ bị ép kiểu âm (âm nhân âm / âm nhân dương)
    // Tôi đã chỉnh lại các số liệu dưới 127 để việc Debug và tính tay dễ dàng hơn.

    run_systolic_test_generic("Input/Weight Zeros", 2, 16, 1, 0, 0, 1);
    run_systolic_test_generic("Max Signed Values (127x127)", 1, 16, 1, 127, 127, 1);
    run_systolic_test_generic("Continuous Stream (50 Tiles)", 4, 32, 50, 2, 3, 1);
    run_systolic_test_generic("Large Accumulation (Cin=256)", 256, 16, 1, 5, 5, 1);
    run_systolic_test_generic("1x1 Convolution Test", 4, 16, 10, 5, 10, 2);
}