#include "tb_top_system.h"
#include <algorithm> // Cho std::min

#include "tb_top_system.h"
#include <algorithm> // Cho std::min

void run_top_system_test(int mode, int IMG_W, int IMG_H, int pad, int Cin, int Cout, bool has_residual) {

    int K;
    if (mode == 2) K = 1;
    else K = 3;

    int stride = (mode == 1) ? 2 : 1;
    
    // 1. TÍNH TOÁN KÍCH THƯỚC PAD VÀ OUTPUT
    int PADDED_W = IMG_W + 2 * pad;
    int PADDED_H = IMG_H + 2 * pad;
    int OUT_W = (PADDED_W - K) / stride + 1;
    int OUT_H = (PADDED_H - K) / stride + 1;
    int TOTAL_OUT_PIXELS = OUT_W * OUT_H * Cout;

    // 2. CẤP PHÁT BỘ NHỚ
    pixel_t* img_mem = new pixel_t[IMG_W * IMG_H * Cin]();
    weight_t* weight_mem = new weight_t[Cout * Cin * K * K]();
    ap_int<8> bias_mem[MAX_CHANNELS];
    pixel_t* residual_mem = new pixel_t[TOTAL_OUT_PIXELS]();
    pixel_t* hw_output = new pixel_t[TOTAL_OUT_PIXELS]();
    pixel_t* gold_output = new pixel_t[TOTAL_OUT_PIXELS]();

    for (int i = 0; i < IMG_W * IMG_H * Cin; i++) img_mem[i] = rand() % 10 - 5;
    for (int i = 0; i < Cout * Cin * K * K; i++) weight_mem[i] = rand() % 6 - 3;
    for (int i = 0; i < Cout; i++) bias_mem[i] = rand() % 4;
    for (int i = 0; i < TOTAL_OUT_PIXELS; i++) residual_mem[i] = rand() % 10 - 5;

    // 3. TÍNH TOÁN KÍCH THƯỚC TILE (TILING THEO CHANNEL DO GIỚI HẠN BRAM)
    // Hardware BRAM có 8096 địa chỉ
    int max_bram_addr = 8096;
    int max_cout_step = 16;
    
    if (mode == 0) {
        max_cout_step = max_bram_addr / Cin; // Winograd: 1 địa chỉ lưu 1 kernel đã transform của Cin
    } else {
        int k_max = (mode == 1) ? 9 : 1;
        int max_cout_blocks = max_bram_addr / (k_max * Cin);
        max_cout_step = max_cout_blocks * 16;
    }
    
    max_cout_step = (max_cout_step / 16) * 16; // Đảm bảo luôn là bội số của 16
    if (max_cout_step == 0) max_cout_step = 16; 
    
    int cout_step = std::min(Cout, max_cout_step);
    
    std::cout << "  -> Tiên đoán BRAM: Chia tính toán làm các chunk có Cout = " 
              << cout_step << " / " << Cout << std::endl;

    // Các streams dùng cho phần cứng
    hls::stream<pixel_t> in_pixels("in_pixels");
    hls::stream<pixel_t> residual_in("residual_in");
    hls::stream<weight_t> weight_in("weight_in");
    hls::stream<axi_dma_out_t> out_data("out_data");

    // VÒNG LẶP CHANNEL TILING: Chạy nhiều lượt để tránh tràn RAM
    for (int cout_start = 0; cout_start < Cout; cout_start += cout_step) {
        int current_cout = std::min(cout_step, Cout - cout_start);
        
        // --- 4. BƠM DỮ LIỆU ĐẦU VÀO CHO TỪNG LƯỢT CHẠY ---
        // A. Bơm lại toàn bộ ảnh (vì Feature map input là không đổi)
        for (int i = 0; i < IMG_W * IMG_H * Cin; i++) in_pixels.write(img_mem[i]);

        // B. Bơm Residual (Chỉ lấy lát cắt cho current_cout)
        if(has_residual) {
            if (mode == 0) {
                int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1;
                int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
                
                for (int ty = 0; ty < tiles_y; ty++) {
                    for (int tx = 0; tx < tiles_x; tx++) {
                        for (int c = 0; c < current_cout; c++) {
                            int global_c = cout_start + c;
                            int global_y = ty * 2;
                            int global_x = tx * 2;
                            
                            if (global_y < OUT_H && global_x < OUT_W) 
                                residual_in.write(residual_mem[(global_y * OUT_W + global_x) * Cout + global_c]);
                            else residual_in.write(0);
                            
                            if (global_y < OUT_H && global_x + 1 < OUT_W) 
                                residual_in.write(residual_mem[(global_y * OUT_W + global_x + 1) * Cout + global_c]);
                            else residual_in.write(0);
                            
                            if (global_y + 1 < OUT_H && global_x < OUT_W) 
                                residual_in.write(residual_mem[((global_y + 1) * OUT_W + global_x) * Cout + global_c]);
                            else residual_in.write(0);
                            
                            if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                                residual_in.write(residual_mem[((global_y + 1) * OUT_W + global_x + 1) * Cout + global_c]);
                            else residual_in.write(0);
                        }
                    }
                }
            } else {
                int current_cout_blocks = (current_cout + 15) / 16;
                for (int t = 0; t < OUT_W * OUT_H; t++) {
                    for (int cb = 0; cb < current_cout_blocks; cb++) {
                        for (int c = 0; c < 16; c++) {
                            int co = cb * 16 + c;
                            int global_c = cout_start + co;
                            if (co < current_cout) {
                                residual_in.write(residual_mem[t * Cout + global_c]);
                            } else {
                                residual_in.write(0);
                            }
                        }
                    }
                }
            }
        } 

        // C. Bơm Weight (Chỉ lấy weight cho các kênh Output hiện tại)
        if (mode == 0) {
            // Lấy con trỏ dịch đến đúng offset của cout_start
            fill_Wino_weight_stream(weight_mem + (cout_start * Cin * K * K), K, Cin, current_cout, weight_in);
        } else {
            int current_cout_blocks = (current_cout + 15) / 16;
            for (int ci = 0; ci < Cin; ci++) {
                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
                        for (int cb = 0; cb < current_cout_blocks; cb++) {
                            for (int c = 0; c < 16; c++) {
                                int co = cb * 16 + c;
                                int global_c = cout_start + co;
                                if (co < current_cout) {
                                    int w_idx = (((global_c * Cin) + ci) * K + ky) * K + kx;
                                    weight_in.write(weight_mem[w_idx]);
                                } else {
                                    weight_in.write(0); 
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- 5. GỌI HARDWARE ACCELERATOR ---
        LayerDescriptor desc;
        desc.W = IMG_W;
        desc.H = IMG_H;
        desc.Cin = Cin;
        desc.Cout = current_cout; 
        desc.kernel_size = K;
        desc.stride = stride;
        desc.pad = pad;
        desc.has_residual = has_residual;
        desc.requant_shift_val = 2;

        cnn_accelerator_top(
            in_pixels, residual_in, 
            weight_in,
            out_data, 
            bias_mem + cout_start, // Truyền lát cắt Bias
            desc, true
        );

        // --- 6. THU THẬP VÀ GÁN DỮ LIỆU ĐÚNG OFFSET ---
        int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1; 
        int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
        
        int expected_packets = 0;
        int current_cout_blocks = (current_cout + 15) / 16;
        
        if (mode == 0) expected_packets = tiles_x * tiles_y * ((current_cout + 3) / 4);
        else           expected_packets = OUT_W * OUT_H * current_cout_blocks;

        int c = 0, t = 0;
        for (int p = 0; p < expected_packets; ++p) {
            if(out_data.empty()) {
                std::cerr << "Warning: stream is empty at packet " << p << std::endl;
                break; 
            }
            axi_dma_out_t hw_packet = out_data.read();

            if (p == expected_packets - 1) {
                if (hw_packet.last != 1) {
                    std::cerr << "LỖI AXI DMA: TLAST KHÔNG ĐƯỢC BẬT ở packet cuối cùng (" << p << ")!" << std::endl;
                }
            } else {
                if (hw_packet.last != 0) {
                    std::cerr << "LỖI AXI DMA: TLAST bị bật sai vị trí ở packet (" << p << ")!" << std::endl;
                }
            }
            
            if (mode == 0) {
                for (int w = 0; w < 4; w++) {
                    if (t >= (tiles_x * tiles_y)) break;
                    int tx = t % tiles_x;
                    int ty = t / tiles_x;
                    int global_y = ty * 2;
                    int global_x = tx * 2;
                    int global_c = cout_start + c;
                    
                    if (global_y < OUT_H && global_x < OUT_W) 
                        hw_output[(global_y * OUT_W + global_x) * Cout + global_c] = hw_packet.data[w*4 + 0];
                    if (global_y < OUT_H && global_x + 1 < OUT_W) 
                        hw_output[(global_y * OUT_W + global_x + 1) * Cout + global_c] = hw_packet.data[w*4 + 1];
                    if (global_y + 1 < OUT_H && global_x < OUT_W) 
                        hw_output[((global_y + 1) * OUT_W + global_x) * Cout + global_c] = hw_packet.data[w*4 + 2];
                    if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                        hw_output[((global_y + 1) * OUT_W + global_x + 1) * Cout + global_c] = hw_packet.data[w*4 + 3];
                    
                    c++;
                    if (c >= current_cout) { c = 0; t++; }
                }
            } else {
                int global_y = t / OUT_W;
                int global_x = t % OUT_W;
                int cb = p % current_cout_blocks;
                
                for (int ch = 0; ch < 16; ch++) {
                    int co = cb * 16 + ch;
                    int global_co = cout_start + co;
                    if (co < current_cout && global_y < OUT_H && global_x < OUT_W) {
                        hw_output[(global_y * OUT_W + global_x) * Cout + global_co] = hw_packet.data[ch];
                    }
                }
                if (cb == current_cout_blocks - 1) t++;
            }
        }
    }

    // 7. TÍNH TOÁN GOLDEN VÀ SO SÁNH
    golden_conv_full(IMG_W, IMG_H, Cin, Cout, K, stride, pad,
                     img_mem, weight_mem, bias_mem, residual_mem, 
                     has_residual, 2, gold_output); // 2 là shift_val cố định
    
    int errors = 0;
    for (int i = 0; i < TOTAL_OUT_PIXELS; i++) {
        if (hw_output[i] != gold_output[i]) {
            if (errors < 15) {
                std::cerr << "Mismatch tại [Pixel " << i << "]: HW = " << (int)hw_output[i] 
                          << " | Golden = " << (int)gold_output[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) std::cout << ">>> KẾT QUẢ: PASS 100% (Khớp toàn bộ " << TOTAL_OUT_PIXELS << " điểm ảnh) <<<\n" << std::endl;
    else             std::cout << ">>> KẾT QUẢ: FAILED (" << errors << " / " << TOTAL_OUT_PIXELS << " lỗi) <<<\n" << std::endl;

    delete[] img_mem;
    delete[] weight_mem;
    delete[] residual_mem;
    delete[] hw_output;
    delete[] gold_output;
}

void TopSystem_TB() {
    std::vector<TestScenario> scenarios = {
        // --- 1. CÁC TEST CƠ BẢN ---
        {0, 16, 16, 0, 3,  8,  true,  "Baseline Winograd (No Pad)"},
        {1, 16, 16, 0, 3,  8,  true,  "Baseline Systolic 3x3 S2 (No Pad)"},
        {2, 16, 16, 0, 3,  8,  true,  "Baseline Systolic 1x1 S1 (No Pad)"},
        {0, 16, 16, 1, 3,  8,  true,  "Winograd 3x3 (Pad=1)"},
        {1, 16, 16, 1, 3,  8,  true,  "Systolic 3x3 S2 (Pad=1)"},

        // // --- 2. CÁC SCENARIO THỰC TẾ TỪ YOLOv8n (INPUT 640x640) ---
        // // [Stage 0] Stem Layer: Lớp đầu tiên của mạng, chuyển từ ảnh RGB (3 channel) sang 16 channel
        // {1, 640, 640, 1, 3,  16, false, "YOLOv8n Stem: 640x640, 3x3 S2, C:3->16"},

        // // [Stage 1] Downsample 1: Giảm kích thước ảnh xuống 1/4 (160x160)
        // {1, 320, 320, 1, 16, 32, false, "YOLOv8n Down1: 320x320, 3x3 S2, C:16->32"},

        // // [Stage 2] C2f Bottleneck 1: Dùng Conv 3x3, Stride 1, có cộng Residual
        // {0, 160, 160, 1, 32, 32, true,  "YOLOv8n C2f-B1: 160x160, 3x3 S1, C:32->32 (Wino+Res)"},

        // // [Stage 3] Downsample 2: Giảm xuống 80x80
        // {1, 160, 160, 1, 32, 64, false, "YOLOv8n Down2: 160x160, 3x3 S2, C:32->64"},

        // // [Stage 4] C2f Bottleneck 2 (1x1 Conv): Lớp gộp kênh đầu vào trong khối C2f
        // {2, 80,  80,  0, 64, 64, false, "YOLOv8n C2f-1x1: 80x80, 1x1 S1, C:64->64 (Sys)"},

        // // [Stage 5] Downsample 3: Giảm xuống 40x40
        // {1, 80,  80,  1, 64, 128, false, "YOLOv8n Down3: 80x80, 3x3 S2, C:64->128"},

        // // [Stage 6] C2f Bottleneck 3 (Deeper): 40x40 xử lý Winograd với số kênh lớn hơn
        // {0, 40,  40,  1, 128, 128, true, "YOLOv8n C2f-B3: 40x40, 3x3 S1, C:128->128 (Wino+Res)"},

        // // [Stage 7] Downsample 4: Giảm xuống độ phân giải nhỏ nhất 20x20
        // {1, 40,  40,  1, 128, 256, false, "YOLOv8n Down4: 40x40, 3x3 S2, C:128->256"},

        // // [Stage 8] Khối SPPF / Head: Dùng nhiều Conv 1x1 ở độ phân giải 20x20
        // {2, 20,  20,  0, 256, 256, false, "YOLOv8n SPPF/Head: 20x20, 1x1 S1, C:256->256 (Sys)"}
    };

    int pass_count = 0;

    for (size_t i = 0; i < scenarios.size(); ++i) {
        std::cout << "\n>>> TEST CASE " << i + 1 << "/" << scenarios.size() 
                  << ": " << scenarios[i].desc << " <<<" << std::endl;
        
        try {
            run_top_system_test(
                scenarios[i].mode, 
                scenarios[i].img_w, 
                scenarios[i].img_h, 
                scenarios[i].pad,
                scenarios[i].cin, 
                scenarios[i].cout, 
                scenarios[i].has_residual
            );
            pass_count++;
        } catch (...) {
            std::cerr << "!!! EXCEPTION CAUGHT IN TEST " << i + 1 << " !!!" << std::endl;
        }
    }

    std::cout << "TEST SUMMARY: " << pass_count << "/" << scenarios.size() << " PASSED." << std::endl;

    std::cout << "\n>>> ALL TESTS COMPLETED: " << pass_count << "/" << scenarios.size() << " PASSED. <<<" << std::endl;
}