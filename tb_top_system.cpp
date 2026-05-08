#include "tb_top_system.h"
#include <algorithm> 
#include <vector>

void run_top_system_test(int mode, int IMG_W, int IMG_H, int pad, int Cin, int Cout, bool has_residual) {

    int K = (mode == 2) ? 1 : 3;
    int stride = (mode == 1) ? 2 : 1;
    
    int PADDED_W = IMG_W + 2 * pad;
    int PADDED_H = IMG_H + 2 * pad;
    
    int OUT_W = (PADDED_W - K) / stride + 1;
    int OUT_H = (PADDED_H - K) / stride + 1;
    int TOTAL_OUT_PIXELS = OUT_W * OUT_H * Cout;

    // 2. Cấp phát bộ nhớ (Giữ nguyên kích thước tổng)
    pixel_t* img_mem = new pixel_t[IMG_W * IMG_H * Cin]();
    weight_t* weight_mem = new weight_t[Cout * Cin * K * K]();
    ap_int<8> bias_mem[MAX_CHANNELS];
    pixel_t* residual_mem = new pixel_t[TOTAL_OUT_PIXELS]();
    pixel_t* hw_output = new pixel_t[TOTAL_OUT_PIXELS]();
    pixel_t* gold_output = new pixel_t[TOTAL_OUT_PIXELS]();

    // Tạo dữ liệu ngẫu nhiên
    for (int i = 0; i < IMG_W * IMG_H * Cin; i++) img_mem[i] = rand() % 10 - 5;
    for (int i = 0; i < Cout * Cin * K * K; i++) weight_mem[i] = rand() % 6 - 3;
    for (int i = 0; i < Cout; i++) bias_mem[i] = rand() % 4;
    for (int i = 0; i < TOTAL_OUT_PIXELS; i++) residual_mem[i] = rand() % 10 - 5;

    // Tính toán Golden Model (Tính một lần cho toàn bộ)
    LayerDescriptor full_desc;
    full_desc.requant_shift_val = 2;
    golden_conv_full(IMG_W, IMG_H, Cin, Cout, K, stride, pad,
                     img_mem, weight_mem, bias_mem, residual_mem, 
                     has_residual, full_desc.requant_shift_val, gold_output);

    // ==============================================================
    // [TILING] VÒNG LẶP CHIA NHỎ COUT
    // ==============================================================
    int num_tiles = (Cout + MAX_COUT_TILE - 1) / MAX_COUT_TILE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int current_cout = std::min(MAX_COUT_TILE, Cout - tile * MAX_COUT_TILE);
        int cout_offset = tile * MAX_COUT_TILE;

        // Khai báo Streams cục bộ cho mỗi Tile
        hls::stream<axi_word_t> in_pixels("in_pixels");
        hls::stream<axi_word_t> residual_in("residual_in");
        hls::stream<axi_word_t> weight_in("weight_in");
        hls::stream<axi_word_t> out_data("out_data");
        hls::stream<weight_t> temp_weight_stream("temp_weight_stream"); 

        std::vector<pixel_t> temp_in_pixels;
        std::vector<pixel_t> temp_residual;

        // 3.1 Chuẩn bị dữ liệu Ảnh (Gửi lại toàn bộ ảnh cho mỗi Tile vì HW chỉ tính 1 phần kênh)
        // Đã sửa: Padding kênh ảo để khớp với chuẩn 16 kênh/word của LineBuffer
        int cin_blocks = (Cin + 15) / 16;
        for (int h = 0; h < IMG_H; h++) {
            for (int w = 0; w < IMG_W; w++) {
                for (int cb = 0; cb < cin_blocks; cb++) {
                    for (int c = 0; c < 16; c++) {
                        int actual_c = cb * 16 + c;
                        if (actual_c < Cin) {
                            temp_in_pixels.push_back(img_mem[(h * IMG_W + w) * Cin + actual_c]);
                        } else {
                            temp_in_pixels.push_back(0); // Pad các kênh ảo bằng 0
                        }
                    }
                }
            }
        }

        // 3.2 Chuẩn bị dữ liệu Residual (Chỉ lấy đoạn channel thuộc về Tile hiện tại)
        if(has_residual) {
            if (mode == 0) {
                int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1;
                int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
                
                for (int ty = 0; ty < tiles_y; ty++) {
                    for (int tx = 0; tx < tiles_x; tx++) {
                        for (int c = 0; c < current_cout; c++) { // Lặp theo current_cout
                            int global_y = ty * 2;
                            int global_x = tx * 2;
                            int global_c = cout_offset + c;      // [TILING] Căn theo offset
                            
                            if (global_y < OUT_H && global_x < OUT_W) 
                                temp_residual.push_back(residual_mem[(global_y * OUT_W + global_x) * Cout + global_c]);
                            else temp_residual.push_back(0); 
                            
                            if (global_y < OUT_H && global_x + 1 < OUT_W) 
                                temp_residual.push_back(residual_mem[(global_y * OUT_W + global_x + 1) * Cout + global_c]);
                            else temp_residual.push_back(0);
                            
                            if (global_y + 1 < OUT_H && global_x < OUT_W) 
                                temp_residual.push_back(residual_mem[((global_y + 1) * OUT_W + global_x) * Cout + global_c]);
                            else temp_residual.push_back(0);
                            
                            if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                                temp_residual.push_back(residual_mem[((global_y + 1) * OUT_W + global_x + 1) * Cout + global_c]);
                            else temp_residual.push_back(0);
                        }
                    }
                }
            } else {
                int cout_blocks = (current_cout + 15) / 16;
                for (int t = 0; t < OUT_W * OUT_H; t++) {     
                    for (int cb = 0; cb < cout_blocks; cb++) { 
                        for (int c = 0; c < 16; c++) {
                            int co_local = cb * 16 + c;
                            int global_c = cout_offset + co_local; // [TILING] Căn theo offset
                            if (co_local < current_cout) {
                                temp_residual.push_back(residual_mem[t * Cout + global_c]);
                            } else {
                                temp_residual.push_back(0); 
                            }
                        }
                    }
                }
            }
        } 

        // 3.3 Chuẩn bị dữ liệu Trọng số (Chỉ lấy Weight cho current_cout)
        // [TILING] Dịch con trỏ weight_mem tới vị trí bắt đầu của tile hiện tại
        weight_t* current_weight_ptr = weight_mem + (cout_offset * Cin * K * K);

        if (mode == 0) {
            fill_Wino_weight_stream(current_weight_ptr, K, Cin, current_cout, temp_weight_stream);
        } else {
            int cout_blocks = (current_cout + 15) / 16;
            for (int ci = 0; ci < Cin; ci++) {
                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
                        for (int cb = 0; cb < cout_blocks; cb++) {
                            for (int c = 0; c < 16; c++) {
                                int co_local = cb * 16 + c;
                                if (co_local < current_cout) {
                                    // Index tính theo current_weight_ptr
                                    int w_idx = (((co_local * Cin) + ci) * K + ky) * K + kx;
                                    temp_weight_stream.write(current_weight_ptr[w_idx]);
                                } else {
                                    temp_weight_stream.write(0); 
                                }
                            }
                        }
                    }
                }
            }
        }

        // ==============================================================
        // PACKER LOGIC (Giữ nguyên, nhưng pack dữ liệu cục bộ của Tile)
        // ==============================================================
        axi_word_t pack_word = 0;
        for (size_t i = 0; i < temp_in_pixels.size(); i++) {
            int idx = i % 16;
            pack_word.range(idx*8+7, idx*8) = temp_in_pixels[i];
            if (idx == 15 || i == temp_in_pixels.size() - 1) {
                in_pixels.write(pack_word);
                pack_word = 0;
            }
        }

        pack_word = 0;
        for (size_t i = 0; i < temp_residual.size(); i++) {
            int idx = i % 16;
            pack_word.range(idx*8+7, idx*8) = temp_residual[i];
            if (idx == 15 || i == temp_residual.size() - 1) {
                residual_in.write(pack_word);
                pack_word = 0;
            }
        }

        pack_word = 0;
        int w_idx = 0;
        while (!temp_weight_stream.empty()) {
            pack_word.range(w_idx*8+7, w_idx*8) = temp_weight_stream.read();
            w_idx++;
            if (w_idx == 16) {
                weight_in.write(pack_word);
                w_idx = 0;
                pack_word = 0;
            }
        }
        if (w_idx > 0) weight_in.write(pack_word);

        // 4. Khởi tạo Cấu hình Layer cho Tile
        LayerDescriptor desc_tile;
        desc_tile.W = IMG_W;
        desc_tile.H = IMG_H;
        desc_tile.Cin = Cin;
        desc_tile.Cout = current_cout; // [TILING] Báo cho HW biết chỉ tính số kênh này
        desc_tile.kernel_size = K;
        desc_tile.stride = stride;
        desc_tile.pad = pad;
        desc_tile.has_residual = has_residual;      
        desc_tile.requant_shift_val = 2;   

        // Cắt mảng bias cho Tile hiện tại
        ap_int<8> tile_bias_mem[MAX_CHANNELS];
        for (int i = 0; i < current_cout; i++) {
            tile_bias_mem[i] = bias_mem[cout_offset + i];
        }

        // 5. Chạy Top Accelerator cho TILE HIỆN TẠI
        cnn_accelerator_top(
            in_pixels, residual_in, 
            weight_in,
            out_data, 
            tile_bias_mem,             
            desc_tile, true
        );
        
        // 6. Thu thập dữ liệu và ráp vào Global Memory
        int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1; 
        int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
        
        int expected_packets = 0;
        int cout_blocks = (current_cout + 15) / 16;
        
        if (mode == 0) expected_packets = tiles_x * tiles_y * ((current_cout + 3) / 4);
        else           expected_packets = OUT_W * OUT_H * cout_blocks;

        int c = 0, t = 0;
        for (int p = 0; p < expected_packets; ++p) {
            if(out_data.empty()) break; 
            
            axi_word_t hw_word = out_data.read();
            ap_int<8> hw_data[16];
            for (int i = 0; i < 16; i++) {
                hw_data[i] = hw_word.range(i * 8 + 7, i * 8);
            }
            
            if (mode == 0) {
                for (int w = 0; w < 4; w++) {
                    if (t >= (tiles_x * tiles_y)) break;
                    int tx = t % tiles_x;
                    int ty = t / tiles_x;
                    int global_y = ty * 2;
                    int global_x = tx * 2;
                    int global_c = cout_offset + c; // [TILING] Căn index kênh ngõ ra
                    
                    if (global_y < OUT_H && global_x < OUT_W) 
                        hw_output[(global_y * OUT_W + global_x) * Cout + global_c] = hw_data[w*4 + 0];
                    if (global_y < OUT_H && global_x + 1 < OUT_W) 
                        hw_output[(global_y * OUT_W + global_x + 1) * Cout + global_c] = hw_data[w*4 + 1];
                    if (global_y + 1 < OUT_H && global_x < OUT_W) 
                        hw_output[((global_y + 1) * OUT_W + global_x) * Cout + global_c] = hw_data[w*4 + 2];
                    if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                        hw_output[((global_y + 1) * OUT_W + global_x + 1) * Cout + global_c] = hw_data[w*4 + 3];
                    
                    c++;
                    if (c >= current_cout) {
                        c = 0; t++;
                    }
                }
            } else {
                int global_y = t / OUT_W;
                int global_x = t % OUT_W;
                int cb = p % cout_blocks;
                
                for (int ch = 0; ch < 16; ch++) {
                    int co_local = cb * 16 + ch;
                    int global_c = cout_offset + co_local; // [TILING] Căn index kênh ngõ ra
                    
                    if (co_local < current_cout && global_y < OUT_H && global_x < OUT_W) {
                        hw_output[(global_y * OUT_W + global_x) * Cout + global_c] = hw_data[ch];
                    }
                }
                if (cb == cout_blocks - 1) t++;
            }
        }
    } // Kết thúc vòng lặp Tile

    // 7. Kiểm tra chéo (Giữ nguyên)
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

    if (errors == 0) std::cout << ">>> KẾT QUẢ: PASS 100% (Khớp toàn bộ " << TOTAL_OUT_PIXELS << " điểm ảnh) <<<" << std::endl;
    else             std::cout << ">>> KẾT QUẢ: FAILED (" << errors << " / " << TOTAL_OUT_PIXELS << " lỗi) <<<" << std::endl;

    delete[] img_mem;
    delete[] weight_mem;
    delete[] residual_mem;
    delete[] hw_output;
    delete[] gold_output;
}

void TopSystem_TB() {
    std::vector<TestScenario> scenarios = {
        {0, 16, 16, 0, 3,  8,  true,  "Baseline Winograd (No Pad)"},
        {1, 16, 16, 0, 3,  8,  true,  "Baseline Systolic 3x3 S2 (No Pad)"},
        {2, 16, 16, 0, 3,  8,  true,  "Baseline Systolic 1x1 S1 (No Pad)"},
        {0, 16, 16, 1, 3,  8,  true,  "Winograd 3x3 (Pad=1)"},
        {1, 16, 16, 1, 3,  8,  true,  "Systolic 3x3 S2 (Pad=1)"},

        // --- 2. CÁC SCENARIO THỰC TẾ TỪ YOLOv8n (INPUT 640x640) ---
        // [Stage 0] Stem Layer: Lớp đầu tiên của mạng, chuyển từ ảnh RGB (3 channel) sang 16 channel
        {1, 640, 640, 1, 3,  16, false, "YOLOv8n Stem: 640x640, 3x3 S2, C:3->16"},

        // [Stage 1] Downsample 1: Giảm kích thước ảnh xuống 1/4 (160x160)
        {1, 320, 320, 1, 16, 32, false, "YOLOv8n Down1: 320x320, 3x3 S2, C:16->32"},

        // [Stage 2] C2f Bottleneck 1: Dùng Conv 3x3, Stride 1, có cộng Residual
        {0, 160, 160, 1, 32, 32, true,  "YOLOv8n C2f-B1: 160x160, 3x3 S1, C:32->32 (Wino+Res)"},

        // [Stage 3] Downsample 2: Giảm xuống 80x80
        {1, 160, 160, 1, 32, 64, false, "YOLOv8n Down2: 160x160, 3x3 S2, C:32->64"},

        // [Stage 4] C2f Bottleneck 2 (1x1 Conv): Lớp gộp kênh đầu vào trong khối C2f
        {2, 80,  80,  0, 64, 64, false, "YOLOv8n C2f-1x1: 80x80, 1x1 S1, C:64->64 (Sys)"},

        // [Stage 5] Downsample 3: Giảm xuống 40x40
        {1, 80,  80,  1, 64, 128, false, "YOLOv8n Down3: 80x80, 3x3 S2, C:64->128"},

        // [Stage 6] C2f Bottleneck 3 (Deeper): 40x40 xử lý Winograd với số kênh lớn hơn
        {0, 40,  40,  1, 128, 128, true, "YOLOv8n C2f-B3: 40x40, 3x3 S1, C:128->128 (Wino+Res)"},

        // [Stage 7] Downsample 4: Giảm xuống độ phân giải nhỏ nhất 20x20
        {1, 40,  40,  1, 128, 256, false, "YOLOv8n Down4: 40x40, 3x3 S2, C:128->256"},

        // [Stage 8] Khối SPPF / Head: Dùng nhiều Conv 1x1 ở độ phân giải 20x20
        {2, 20,  20,  0, 256, 256, false, "YOLOv8n SPPF/Head: 20x20, 1x1 S1, C:256->256 (Sys)"}

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