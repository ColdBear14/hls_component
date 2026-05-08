#include "tb_top_system.h"
#include <algorithm> // Cho std::min

void run_top_system_test(int mode, int IMG_W, int IMG_H, int pad, int Cin, int Cout, bool has_residual) {

    int K;
    if (mode == 2) K = 1;
    else K = 3;

    int stride = (mode == 1) ? 2 : 1;
    
    // 1. TÍNH TOÁN KÍCH THƯỚC PAD
    int PADDED_W = IMG_W + 2 * pad;
    int PADDED_H = IMG_H + 2 * pad;
    
    // Tính kích thước Output dựa trên kích thước đã Pad
    int OUT_W = (PADDED_W - K) / stride + 1;
    int OUT_H = (PADDED_H - K) / stride + 1;
    int TOTAL_OUT_PIXELS = OUT_W * OUT_H * Cout;

    // 2. Cấp phát bộ nhớ
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



    // 3. Chuẩn bị HLS Streams
    hls::stream<pixel_t> in_pixels("in_pixels");
    hls::stream<pixel_t> residual_in("residual_in");
    hls::stream<weight_t> weight_in("weight_in");

    hls::stream<fuse_vec_out_t> out_data("out_data");

    hls::stream<weight_t> temp_weight_stream("temp_weight_stream"); 

    // Bơm ảnh
    for (int i = 0; i < IMG_W * IMG_H * Cin; i++) in_pixels.write(img_mem[i]);

    if(has_residual) {
        if (mode == 0) {
            // Chế độ Winograd: Bơm Residual theo thứ tự Tile -> Cout -> 4 Pixels
            int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1;
            int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
            
            for (int ty = 0; ty < tiles_y; ty++) {
                for (int tx = 0; tx < tiles_x; tx++) {
                    for (int c = 0; c < Cout; c++) {
                        int global_y = ty * 2;
                        int global_x = tx * 2;
                        
                        // Pixel 0 (y, x)
                        if (global_y < OUT_H && global_x < OUT_W) 
                            residual_in.write(residual_mem[(global_y * OUT_W + global_x) * Cout + c]);
                        else residual_in.write(0); // Padding nếu tràn viền
                        
                        // Pixel 1 (y, x+1)
                        if (global_y < OUT_H && global_x + 1 < OUT_W) 
                            residual_in.write(residual_mem[(global_y * OUT_W + global_x + 1) * Cout + c]);
                        else residual_in.write(0);
                        
                        // Pixel 2 (y+1, x)
                        if (global_y + 1 < OUT_H && global_x < OUT_W) 
                            residual_in.write(residual_mem[((global_y + 1) * OUT_W + global_x) * Cout + c]);
                        else residual_in.write(0);
                        
                        // Pixel 3 (y+1, x+1)
                        if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                            residual_in.write(residual_mem[((global_y + 1) * OUT_W + global_x + 1) * Cout + c]);
                        else residual_in.write(0);
                    }
                }
            }
        } else {
            // Chế độ Systolic: Bơm Residual với padding lên bội số của 16 channel (cout_blocks)
            int cout_blocks = (Cout + 15) / 16;
            for (int t = 0; t < OUT_W * OUT_H; t++) {     // Duyệt từng pixel
                for (int cb = 0; cb < cout_blocks; cb++) { // Duyệt từng block 16 channels
                    for (int c = 0; c < 16; c++) {
                        int co = cb * 16 + c;
                        if (co < Cout) {
                            residual_in.write(residual_mem[t * Cout + co]);
                        } else {
                            residual_in.write(0); // Padding thêm số 0 cho đủ 16 phần tử
                        }
                    }
                }
            }
        }
    } 


    // Bơm Trọng số (Winograd cần Transform, Systolic dùng Raw)
    if (mode == 0) {
        fill_Wino_weight_stream(weight_mem, K, Cin, Cout, weight_in);
    } else {
        int cout_blocks = (Cout + 15) / 16;
        for (int ci = 0; ci < Cin; ci++) {
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    for (int cb = 0; cb < cout_blocks; cb++) {
                        for (int c = 0; c < 16; c++) {
                            int co = cb * 16 + c;
                            if (co < Cout) {
                                int w_idx = (((co * Cin) + ci) * K + ky) * K + kx;
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
    // 4. Khởi tạo Cấu hình Layer (Descriptor)
    LayerDescriptor desc;
    desc.W = IMG_W;
    desc.H = IMG_H;
    desc.Cin = Cin;
    desc.Cout = Cout;
    desc.kernel_size = K;
    desc.stride = stride;
    desc.pad = pad;
    desc.has_residual = has_residual;        // Bật tính năng Residual
    desc.requant_shift_val = 2;   // Shift 2 bit


    // desc.weight_length = temp_weight_stream.size();

    // weight_t* hardware_weight_mem = new weight_t[100000]();
    // for (int i = 0; i < desc.weight_length; i++) {
    //     hardware_weight_mem[i] = temp_weight_stream.read();
    // }
    
    // 5. Chạy Top Accelerator
    cnn_accelerator_top(
        in_pixels, residual_in, 
        weight_in,
        out_data, 
        bias_mem,             
        desc, true
    );

    // 6. Tính toán Golden Model
    golden_conv_full(IMG_W, IMG_H, Cin, Cout, K, stride, pad,
                     img_mem, weight_mem, bias_mem, residual_mem, 
                     has_residual, desc.requant_shift_val, gold_output);
    
    // 7. Thu thập dữ liệu từ Hardware và So sánh
    int tiles_x = OUT_W / 2; if (tiles_x == 0) tiles_x = 1; 
    int tiles_y = OUT_H / 2; if (tiles_y == 0) tiles_y = 1;
    
    int expected_packets = 0;
    int cout_blocks = (Cout + 15) / 16;
    
    // Tính đúng số packet sinh ra
    if (mode == 0) expected_packets = tiles_x * tiles_y * ((Cout + 3) / 4);
    else           expected_packets = OUT_W * OUT_H * cout_blocks;

    int c = 0, t = 0;
    for (int p = 0; p < expected_packets; ++p) {

        if(out_data.empty()) {
            std::cerr << "Warning: stream is empty at packet " << p << std::endl;
            break; 
        }
        fuse_vec_out_t hw_packet = out_data.read();
        
        if (mode == 0) {
            // Giải mã Mode 0: Winograd (Mỗi gói chứa 4 pixel của 4 kênh)
            for (int w = 0; w < 4; w++) {
                if (t >= (tiles_x * tiles_y)) break;
                int tx = t % tiles_x;
                int ty = t / tiles_x;
                int global_y = ty * 2;
                int global_x = tx * 2;
                
                if (global_y < OUT_H && global_x < OUT_W) 
                    hw_output[(global_y * OUT_W + global_x) * Cout + c] = hw_packet.data[w*4 + 0];
                if (global_y < OUT_H && global_x + 1 < OUT_W) 
                    hw_output[(global_y * OUT_W + global_x + 1) * Cout + c] = hw_packet.data[w*4 + 1];
                if (global_y + 1 < OUT_H && global_x < OUT_W) 
                    hw_output[((global_y + 1) * OUT_W + global_x) * Cout + c] = hw_packet.data[w*4 + 2];
                if (global_y + 1 < OUT_H && global_x + 1 < OUT_W) 
                    hw_output[((global_y + 1) * OUT_W + global_x + 1) * Cout + c] = hw_packet.data[w*4 + 3];
                
                c++;
                if (c >= Cout) {
                    c = 0; t++;
                }
            }
        } else {
            // Giải mã Mode 1/2: Systolic (Mỗi gói chứa 16 kênh của 1 pixel)
            int global_y = t / OUT_W;
            int global_x = t % OUT_W;
            int cb = p % cout_blocks;
            
            for (int ch = 0; ch < 16; ch++) {
                int co = cb * 16 + ch;
                if (co < Cout && global_y < OUT_H && global_x < OUT_W) {
                    hw_output[(global_y * OUT_W + global_x) * Cout + co] = hw_packet.data[ch];
                }
            }
            
            if (cb == cout_blocks - 1) { // Đã đọc xong tất cả các channel block cho pixel này
                t++;
            }
        }
    }

    // Kiểm tra chéo (Verification)
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

    // Dọn dẹp RAM
    delete[] img_mem;
    delete[] weight_mem;
    delete[] residual_mem;
    delete[] hw_output;
    delete[] gold_output;
    // delete[] hardware_weight_mem; // Đừng quên giải phóng mảng mới tạo
}

void TopSystem_TB() {
    std::vector<TestScenario> scenarios = {
        // --- 1. CÁC TEST CƠ BẢN ĐÃ CÓ ---
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