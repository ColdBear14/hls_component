#include "helper.h"

void golden_winograd(Tile4x4 d, Tile3x3 g, Tile2x2 &y_ref) {
    // Compute standard convolution: input d (4x4) with kernel v (3x3) -> output y (2x2)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            data_t sum = 0;
            for (int ki = 0; ki < 3; ++ki) {
                for (int kj = 0; kj < 3; ++kj) {
                    sum += (data_t)(d.data[i+ki][j+kj] * g.data[ki][kj]);
                }
            }
            y_ref.data[i][j] = sum;
        }
    }
}

void weight_transform(Tile3x3 g, Tile4x4 &V) {
    #pragma HLS INLINE
    int32_t temp[4][3];

    // Bước 1: temp = G_scaled * g
    for (int j = 0; j < 3; j++) {
        #pragma HLS UNROLL
        int32_t g0 = g.data[0][j];
        int32_t g1 = g.data[1][j];
        int32_t g2 = g.data[2][j];

        temp[0][j] = g0 << 1;           // g0 * 2
        temp[1][j] = (g0 + g1 + g2);    // (g0 + g1 + g2) - Không chia 2
        temp[2][j] = (g0 - g1 + g2);    // (g0 - g1 + g2) - Không chia 2
        temp[3][j] = g2 << 1;           // g2 * 2
    }

    // Bước 2: V = temp * G_scaled^T
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        int32_t t0 = temp[i][0];
        int32_t t1 = temp[i][1];
        int32_t t2 = temp[i][2];

        V.data[i][0] = t0 << 1;         // t0 * 2
        V.data[i][1] = (t0 + t1 + t2);
        V.data[i][2] = (t0 - t1 + t2);
        V.data[i][3] = t2 << 1;         // t2 * 2
    }
}

void fill_axi_pixel_stream(hls::stream<axi_word_t>& in_stream, int W, int H, int Cin) {
    int total_pixels = W * H * Cin;
    axi_word_t pack_word = 0;
    pixel_t dummy_val = 1; // Biến tăng dần để dễ kiểm tra

    for (int i = 0; i < total_pixels; i++) {
        int idx = i % 16;
        pack_word.range(idx * 8 + 7, idx * 8) = dummy_val++;
        
        // Đủ 16 pixels hoặc pixel cuối cùng thì ghi vào stream
        if (idx == 15 || i == total_pixels - 1) {
            in_stream.write(pack_word);
            pack_word = 0;
        }
    }
}

void fill_weight_stream(hls::stream<weight_t>& stream, int kernel_size, int Cin, int Cout) {
    std::cout << "\n=== WEIGHT MATRIX ===" << std::endl;
    
    // Tạo buffer lưu trữ tạm thời để in các channel ngang hàng nhau (side-by-side)
    std::vector<std::vector<std::vector<weight_t>>> buffer(Cin, 
        std::vector<std::vector<weight_t>>(kernel_size, std::vector<weight_t>(kernel_size)));

    for (int cout = 0; cout < Cout; ++cout) {
        std::cout << "Output Channel " << cout << ":" << std::endl;
        
        // 1. Ghi dữ liệu vào stream (Giữ nguyên logic chuẩn để phần cứng đọc đúng thứ tự)
        for (int cin = 0; cin < Cin; ++cin) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    weight_t w = (i == j) ? 1 : 0;
                    stream.write(w);
                    buffer[cin][i][j] = w; // Lưu tạm vào buffer để in
                }
            }
        }

        // 2. In nhãn Input Channel (Cin 0, Cin 1,...) trên cùng 1 dòng
        std::cout << "    ";
        for (int cin = 0; cin < Cin; ++cin) {
            std::cout << "Cin " << cin;
            // Tính toán khoảng trắng để căn lề tương đối theo kernel_size
            for(int space = 0; space < (kernel_size * 2) - 1; ++space) std::cout << " ";
        }
        std::cout << std::endl;

        // 3. In các ma trận kernel cạnh nhau
        for (int i = 0; i < kernel_size; ++i) {
            std::cout << "    ";
            for (int cin = 0; cin < Cin; ++cin) {
                std::cout << "[";
                for (int j = 0; j < kernel_size; ++j) {
                    std::cout << buffer[cin][i][j];
                    if (j < kernel_size - 1) std::cout << " ";
                }
                std::cout << "]   "; // Khoảng cách giữa các ma trận
            }
            std::cout << std::endl; // Xuống dòng khi in xong 1 hàng của tất cả các Cin
        }
        std::cout << std::endl;
    }
    std::cout << "===================\n" << std::endl;
}

// MỚI: Chuyển đổi trọng số Winograd (Khớp thứ tự Cin -> Cout)
void fill_Wino_weight_stream(weight_t* weight_mem, int kernel_size, int Cin, int Cout, hls::stream<weight_t>& output) {
    for (int cin = 0; cin < Cin; ++cin) {
        for (int cout = 0; cout < Cout; ++cout) {
            Tile3x3 g;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    // Trích xuất phần tử dựa trên layout ban đầu [Cout][Cin][K][K]
                    int w_idx = (((cout * Cin) + cin) * kernel_size + i) * kernel_size + j;
                    g.data[i][j] = weight_mem[w_idx];
                }
            }
            
            Tile4x4 V;
            weight_transform(g, V);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    output.write(V.data[i][j]);
                }
            }
        }
    }
}

void standard_convolution_golden(
    int W, int H, int Cin, int Cout, int Tiles_X, int Tiles_Y,
    pixel_t* img,  
    weight_t* weight, // Changed to flat pointer
    data_t* output_gold // Changed to flat pointer
) {
    int out_h = Tiles_Y * 2;
    int out_w = Tiles_X * 2;
    
    for(int i = 0; i < out_h * out_w * Cout; i++) {
        output_gold[i] = 0;
    }

    // Standard convolution
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            for (int r = 0; r < out_h; r++) {
                for (int c = 0; c < out_w; c++) {
                    data_t sum = 0;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            int img_idx = ((r + kr) * W + (c + kc)) * Cin + ci;
                            // weight[co][ci][kr][kc] calculation:
                            int weight_idx = (((co * Cin) + ci) * 3 + kr) * 3 + kc;
                            
                            sum += (data_t)img[img_idx] * (data_t)weight[weight_idx];
                        }
                    }
                    // output_gold[r][c][co] calculation:
                    int out_idx = (r * out_w + c) * Cout + co;
                    output_gold[out_idx] += sum;
                }
            }
        }
    }
}

// Thêm 'int pad' vào danh sách tham số
void golden_conv_full(
    int W, int H, int Cin, int Cout, int K, int stride, int pad,
    pixel_t* img, weight_t* weight, ap_int<8>* bias, pixel_t* residual,
    bool has_residual, int requant_shift, pixel_t* final_out
) {
    // Tính kích thước đầu ra DỰA TRÊN KÍCH THƯỚC ĐÃ PAD
    int out_w = (W + 2 * pad - K) / stride + 1;
    int out_h = (H + 2 * pad - K) / stride + 1;

    for (int out_y = 0; out_y < out_h; out_y++) {
        for (int out_x = 0; out_x < out_w; out_x++) {
            for (int co = 0; co < Cout; co++) {
                int32_t sum = 0;
                
                // Tính MAC tích chập
                for (int ci = 0; ci < Cin; ci++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            // Tính toán tọa độ ánh xạ ngược về ảnh gốc (có trừ đi pad)
                            int in_y = out_y * stride + ky - pad;
                            int in_x = out_x * stride + kx - pad;
                            
                            pixel_t px_val = 0; // Mặc định là 0 (Padding)
                            
                            // Chỉ lấy dữ liệu thật nếu tọa độ nằm trong biên ảnh gốc
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                int img_idx = (in_y * W + in_x) * Cin + ci;
                                px_val = img[img_idx];
                            }
                            
                            int w_idx = (((co * Cin) + ci) * K + ky) * K + kx;
                            sum += px_val * weight[w_idx];
                        }
                    }
                }
                
                // --- Post-Processing ---
                sum += bias[co];                

                if (sum < 0) {
                    int32_t relu_mult = sum * 13;
                    sum = relu_mult >> 7;
                }

                sum = sum >> requant_shift;     

                if (sum > 127) sum = 127;
                else if (sum < -128) sum = -128;

                if (has_residual) {
                    int res_idx = (out_y * out_w + out_x) * Cout + co;
                    sum += residual[res_idx];
                    if (sum > 127) sum = 127;
                    else if (sum < -128) sum = -128;
                }

                // Lưu kết quả Golden
                int out_idx = (out_y * out_w + out_x) * Cout + co;
                final_out[out_idx] = (pixel_t)sum;
            }
        }
    }
}