#include "tb_weightram.h"

void tb_weight_ram() {
    hls::stream<weight_t> in_weights("in_weights_stream");
    hls::stream<weight_mat_t> out_weight_stream("out_weights_stream");
    
    // MỚI: Thay 3 stream int bằng 1 struct
    hls::stream<WeightRamConfig> config_stream("config_stream");

    int test_num_tiles = 3;    
    int test_weight_size = 2;  

    cout << ">> [TESTBENCH] Bat dau kiem tra Weight Controller (Stream-based)" << endl;
    
    // Ghi Struct
    config_stream.write({test_num_tiles, test_weight_size});

    cout << "\n>> [TESTBENCH] Bom du lieu vao in_weights stream" << endl;
    for (int w = 0; w < test_weight_size; w++) {
        for (int bank = 0; bank < 16; bank++) {
            weight_t val = (weight_t)((1 * 100) + (w * 16) + bank);
            in_weights.write(val);
        }
    }


    cout << "\n>> [TESTBENCH] Dang chay weight_controller_top" << endl;

    // Cập nhật hàm gọi
    weight_controller_top(
        in_weights, 
        out_weight_stream, 
        config_stream
    );

    cout << "\n>> [TESTBENCH] Kiem tra luong du lieu dau ra (Co che Reuse)" << endl;

    int pass_count = 0;
    // Tổng số lần đọc mong muốn (mỗi lần đọc ra 1 weight_mat_t chứa 16 phần tử)
    int expected_total_reads = 1 * test_num_tiles * test_weight_size;
    int actual_reads = 0;

        
    // Kiểm tra vòng lặp Reuse
    for (int t = 0; t < test_num_tiles; t++) {
        cout << "  Tile (Reuse) thu " << t << ":" << endl;
        
        // Kiểm tra từng vector trong bộ weight của Tile đó
        for (int w = 0; w < test_weight_size; w++) {
            
            if (out_weight_stream.empty()) {
                cout << "  [LOI FATAL] Stream rong dot ngot tai Tile " << t << ", Vector " << w << "!" << endl;
                break;
            }

            weight_mat_t out_mat = out_weight_stream.read();
            actual_reads++;
            bool vector_pass = true;

            // Kiểm tra chéo 16 phần tử trong 1 vector so với giá trị đã bơm vào
            for (int bank = 0; bank < 16; bank++) {
                weight_t expected_val = (weight_t)((1 * 100) + (w * 16) + bank);
                // Lấy ra 8-bit tương ứng với bank đó
                weight_t actual_val = out_mat.range(bank * 8 + 7, bank * 8); 

                if (actual_val != expected_val) {
                    vector_pass = false;
                    cout << "    [LOI DATA] Bank " << bank << " sai. Mong doi: " 
                            << (int)expected_val << ", Thuc te: " << (int)actual_val << endl;
                }
            }

            if (vector_pass) {
                cout << "    Vector " << w << ": PASS" << endl;
                pass_count++;
            }
        }
    }


    cout << ">> TONG KET TESTBENCH:" << endl;
    if (actual_reads == expected_total_reads && pass_count == expected_total_reads) {
        cout << ">> [Thanh cong] TAT CA TEST DEU PASS (" << pass_count << "/" << expected_total_reads << ")" << endl;
        cout << "==============================================" << endl;
    } else {
        cout << ">> [That bai] CO LOI XAY RA! Pass: " << pass_count << ", Expected: " << expected_total_reads << endl;
        cout << "==============================================" << endl;
    }
}