#include "tb_weightram.h"

void tb_weight_ram() {
    // ĐÃ CẬP NHẬT: Đổi từ weight_t sang axi_word_t
    hls::stream<axi_word_t> in_weights("in_weights_stream");
    hls::stream<weight_mat_t> out_weight_stream("out_weights_stream");
    
    hls::stream<WeightRamConfig> config_stream("config_stream");

    int test_num_tiles = 3;    
    int test_weight_size = 2;  

    cout << ">> [TESTBENCH] Bat dau kiem tra Weight Controller (128-bit Stream)" << endl;
    
    // Ghi Struct
    config_stream.write({test_num_tiles, test_weight_size});

    cout << "\n>> [TESTBENCH] Bom du lieu vao in_weights stream (Pack thanh 128-bit)" << endl;
    for (int w = 0; w < test_weight_size; w++) {
        axi_word_t pack_word = 0;
        
        for (int bank = 0; bank < 16; bank++) {
            weight_t val = (weight_t)((1 * 100) + (w * 16) + bank);
            // Ghép 16 giá trị 8-bit vào 1 block 128-bit
            pack_word.range(bank * 8 + 7, bank * 8) = val;
        }
        
        // Ghi trọn 128-bit vào phần cứng (tương đương 1 chu kỳ AXI)
        in_weights.write(pack_word);
    }

    cout << "\n>> [TESTBENCH] Dang chay weight_controller_top" << endl;

    weight_controller_top(
        in_weights, 
        out_weight_stream, 
        config_stream
    );

    cout << "\n>> [TESTBENCH] Kiem tra luong du lieu dau ra (Co che Reuse)" << endl;

    int pass_count = 0;
    // Tổng số lần đọc mong muốn
    int expected_total_reads = 1 * test_num_tiles * test_weight_size;
    int actual_reads = 0;
        
    // Kiểm tra vòng lặp Reuse
    for (int t = 0; t < test_num_tiles; t++) {
        cout << "  Tile (Reuse) thu " << t << ":" << endl;
        
        for (int w = 0; w < test_weight_size; w++) {
            
            if (out_weight_stream.empty()) {
                cout << "  [LOI FATAL] Stream rong dot ngot tai Tile " << t << ", Vector " << w << "!" << endl;
                break;
            }

            weight_mat_t out_mat = out_weight_stream.read();
            actual_reads++;
            bool vector_pass = true;

            for (int bank = 0; bank < 16; bank++) {
                weight_t expected_val = (weight_t)((1 * 100) + (w * 16) + bank);
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