#include "tb_winograd.h"

void WinogradEngine_TB(){
    Tile4x4 d_zero, d_max;
    Tile3x3 g_zero, g_max;
    
    for(int i=0; i<16; i++) {
    d_zero.data[i/4][i%4] = 0;
    d_max.data[i/4][i%4] = 255;
    }
    for(int i=0; i<9; i++) {
        g_zero.data[i/3][i%3] = 0;
        g_max.data[i/3][i%3] = 255;
    }

    // 1. Input toàn 0
    run_single_test("Input All Zeros", d_zero, g_max);

    // 2. Weight toàn 0
    run_single_test("Weight All Zeros", d_max, g_zero);

    // 3. Giá trị lớn nhất (255) - Kiểm tra saturation hoặc overflow đơn giản
    run_single_test("Max Values (255)", d_max, g_max);

    Tile4x4 d_rand;
    Tile3x3 g_rand;
    for(int i=0; i<16; i++) d_rand.data[i/4][i%4] = rand() % 256;
    for(int i=0; i<9; i++) g_rand.data[i/3][i%3] = rand() % 256;

    run_single_test("Random Values", d_rand, g_rand);

    Tile4x4 d_overflow;
    Tile3x3 g_overflow;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) d_overflow.data[i][j] = ((i+j)%2 == 0) ? 255 : 0;
    }
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) g_overflow.data[i][j] = 127; // Trọng số dương lớn
    }

    run_single_test("Potential Overflow Case", d_overflow, g_overflow);

    // Test với nhiều tile liên tiếp
    test_multiple_tiles(10);
    test_multiple_tiles(100);
    test_multiple_tiles(1000);

}

void run_single_test(std::string test_name, Tile4x4 d, Tile3x3 g) {
    hls::stream<Tile4x4> in_tile_stream;
    hls::stream<Tile4x4> weight_v_stream;
    hls::stream<Tile2x2> out_tile_stream;
    hls::stream<ap_uint<2>> mode_stream;
    
    // MỚI: Luồng cấu hình
    hls::stream<WinogradConfig> config_stream;

    std::cout << "\n--- TEST CASE: " << test_name << " ---" << std::endl;

    Tile4x4 v;
    weight_transform(g, v);

    in_tile_stream.write(d);
    weight_v_stream.write(v);
    mode_stream.write(0); 
    
    // Ghi num_tiles, Cin, Cout
    config_stream.write({1, 1, 1}); 

    // Chạy module
    winograd_engine_top(in_tile_stream, weight_v_stream, out_tile_stream, 
                        mode_stream, config_stream);

    // Kiểm tra kết quả
    if (!out_tile_stream.empty()) {
        Tile2x2 result = out_tile_stream.read();
        Tile2x2 golden;
        golden_winograd(d, g, golden);

        int errors = 0;
        for(int i=0; i<2; i++) {
            for(int j=0; j<2; j++) {
                if(result.data[i][j] != golden.data[i][j]) {
                    errors++;
                    std::cout << "Mismatch at [" << i << "][" << j << "]: HW=" 
                              << result.data[i][j] << ", Golden=" << golden.data[i][j] << std::endl;
                }
            }
        }
        if(errors == 0) std::cout << ">> KẾT QUẢ: PASS" << std::endl;
        else std::cout << ">> KẾT QUẢ: FAIL (" << errors << " lỗi)" << std::endl;
    }
}

void test_multiple_tiles(int num_tiles) {
    hls::stream<Tile4x4> in_tile_stream;
    hls::stream<Tile4x4> weight_v_stream;
    hls::stream<Tile2x2> out_tile_stream;
    hls::stream<ap_uint<2>> mode_stream;
    hls::stream<WinogradConfig> config_stream;

    std::cout << "\n--- TEST CASE: Multiple Tiles (" << num_tiles << ") ---" << std::endl;

    Tile3x3 g;
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) g.data[i][j] = (i==j) ? 1 : 0;
    Tile4x4 v; weight_transform(g, v);
    
    config_stream.write({num_tiles, 1, 1});
    mode_stream.write(0);

    for(int n=0; n < num_tiles; n++) {
        Tile4x4 d;
        for(int i=0; i<16; i++) d.data[i/4][i%4] = n + i; 
        in_tile_stream.write(d);
        weight_v_stream.write(v);
    }

    winograd_engine_top(in_tile_stream, weight_v_stream, out_tile_stream, 
                        mode_stream, config_stream);

    int count = 0;
    while(!out_tile_stream.empty()) {
        out_tile_stream.read();
        count++;
    }
    std::cout << "Tiles received: " << count << (count == num_tiles ? " (PASS)" : " (FAIL)") << std::endl;
}