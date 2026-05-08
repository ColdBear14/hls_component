#include "tb_linebuffer.h"

void tb_linebuffer() {
    std::cout << "--- BẮT ĐẦU TESTBENCH LINEBUFFER ---" << std::endl;
    hls::stream<pixel_t> in_stream;
    hls::stream<Tile4x4> out_tile_stream;
    hls::stream<ap_uint<2>> mode; 
    
    // MỚI: Khai báo 1 luồng Struct duy nhất thay cho 4 luồng int
    hls::stream<LineBufferConfig> config_stream; 

    Tile4x4 tile;

    std::cout<<"---Wino(3x3,s1)---"<<std::endl;
    mode.write(0);
    config_stream.write({8, 8, 1, 0}); // W, H, Cin, pad
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, config_stream);
    
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout<<string(50,'-')<<std::endl;

    std::cout<<"---Sys(3x3,s2)---"<<std::endl;
    mode.write(1);
    config_stream.write({8, 8, 1, 0});
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, config_stream);
    
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout<<string(50,'-')<<std::endl;


    std::cout<<"---Sys(1x1,s2)---"<<std::endl;
    mode.write(2);
    config_stream.write({8, 8, 1, 0});
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, config_stream);
    
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout<<string(50,'-')<<std::endl;
}