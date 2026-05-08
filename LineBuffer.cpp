#include "LineBuffer.h"


// ==========================================================
// STAGE 1: Đọc AXI 128-bit và trượt cửa sổ 16 kênh/chu kỳ
// ==========================================================
void read_and_shift(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<WindowVec16> &win_stream,
    int padded_W, int padded_H, int Cin, int pad, ap_uint<2> mode
) {
    // Vector hóa theo số kênh (HWC layout)
    int Cin_vec = (Cin + 15) / 16; 
    int total_iters = padded_W * padded_H * Cin_vec;

    // Buffer 128-bit, lưu 16 kênh song song
    static ap_uint<128> line_buf[3][MAX_LINE_BUFFER_SIZE / 16];
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram

    // Cửa sổ lưu trữ 16 kênh song song
    static ap_uint<128> window[4][4][MAX_CIN / 16];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window complete dim=2

    int col = 0, row = 0, cin_v = 0;

    process_loop_vec: for (int i = 0; i < total_iters; i++) {
        #pragma HLS PIPELINE II=1
        
        bool is_pad_top = (row < pad);
        bool is_pad_bottom = (row >= padded_H - pad);
        bool is_pad_left = (col < pad);
        bool is_pad_right = (col >= padded_W - pad);

        ap_uint<128> in_word = 0;
        
        // Đọc 128-bit (16 pixels) trực tiếp từ AXI nếu không thuộc vùng Padding
        if (!(is_pad_top || is_pad_bottom || is_pad_left || is_pad_right)) {
            in_word = in_stream.read();
        }

        int idx = col * Cin_vec + cin_v;

        // 1. Đọc giá trị cũ từ Line Buffer
        ap_uint<128> lb0 = line_buf[0][idx];
        ap_uint<128> lb1 = line_buf[1][idx];
        ap_uint<128> lb2 = line_buf[2][idx];

        // 2. Cập nhật Line Buffer (Dịch dòng)
        line_buf[0][idx] = lb1;
        line_buf[1][idx] = lb2;
        line_buf[2][idx] = in_word;

        // 3. Trượt toàn bộ Window 4x4 (128-bit/vị trí)
        for (int r = 0; r < 4; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 3; c++) {
                #pragma HLS UNROLL
                window[r][c][cin_v] = window[r][c+1][cin_v];
            }
        }
        window[0][3][cin_v] = lb0;
        window[1][3][cin_v] = lb1;
        window[2][3][cin_v] = lb2;
        window[3][3][cin_v] = in_word;

        // 4. Logic xác định thời điểm xuất Tile (Giữ nguyên logic gốc)
        bool valid_output = false;
        if (mode == 2) valid_output = true;
        else if (row >= 3 && col >= 3) valid_output = ((col & 1) == 1) && ((row & 1) == 1);

        // 5. Nếu hợp lệ, đóng gói 16 kênh vào WindowVec16 và đẩy vào FIFO
        if (valid_output) {
            WindowVec16 out_vec;
            #pragma HLS ARRAY_PARTITION variable=out_vec.p complete dim=1
            #pragma HLS ARRAY_PARTITION variable=out_vec.p complete dim=2
            
            for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 4; c++) {
                    #pragma HLS UNROLL
                    ap_uint<128> w_val = window[r][c][cin_v];
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS UNROLL
                        out_vec.p[r][c][k] = w_val.range(k*8+7, k*8);
                    }
                }
            }
            win_stream.write(out_vec);
        }

        // 6. Tăng bộ đếm
        cin_v++;
        if (cin_v == Cin_vec) {
            cin_v = 0;
            if (col == padded_W - 1) {
                col = 0;
                row++;
            } else {
                col++;
            }
        }
    }
}

// ==========================================================
// STAGE 2: Bóc tách 16 Tile và Serialize ra luồng đầu ra (ĐÃ TỐI ƯU)
// ==========================================================
void serialize_output(
    hls::stream<WindowVec16> &win_stream,
    hls::stream<Tile4x4> &out_tile_stream,
    int padded_W, int padded_H, int Cin, ap_uint<2> mode
) {
    // Tính toán tổng số Tile không gian dựa theo Scheduler
    int tiles_X, tiles_Y;
    if (mode == 2) {
        tiles_X = padded_W; 
        tiles_Y = padded_H;
    } else if (mode == 1) { 
        tiles_X = (padded_W - 3) / 2 + 1; 
        tiles_Y = (padded_H - 3) / 2 + 1;
    } else { 
        tiles_X = (padded_W - 2) / 2; 
        tiles_Y = (padded_H - 2) / 2;
    }
    
    int total_tiles = tiles_X * tiles_Y;
    int Cin_vec = (Cin + 15) / 16;
    
    // Gom tất cả thành 1 vòng lặp phẳng (Flat Loop)
    int total_iters = total_tiles * Cin_vec * 16;
    
    WindowVec16 in_vec;
    int cv = 0;
    int k = 0;

    serialize_flat_loop: for (int i = 0; i < total_iters; i++) {
        #pragma HLS PIPELINE II=1
        // Thêm TRIPCOUNT để Vitis HLS không báo cáo ảo 3.7 tỷ cycles nữa
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=65536 
        
        // Chỉ đọc stream mới khi bắt đầu một block 16 kênh
        if (k == 0) {
            in_vec = win_stream.read();
        }
        
        // CHỈ GHI VÀO LUỒNG NẾU LÀ KÊNH THẬT
        if (cv * 16 + k < Cin) {
            Tile4x4 out_tile;
            
            for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 4; c++) {
                    #pragma HLS UNROLL
                    out_tile.data[r][c] = (mode == 2) ? (data_t)in_vec.p[3][3][k] : (data_t)in_vec.p[r][c][k];
                }
            }
            out_tile_stream.write(out_tile);
        }
        
        k++;
        if (k == 16) {
            k = 0;
            cv++;
            if (cv == Cin_vec) {
                cv = 0;
            }
        }
    }
}

// ==========================================================
// TOP MODULE LINE BUFFER
// ==========================================================
void line_buffer(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<Tile4x4> &out_tile_stream, 
    hls::stream<ap_uint<2>>& mode_stream,             
    hls::stream<LineBufferConfig>& config_stream
) {
    #pragma HLS INLINE off
    
    LineBufferConfig cfg = config_stream.read();
    int img_width  = cfg.W;
    int img_height = cfg.H;
    int Cin        = cfg.Cin;
    int pad        = cfg.pad;
    ap_uint<2> mode = mode_stream.read(); 

    int padded_W = img_width + 2 * pad;
    int padded_H = img_height + 2 * pad;

    // Sử dụng luồng dữ liệu trung gian với độ sâu vừa đủ
    hls::stream<WindowVec16> win_stream("window_vec_stream");
    #pragma HLS STREAM variable=win_stream depth=8

    // Kích hoạt mô hình Dataflow để hai stage chạy song song
    #pragma HLS DATAFLOW

    read_and_shift(in_stream, win_stream, padded_W, padded_H, Cin, pad, mode);
    serialize_output(win_stream, out_tile_stream, padded_W, padded_H, Cin, mode);
}