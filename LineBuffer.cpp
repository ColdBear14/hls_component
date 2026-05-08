#include "LineBuffer.h"

void line_buffer(
    hls::stream<pixel_t> &in_stream,       
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

    // KHAI BÁO MẢNG LINE_BUF
    static pixel_t line_buf[3][MAX_LINE_BUFFER_SIZE];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
    
    // KHAI BÁO MẢNG WINDOW (Lưu trên BRAM)
    static pixel_t window[4][4][MAX_CIN];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window complete dim=2
    #pragma HLS BIND_STORAGE variable=window type=ram_2p impl=bram

    int total_iters = (padded_W * padded_H) * Cin;
    int col = 0, row = 0, cin = 0;

    // BIẾN CHO LOGIC DATA FORWARDING (Chống lỗi RAW khi Cin = 1)
    pixel_t prev_window_val[4][4];
    #pragma HLS ARRAY_PARTITION variable=prev_window_val complete dim=0
    int prev_cin = -1;

    process_loop: for (int i = 0; i < total_iters; i++) {
        #pragma HLS PIPELINE II=1
        
        // Vẫn phải giữ dependence false để compiler bỏ qua cảnh báo
        #pragma HLS dependence variable=line_buf type=inter false
        #pragma HLS dependence variable=window type=inter false

        pixel_t new_pixel;

        bool is_pad_top = (row < pad);
        bool is_pad_bottom = (row >= img_height + pad);
        bool is_pad_left = (col < pad);
        bool is_pad_right = (col >= img_width + pad);

        if (is_pad_top || is_pad_bottom || is_pad_left || is_pad_right) {
            new_pixel = 0;
        } else {
            new_pixel = in_stream.read();
        }
        
        // TÍNH CHỈ SỐ 1D CHO MẢNG LINE_BUF
        int idx = col * Cin + cin;

        // BƯỚC 1: ĐỌC DỮ LIỆU CŨ VỚI LOGIC BYPASSING (Data Forwarding)
        pixel_t read_col[4][4];
        #pragma HLS ARRAY_PARTITION variable=read_col complete dim=0

        for (int r = 0; r < 4; r++) {
            #pragma HLS UNROLL
            // Chỉ cần đọc cột 1, 2, 3 vì cột 0 sẽ bị ghi đè hoàn toàn
            for (int c = 1; c < 4; c++) {
                #pragma HLS UNROLL
                if (cin == prev_cin) {
                    // Nếu Cin=1, lấy thẳng dữ liệu từ Register tạm của cycle trước (Bypass BRAM)
                    read_col[r][c] = prev_window_val[r][c];
                } else {
                    // Nếu bình thường, đọc từ BRAM
                    read_col[r][c] = window[r][c][cin];
                }
            }
        }

        // BƯỚC 2: CHUẨN BỊ MẢNG DỮ LIỆU MỚI SẼ GHI Ở CYCLE NÀY
        pixel_t write_val[4][4];
        #pragma HLS ARRAY_PARTITION variable=write_val complete dim=0

        // Thực hiện logic Shift: Cột 0 lấy 1, Cột 1 lấy 2, Cột 2 lấy 3
        for (int r = 0; r < 4; r++) {
            #pragma HLS UNROLL
            write_val[r][0] = read_col[r][1];
            write_val[r][1] = read_col[r][2];
            write_val[r][2] = read_col[r][3];
        }

        // Cập nhật Cột 3 từ line_buf và new_pixel
        write_val[0][3] = line_buf[0][idx];
        write_val[1][3] = line_buf[1][idx];
        write_val[2][3] = line_buf[2][idx];
        write_val[3][3] = new_pixel;

        // BƯỚC 3: CẬP NHẬT LẠI LINE_BUF VÀ ĐẨY VÀO WINDOW
        line_buf[0][idx] = line_buf[1][idx];
        line_buf[1][idx] = line_buf[2][idx];
        line_buf[2][idx] = new_pixel;

        for (int r = 0; r < 4; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 4; c++) {
                #pragma HLS UNROLL
                // Ghi vào BRAM
                window[r][c][cin] = write_val[r][c];
                // Lưu lại vào thanh ghi tạm cho chu kỳ tiếp theo
                prev_window_val[r][c] = write_val[r][c];
            }
        }
        
        // Cập nhật lại prev_cin
        prev_cin = cin;

        // BƯỚC 4: XUẤT OUTPUT TILE
        bool valid_output = false;
        if (mode == 2) valid_output = true;
        else if (row >= 3 && col >= 3) valid_output = ((col & 1) == 1) && ((row & 1) == 1);

        if (valid_output) {
            Tile4x4 out_tile;
            for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 4; c++) {
                    #pragma HLS UNROLL
                    // Thay vì đọc lại từ window, lấy trực tiếp từ mảng write_val 
                    // để giảm tải Read Port cho BRAM
                    out_tile.data[r][c] = (mode == 2) ? write_val[3][3] : write_val[r][c];
                }
            }
            out_tile_stream.write(out_tile);
        }

        // BƯỚC 5: TĂNG BỘ ĐẾM
        cin++;
        if (cin == Cin) {
            cin = 0;
            if (col == padded_W - 1) {
                col = 0;
                row++;
            } else {
                col++;
            }
        }
    }
}