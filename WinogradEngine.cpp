#include "WinogradEngine.h"

void input_transform(Tile4x4 d, Tile4x4 &U) {
    #pragma HLS INLINE
    data_t T[4][4];

    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        T[0][j] = d.data[0][j] - d.data[2][j];
        T[1][j] = d.data[1][j] + d.data[2][j];
        T[2][j] = d.data[2][j] - d.data[1][j];
        T[3][j] = d.data[1][j] - d.data[3][j];
    }

    for (int p = 0; p < 4; p++) {
        #pragma HLS UNROLL
        U.data[p][0] = T[p][0] - T[p][2];
        U.data[p][1] = T[p][1] + T[p][2];
        U.data[p][2] = T[p][2] - T[p][1];
        U.data[p][3] = T[p][1] - T[p][3];
    }
}

void ewmm(Tile4x4 U, Tile4x4 V, Tile4x4 &M) {
    #pragma HLS INLINE
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < 4; j++) {
            #pragma HLS UNROLL
            M.data[i][j] = U.data[i][j] * V.data[i][j];
        }
    }
}

void output_transform(Tile4x4 M, Tile2x2 &Y) {
    #pragma HLS INLINE
    int32_t S[2][4];

    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        S[0][j] = M.data[0][j] + M.data[1][j] + M.data[2][j];
        S[1][j] = M.data[1][j] - M.data[2][j] - M.data[3][j];
    }

    for (int p = 0; p < 2; p++) {
        #pragma HLS UNROLL
        int32_t y0_scaled = S[p][0] + S[p][1] + S[p][2];
        int32_t y1_scaled = S[p][1] - S[p][2] - S[p][3];
        
        Y.data[p][0] = y0_scaled;
        Y.data[p][1] = y1_scaled;
    }
}

void winograd_engine_top(
    hls::stream<Tile4x4> &in_tile_stream,  
    hls::stream<Tile4x4> &weight_v_stream, 
    hls::stream<Tile2x2> &out_tile_stream,  
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<WinogradConfig> &config_stream
) {
    #pragma HLS INTERFACE axis port=in_tile_stream
    #pragma HLS INTERFACE axis port=weight_v_stream
    #pragma HLS INTERFACE axis port=out_tile_stream

    WinogradConfig cfg = config_stream.read();
    int num_tiles = cfg.num_tiles;
    int Cin = cfg.Cin;
    int Cout = cfg.Cout;
    ap_uint<2> sel = mode_stream.read();

    if (sel != 0) {
        return;
    }

    spatial_loop: for (int t = 0; t < num_tiles; t++) {
        
        Tile2x2 acc_out[MAX_COUT];
        #pragma HLS AGGREGATE variable=acc_out

        init_acc: for(int cout = 0; cout < Cout; cout++) {
            #pragma HLS UNROLL
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                    acc_out[cout].data[i][j] = 0;
        }

        cin_loop: for (int cin = 0; cin < Cin; cin++) {
            Tile4x4 d_tile = in_tile_stream.read();
            Tile4x4 u_tile;
            
            input_transform(d_tile, u_tile); 

            cout_loop: for (int cout = 0; cout < Cout; cout++) {
                #pragma HLS PIPELINE II=1
                Tile4x4 v_tile = weight_v_stream.read();
                Tile4x4 m_tile;
                Tile2x2 y_tile;

                ewmm(u_tile, v_tile, m_tile);
                output_transform(m_tile, y_tile);

                accumulate: for(int i = 0; i < 2; i++) {
                    for(int j = 0; j < 2; j++) {
                        acc_out[cout].data[i][j] += y_tile.data[i][j];
                    }
                }
            }
        }

        write_out: for (int cout = 0; cout < Cout; cout++) {
            #pragma HLS PIPELINE II=1
            Tile2x2 final_tile;

            for(int i = 0; i < 2; i++) {
                #pragma HLS UNROLL
                for(int j = 0; j < 2; j++) {
                    #pragma HLS UNROLL
                    data_t val = acc_out[cout].data[i][j];
                    if (val < 0 && (val % 4 != 0)) {
                        final_tile.data[i][j] = (val >> 2) + 1;
                    } else {
                        final_tile.data[i][j] = val >> 2;
                    }
                }
            }
            out_tile_stream.write(final_tile);
        }
    }
}