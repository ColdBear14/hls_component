#ifndef tb_top_system_H
#define tb_top_system_H

#include "helper.h"

struct TestScenario {
    int mode;           // 0: Wino 3x3 S1, 1: Sys 3x3 S2, 2: Sys 1x1 S1
    int img_w;
    int img_h;
    int pad;
    int cin;
    int cout;
    bool has_residual;
    std::string desc;
};

int run_top_system_test(int mode, int IMG_W, int IMG_H, int pad, int Cin, int Cout, bool has_residual);

void TopSystem_TB();
#endif // tb_top_system_H