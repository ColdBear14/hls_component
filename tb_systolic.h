#ifndef tb_systolic_H
#define tb_systolic_H

#include "helper.h"

void SystolicEngine_TB();

void run_systolic_test_generic(std::string test_name, int Cin, int Cout, int tiles, 
                                pixel_t pixel_val, weight_t w_val, int mode_sel);
#endif // tb_systolic_H