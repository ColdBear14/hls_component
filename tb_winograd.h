#ifndef tb_winograd_H
#define tb_winograd_H

#include "helper.h"

void WinogradEngine_TB();

void run_single_test(std::string test_name, Tile4x4 d, Tile3x3 g);

void test_multiple_tiles(int num_tiles);

#endif // tb_winograd_H