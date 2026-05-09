#include "tb_winograd.h"
#include "tb_top_system.h"
#include "tb_systolic.h"
#include "tb_linebuffer.h"
#include "tb_weightram.h"


int main() {    
    // WinogradEngine_TB();

    // SystolicEngine_TB();

    // tb_linebuffer();

    // tb_weight_ram();

    TopSystem_TB();
    
    return 0;
}