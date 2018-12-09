/*----OMT with neighbor acceleration algorithm ----*/
//#include "cuOMT.cuh"
//#include "cuOMT_simple.cuh"
//#include "cuOMT_batched.cuh"
#include "cuOMT_multi_batch.cuh"
#include <stdlib.h> 
#include <iostream> 
#include <conio.h> 
#include <time.h> 
#include <set>
using namespace std;

int main(int argc, char* argv[]) {
	
	/*----cuOMT main----*/
	/*cuOMT omt(2, 2025, 100000, 20000, 0.003, 0.001, 10);
	omt.gd_init(argc, argv);
	omt.gd_pre_calc();
	omt.tic();
	while (omt.iter < omt.maxIter && omt.gd_loop());
	std::cout << "cuOMT loop takes " << omt.toc() << "s" << std::endl;
	omt.gd_clean();
	return 0;*/
	
	/*----cuOMT_simple main----*/
	
	

	/*---------------------------------------------------------------------------------------------------------------------------

	
	cuOMT_simple simple_omt(1, 4192, 140000, 20000, 0.003, 0.001);
	simple_omt.run_simple_omt(argc, argv);
	return 0;

	-------------------------------------------------------------------------------------------------------------------------------
	*/

	///*----cuOMT_batched main----*/
 //   const int dim = 100;
 //   const int num_cell = 100000;
 //   const int num_MC_sample = 3000;
 //   const int max_iter = 60000;
 //   const double eps = 0.02 * (1 / ((float)num_cell));
 //   const double lr = 0.08;
 //   const int num_batch = 1;
	//cuOMT_batched batched_omt(dim, num_cell, num_MC_sample, max_iter, eps, lr, num_batch);
	//batched_omt.run_dyn_bat_gd(argc, argv);
	//return 0;


    /*----cuOMT multi batch main----*/
    const int dim = 100;
    const int num_total_cell = 100000;
    const int num_cell_batch_size = 10000;
    const int num_MC_sample = 3000;
    const int max_iter = 60000;
    const double eps = 0.02 * (1 / ((float)num_total_cell));
    const double lr = 0.08;
    const int num_batch = 1;
    cuOMT_multi_batch mul_bat_omt(dim, num_cell_batch_size, num_MC_sample, max_iter, eps, lr, num_batch, num_total_cell);
    mul_bat_omt.run_cuOMT_mul_bat_gd(argc, argv);
    return 0;
}

