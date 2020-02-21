/*----OMT with neighbor acceleration algorithm ----*/
#include "cuOMT_multi_batch.cuh"
#include <stdlib.h> 
#include <string>
#include <iostream> 
#include <time.h> 
#include <set>
using namespace std;
#include <sys/stat.h>
#include <cstdlib>

int make_dir(const char* path) {
	std::string cmd = std::string("mkdir -p ") + std::string(path);
	const int dir_err = system(cmd.c_str());
	if (-1 == dir_err)
	{
		return 1;
	}
	return 0;
}


int clear_cache() {
	system("exec rm -r ./h/*");
	system("exec rm -r ./adam_m/*");
	system("exec rm -r ./adam_v/*");
	system("exec rm -r ./pushed_mu/*");
	system("exec rm -r ./volP/*");
	system("exec rm -r ./ind/*");
	return 0;
}

struct  arguments
{
	/* flags */
	bool simple_gd;
	bool batch_gd;
	bool mul_batch_gd;
	bool no_output;
	bool quiet_mode;
	/* data */
	int dim;
	int num_cell;
	int num_cell_batch_size;
	int num_MC_sample;
	int max_iter;
	double eps;
	double lr;
	int num_batch;
};

int main(int argc, char* argv[]) {
	//parse arguments
	arguments args = {
		false, //simgle gd
		false, //batch gd
		false, //mul_batch_gd
		false, //no_output
		false, //quiet_mode
		2, //dim
		20, //num_cell
		1, //cell_batch_size
		20000, //num_MC_sample
		5000, //max_iter
		0.05, //eps
		0.001, //lr
		1, //num_batch
	};

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "clean") == 0)
		{
			clear_cache();
            std::cout << "Cleaned cached files." << std::endl;
            return 0;	
		}

		if (strcmp(argv[i], "simple") == 0)
		{
			args.simple_gd = true;
            std::cout << "Gradient descent mode: SIMPLE" << std::endl;	
		}

		if (strcmp(argv[i], "batch") == 0)
		{
			args.batch_gd = true;
            std::cout << "Gradient descent mode: BATCH" << std::endl;	
		}

		if (strcmp(argv[i], "multi") == 0)
		{
			args.mul_batch_gd = true;
            std::cout << "Gradient descent mode: MULTI" << std::endl;	
		}

		if (strcmp(argv[i], "--no_output") == 0)
		{
			args.no_output = true;
            // std::cout << "Gradient descent mode: MULTI" << std::endl;	
		}

		if (strcmp(argv[i], "--quiet_mode") == 0)
		{
			args.quiet_mode = true;
            // std::cout << "Gradient descent mode: MULTI" << std::endl;	
		}

		if (strcmp(argv[i], "-dim") == 0)
		{
			args.dim = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded dimension: " << args.dim << std::endl;	
		}

		if (strcmp(argv[i], "-num_cell") == 0)
		{
			args.num_cell = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded num_cell: " << args.num_cell << std::endl;	
		}

		if (strcmp(argv[i], "-num_cell_batch_size") == 0)
		{
			args.num_cell_batch_size = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded num_cell_batch_size: " << args.num_cell_batch_size << std::endl;	
		}

		if (strcmp(argv[i], "-num_MC_sample") == 0)
		{
			args.num_MC_sample = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded num_MC_sample: " << args.num_MC_sample << std::endl;	
		}

		if (strcmp(argv[i], "-max_iter") == 0)
		{
			args.max_iter = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded max_iter: " << args.max_iter << std::endl;	
		}

		if (strcmp(argv[i], "-eps") == 0)
		{
			args.eps = std::stod(std::string(argv[i+1]));
            std::cout << "Loaded eps: " << args.eps << std::endl;	
		}

		if (strcmp(argv[i], "-lr") == 0)
		{
			args.lr = std::stod(std::string(argv[i+1]));
            std::cout << "Loaded lr: " << args.lr << std::endl;	
		}

		if (strcmp(argv[i], "-num_batch") == 0)
		{
			args.num_batch = std::stoi(std::string(argv[i+1]));
            std::cout << "Loaded num_batch: " << args.num_batch << std::endl;	
		}
	}

	if (!args.simple_gd && !args.batch_gd && !args.mul_batch_gd)
	{
		std::cout << "No gradient method specified. Falling back to simple method." <<std::endl;
		args.simple_gd = true;
	}
	
	/*----cuOMT main----*/
	// make dirs
	make_dir("./adam_m");
	make_dir("./adam_v");
	make_dir("./h");
	make_dir("./ind");
	make_dir("./pushed_mu");
	make_dir("./volP");


	const int dim = args.dim;
    const int num_cell = args.num_cell;
    const int num_cell_batch_size = args.num_cell_batch_size;
    const int num_MC_sample = args.num_MC_sample;
    const int max_iter = args.max_iter;
    const double eps = args.eps * (1 / ((float)args.num_cell));
    const double lr = args.lr;
    const int num_batch = args.num_batch;
	
	if (args.simple_gd){
		/*----cuOMT_simple main----*/
		cuOMT_simple simple_omt(dim, num_cell, num_MC_sample, max_iter, eps, lr, args.no_output, args.quiet_mode);
		simple_omt.run_simple_omt(argc, argv);
		return 0;
	}	
	else if (args.batch_gd){
		//*----cuOMT_batched main----*/
		cuOMT_batched batched_omt(dim, num_cell, num_MC_sample, max_iter, eps, lr, args.no_output, args.quiet_mode, num_batch);
		batched_omt.run_dyn_bat_gd(argc, argv);
		return 0;
	}
	else if (args.mul_batch_gd){
    /*----cuOMT multi batch main----*/
    const int num_total_cell = args.num_cell;
    cuOMT_multi_batch mul_bat_omt(dim, num_cell_batch_size, num_MC_sample, max_iter, eps, lr, args.no_output, args.quiet_mode, num_batch, num_total_cell);
    mul_bat_omt.run_cuOMT_mul_bat_gd(argc, argv);
    return 0;
	}

	return 0;
}

