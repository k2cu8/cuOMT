/*----OMT with neighbor acceleration algorithm ----*/
//#include "cuOMT.cuh"
//#include "cuOMT_simple.cuh"
#include "cuOMT_batched.cuh"
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

	/*float low = -50.0f;
	float high = 50.0f;
	long NumberOfSample = 5000000;
	long length = 3 * NumberOfSample;
	
	float *SampleCoord;
	SampleCoord = new float[length];
	
	float r = 0.0f;
	set<float> Sample;

	for (long i = 0; i < 999999; i++)
	{
		r = low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (high - low)));
		cout << r << endl;

		Sample.insert(r);
	}

	cout << length<<endl;
	cout << Sample.size() << endl;
	int size;
	cin >> size;
*/

	/*----cuOMT_batched main----*/
    const int dim = 100;
    const int num_cell = 8000;
    const int num_MC_sample = 30000;
    const int max_iter = 60000;
    const double eps = 0.02 * (1 / ((float)num_cell));
    const double lr = 0.08;
    const int num_batch = 1;
	cuOMT_batched batched_omt(dim, num_cell, num_MC_sample, max_iter, eps, lr, num_batch);
	batched_omt.run_dyn_bat_gd(argc, argv);
	return 0;

	/*-----main for SAG-----
	const char* input_P(argv[1]);
	const int dim(std::stoi(argv[2]));
	const int batSize(std::stoi(argv[3]));
	const int numBat(std::stoi(argv[4]));
	const int maxIter(std::stoi(argv[5]));
	run_SAG(input_P, dim, batSize, numBat, maxIter);*/
	
}

