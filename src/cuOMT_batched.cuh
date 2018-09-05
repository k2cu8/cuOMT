#include "cuOMT_simple.cuh"

class cuOMT_batched : public cuOMT_simple
{
protected:
	//number of batches, must be a square number
	const int numBat; 
	//aggregate measures
	thrust::device_vector<float> d_g_sum;

	//random point pool
	thrust::host_vector<float> h_rn_pool;

protected:
	/* batch initialization*/
	int gd_bat_init(int argc, char* argv[]);

	int gd_bat_pre_calc(int count);

	/* utility functions */
	void load_RN_from_pool(float* d_P, int count, const int num, const int dim);

public:
	cuOMT_batched(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, const int _numBat) :
		cuOMT_simple(_dim, _numP, _voln, _maxIter, _eps, _lr), numBat(_numBat)
	{
		
	};
	
	~cuOMT_batched() {};	

	// run batched gradient descent
	void run_bat_gd(int argc, char* argv[]);
};