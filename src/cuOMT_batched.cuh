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

    bool m_use_rn_pool;

protected:
	/* batch initialization*/
	int gd_bat_init(int argc, char* argv[]);

	int gd_bat_pre_calc(int count);

	/* utility functions */
	void load_RN_from_pool(float* d_P, int count, const int num, const int dim);

    /*output pushed mu in dynamic batched gd*/
    void write_dyn_pushed_mu(const char* output, int nb);

public:
	cuOMT_batched(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, bool _no_output, bool _quiet_mode, const int _numBat) :
		cuOMT_simple(_dim, _numP, _voln, _maxIter, _eps, _lr, _no_output, _quiet_mode), numBat(_numBat)
	{
        m_use_rn_pool = false;
	};
	
	~cuOMT_batched() {};	

	// run batched gradient descent
	void run_bat_gd(int argc, char* argv[]);

    // run dynamic batched gradient descent, st. number of monte carlo samples are added when convergence rate is low
    void run_dyn_bat_gd(int argc, char* argv[]);
};