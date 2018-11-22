//Various gradient descent methods can be found at: http://cs231n.github.io/neural-networks-3/

#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

#include <time.h>
#include <stack>
#include <math.h>

#include <iostream>
#include <stdio.h>
#include <fstream>      // std::ifstream
#include <sstream> 

#include "OMT_functors.cuh"

#define USE_FANCY_GD
class cuOMT_simple
{
protected:

	/*problem settings*/
	const char* input_P;
	const int numP;
	const int dim;
	const int voln;
	const float eps;
	const float lr;

	thrust::device_vector<float> d_eps;

	/*allocate device vectors*/
	float *d_P;
	float *d_volP;
	
	float *d_U;
	float *d_PX;
	int* d_cellId_numP;
    
#ifdef USE_FANCY_GD
    float *d_cache;
    float *d_adam_m;
    float *d_adam_v;
#endif

	// P
	thrust::device_ptr<float> d_P_ptr;

	// volP
	thrust::device_ptr<float> d_volP_ptr;

	// h
	thrust::device_vector<float> d_h;


	// A
	thrust::device_vector<float> d_A;
#ifdef USE_FANCY_GD
	// cache
	thrust::device_ptr<float> d_cache_ptr;
    thrust::device_ptr<float> d_adam_m_ptr;
    thrust::device_ptr<float> d_adam_v_ptr;
#endif 
	// ind
	thrust::device_vector<int> d_ind;
	thrust::device_vector<float> d_ind_val;

	// g
	thrust::device_vector<float> d_g;



	/*---gradient descent---*/
	cublasHandle_t handle;
	// parameters	
	const float alf = 1;
	const float bet = 1;
	const float *alpha = &alf;
	const float *beta = &bet;
	const float n2 = -2.0f;
	const float *neg_2 = &n2;
	const float ze = 0;
	const float *zero = &ze;


	thrust::host_vector<float> h_g_norm;
	thrust::host_vector<float> h_time;
	thrust::device_vector<float> d_g_norm;


	thrust::device_ptr<float> d_U_ptr;
	thrust::device_ptr<float> d_PX_ptr;
	thrust::device_vector<float> d_delta_h;


	// loop parameters
	thrust::device_vector<int> d_voln_key;

	// sample_id
	thrust::device_vector<int> d_sampleId;

	// cell_id
	thrust::device_vector<int> d_cellId;

	// duplicate of histogram by dimension
	thrust::device_vector<float> d_Hist;

	// dummy key for reduction in calculating knn
	thrust::device_vector<int> d_knn_dummy_key;

	// voln * numP stride used to replicate vector dim times
	thrust::device_ptr<int> d_numP_voln_rep_ptr;


	/*random matrix generation*/
	int GPU_generate_RNM(float* P, const int nRowP, const int nColP);

	/*random matrix generation, each element uniformly sampled between [a,b]*/
	int GPU_generate_RNM(float* P, const int nRowP, const int nColP, float lower_b, float upper_b);

    /*generate sobol sequence*/
    int curand_RNG_sobol(float* P, const int nRowP, const int nColP, unsigned long long offset);

	/*debugging functions*/
	std::stack<clock_t> tictoc_stack;

protected:
	template <typename Vector1, typename Vector2>
	void dense_histogram(Vector1& input, Vector2& histogram);

	template <typename T>
	void print_matrix(T* A, int nr_rows_A, int nr_cols_A);

	template <typename T>
	void print_matrix_csv(T* A, int nr_rows_A, int nr_cols_A, const char* output);


public:
	cuOMT_simple(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr);
	~cuOMT_simple() {};

	/*gradient descent*/
	int gd_init(int argc, char* argv[]);
	int gd_pre_calc();
	int gd_calc_measure();
	int gd_update_h();
	int gd_clean();

	void tic();
	double toc();

	void run_simple_omt(int argc, char* argv[]);

	/*output pushed measure*/
	void write_pushed_mu(const char* output);

	/*output current h*/
	void write_h(const char* output);
#ifdef USE_FANCY_GD
    /*output cache*/
    void write_cache(const char* output);
    void write_adam_m(const char* output);
    void write_adam_v(const char* output);
#endif
	void write_generated_P(const char* output);

	/*set parameters*/
	// set device vector from csv file
	template <typename T>
	void _set_from_csv(const char* input, T* d_vec, int row, int col);

	// set parameter of size row x col randomly between [0, 1] uniform
	template <typename T>
	void _set_random_parameter(T* d_para, const int row, const int col);

	/*get parameters*/
	template <typename T>
	void _get_to_csv(const char* output, T* d_vec, int row, int col);

	

public:
	const int maxIter;
	int iter;
};