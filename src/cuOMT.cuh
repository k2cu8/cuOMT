#include <cuda.h>
#include <cuda_device_runtime_api.h>
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

//#include "histogram.cuh"
//#include "knn.cuh"
#include "OMT_functors.cuh"

class cuOMT
{
protected:

	/*problem settings*/
	const char* input_P;
	const int numP;
	const int dim;
	const int voln;
	
	const float eps;
	const float lr;
	const int nnb;

	thrust::device_vector<float> d_eps;

	/*allocate device vectors*/
	float *d_P;
	float *d_volP;
	float *d_volP_data;
	float *d_cache;
	float *d_U;
	float *d_PX;
	int* d_Ind;
	float* d_pos;
	float* d_X;
	float* d_pos_norm;
	float* d_pos_Norm;
	float* d_pos_NormT;
	int* d_pos_stride;
	int* d_pos_stride_data;
	int* d_voln_dim_rep;
	int* d_numP_dim_rep;
	int* d_numP_numP_rep;
	int* d_nb;
	int* d_knn_range;
	int* d_cellId_numP;
	int* d_nb_vol;
	int* d_nb_vol_key;

	float* d_U_tilt;
	float* d_PX_tilt;



	// P
	thrust::device_ptr<float> d_P_ptr;

	// h
	thrust::device_vector<float> d_h;

	// target measure
	thrust::device_vector<float> d_A;

	// cache
	thrust::device_ptr<float> d_cache_ptr;

	// ind MC vertex belongs to which cell index
	thrust::device_vector<int> d_ind;
	thrust::device_vector<float> d_ind_val;

	// gradient vector
	thrust::device_vector<float> d_g;

	// each cell's neighbors' id
	thrust::device_ptr<int> d_nb_ptr;

	// each MC vertex's neighbors
	thrust::device_ptr<int> d_nb_vol_ptr;
	thrust::device_ptr<int> d_nb_vol_key_ptr;

	// Center of each cell's position
	thrust::device_ptr<float> d_pos_ptr;

	/*---gradient descent---*/
	cublasHandle_t handle;
	// parameters	
	//alpha beta are scale parameters which are used for matrix multuply
	const float alf = 1;
	const float bet = 1;
	const float *alpha = &alf;
	const float *beta = &bet;
	//Negative 2
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

	thrust::device_ptr<float> d_U_tilt_ptr;
	thrust::device_ptr<float> d_PX_tilt_ptr;

	// loop parameters
	thrust::device_vector<int> d_voln_key;

	// sample_id
	thrust::device_vector<int> d_sampleId;

	// sample_id_nnb
	thrust::device_vector<int> d_sampleId_nnb;

	// cell_id
	thrust::device_vector<int> d_cellId;

	// dummy key for reduction in calculating knn
	thrust::device_vector<int> d_knn_dummy_key;

	// duplicate of histogram by dimension
	thrust::device_vector<float> d_Hist;

	// pos key 
	thrust::device_vector<int> d_pos_key;

	// pos value
	thrust::device_vector<float> d_pos_val;

	// voln * dim stride used to replicate vector dim times
	thrust::device_ptr<int> d_voln_dim_rep_ptr;

	// voln * numP stride used to replicate vector dim times
	thrust::device_ptr<int> d_numP_voln_rep_ptr;

	thrust::device_ptr<int> d_nnb_voln_rep_ptr;

	// volP
	thrust::device_ptr<float> d_volP_ptr;

	/*random matrix generation*/
	int GPU_generate_RNM(float* P, const int nRowP, const int nColP, unsigned int seed);

	int GPU_generate_RNM(float* P, const int nRowP, const int nColP);


	/*debugging functions*/
	std::stack<clock_t> tictoc_stack;

protected:
	//Histogram function: Calulate the MC vetex in each cell
	template <typename Vector1, typename Vector2>
	void dense_histogram(Vector1& input, Vector2& histogram);

	template <typename T>
	void print_matrix(T* A, int nr_rows_A, int nr_cols_A);

	template <typename T>
	void print_matrix_csv(T* A, int nr_rows_A, int nr_cols_A, const char* output);

	
	//each cell's center position (x,y,z)
	int compute_pos(cublasHandle_t &handle, float* d_volP, float* d_volP_data, int voln, int* d_ind, int dim, int* d_Ind, int* d_voln_dim_rep, int numP,
		const float* alf, int* d_numP_dim_rep, int* d_pos_key, float* d_pos_val, float* d_hist, float* d_Hist, float* d_pos);
	//each cell's k-nearest neighbors
	int compute_knn(cublasHandle_t &handle, float* d_pos, float *d_X, int numP, int dim, const int k, float* d_pos_norm, float* d_pos_Norm, float* d_pos_NormT,
		int* d_pos_stride, int* d_pos_stride_data, int* d_knn_dummy_key, int* d_numP_numP_rep, const float* alpha, const float* beta, const float* neg_2, const float* zero,
		int* d_knn_range, int* d_cellId_numP, const int voln, int* d_ind, int *d_nb);


public:
	cuOMT(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, const int _nnb);
	~cuOMT() {};

	/*gradient descent*/
	int gd_init(int argc, char* argv[]);
	int gd_pre_calc();
	int gd_loop();
	int gd_clean();
	

	void tic();
	double toc();

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