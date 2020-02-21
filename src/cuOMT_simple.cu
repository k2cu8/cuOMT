#include "cuOMT_simple.cuh"
#include <curand.h>
#include <iomanip>



int cuOMT_simple::GPU_generate_RNM(float* P, const int nRowP, const int nColP)
{
	int seed = (int)clock();
	thrust::device_ptr<float> P_ptr(P);
	thrust::transform(
		thrust::make_counting_iterator(seed),
		thrust::make_counting_iterator(seed + nRowP * nColP),
		P_ptr,
		GenRandFloat());
	return 0;
}

int cuOMT_simple::GPU_generate_RNM(float* P, const int nRowP, const int nColP, float lower_b, float upper_b)
{
	int seed = (int)clock();
	thrust::device_ptr<float> P_ptr(P);
	thrust::transform(
		thrust::make_counting_iterator(seed),
		thrust::make_counting_iterator(seed + nRowP * nColP),
		P_ptr,
		GenRandFloatUni(lower_b, upper_b));
	return 0;
}

int cuOMT_simple::curand_RNG_sobol(float* P, const int nRowP, const int nColP, unsigned long long offset)
{
    using namespace std;
    curandStatus_t curandResult;
    curandGenerator_t qrng;

    curandResult = curandCreateGenerator(&qrng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not create quasi-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = curandSetQuasiRandomGeneratorDimensions(qrng, nColP);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not set number of dimensions for quasi-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = curandSetGeneratorOrdering(qrng, CURAND_ORDERING_QUASI_DEFAULT);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not set order for quasi-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = curandSetGeneratorOffset(qrng, offset);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not set offset for quasi-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = curandGenerateUniform(qrng, P, nRowP * nColP);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not generate quasi-random numbers: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = curandDestroyGenerator(qrng);
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not destroy quasi-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    thrust::device_ptr<float> P_ptr(P);
    thrust::transform(P_ptr, P_ptr + nRowP * nColP, P_ptr, axpb<float>(1, 0.5));

    return 0;
}

int cuOMT_simple::gd_init(int argc, char* argv[])
{
	/*set problem parameters*/
	d_eps.resize(1);
	d_eps[0] = eps;

	/*allocate device vectors*/
	d_P = 0;
	d_volP = 0;
	
	d_U = 0;
	d_PX = 0;
#ifdef USE_FANCY_GD
    d_adam_m = 0;
    d_adam_v = 0;
#endif

	// printf("cuOMT running...\n");

	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	/* allocate device vectors*/
	if (cudaMalloc((void **)&d_P, numP * dim * sizeof(d_P[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (allocate P)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_volP, voln * dim * sizeof(d_volP[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (allocate volP)\n");
		return EXIT_FAILURE;
	}
#ifdef USE_FANCY_GD
    if (cudaMalloc((void **)&d_adam_m, numP * sizeof(d_adam_m[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_adam_m)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_adam_v, numP * sizeof(d_adam_v[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_adam_v)\n");
        return EXIT_FAILURE;
    }
#endif
	if (cudaMalloc((void **)&d_U, numP * voln * sizeof(d_U[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (U)\n");
		return EXIT_FAILURE;
	}


	/*fill in data*/

	// set parameters from command line arguments: P, A, h, adam_m, adam_v
	d_P_ptr = thrust::device_pointer_cast(d_P);
	d_A.resize(numP);
	d_h.resize(numP);
	bool set_P(false), set_A(false), set_h(false);
#ifdef USE_FANCY_GD
    d_adam_m_ptr = thrust::device_pointer_cast(d_adam_m);
    d_adam_v_ptr = thrust::device_pointer_cast(d_adam_v);
    bool set_adam_m(false), set_adam_v(false);
#endif
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-P") == 0)
		{
			_set_from_csv<float>(argv[i + 1], d_P, numP, dim);
			set_P = true;

            std::cout << "Read P successfully." << std::endl;	
		}

		if (strcmp(argv[i], "-A") == 0)
		{
			_set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_A[0]), 1, numP);
			set_A = true;

            std::cout << "Read target measure successfully." << std::endl;
		}

		if (strcmp(argv[i], "-h") == 0)
		{
			_set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_h[0]), 1, numP);
			set_h = true; 

            std::cout << "Read initial h successfully." << std::endl;
		}
#ifdef USE_FANCY_GD
        if (strcmp(argv[i], "-adam_m") == 0)
        {
            _set_from_csv<float>(argv[i + 1], d_adam_m, 1, numP);
            set_adam_m = true;


            std::cout << "Read initial adam_m successfully." << std::endl;
        }

        if (strcmp(argv[i], "-adam_v") == 0)
        {
            _set_from_csv<float>(argv[i + 1], d_adam_v, 1, numP);
            set_adam_v = true;


            std::cout << "Read initial adam_v successfully." << std::endl;
        }
#endif
	}
	if (!set_P)
	{
		/*_set_random_parameter(d_P, numP, dim);
		thrust::transform(d_P_ptr, d_P_ptr + numP * dim, d_P_ptr, axpb<float>(1, -0.5));*/
		GPU_generate_RNM(d_P, numP, dim, -0.5f, 0.5f);

	}
	if (!set_A)
	{
		thrust::fill(d_A.begin(), d_A.end(), 1.0f / numP);
	}
	if (!set_h)
	{
		thrust::fill(d_h.begin(), d_h.end(), 0.0f);
	}
#ifdef USE_FANCY_GD
    if (!set_adam_m)
    {
        thrust::fill(d_adam_m_ptr, d_adam_m_ptr + numP, 0.0f);
    }
    if (!set_adam_v)
    {
        thrust::fill(d_adam_v_ptr, d_adam_v_ptr + numP, 0.0f);
    }
#endif
	// volP
	d_volP_ptr = thrust::device_pointer_cast(d_volP);

	// ind
	d_ind.resize(voln);
	thrust::fill(d_ind.begin(), d_ind.end(), -1);
	d_ind_val.resize(voln);
	thrust::fill(d_ind_val.begin(), d_ind_val.end(), 0.0f);

	// g
	d_g.resize(numP);



	/*---gradient descent---*/
	iter = 0;
	d_g_norm.resize(1);

	d_U_ptr = thrust::device_pointer_cast(d_U);

	d_delta_h.resize(numP);

	// loop parameters
	d_voln_key.resize(voln);

	// sample_id
	d_sampleId.resize(numP*voln);
	thrust::host_vector<int> h_sampleId(numP * voln);
	for (int i = 0; i < numP; ++i)
		for (int j = 0; j < voln; ++j)
		{
			h_sampleId[i + j * numP] = j;
		}
	thrust::copy(h_sampleId.begin(), h_sampleId.end(), d_sampleId.begin());
	h_sampleId.clear();


	// cell_id
	d_cellId.resize(numP*voln);
	thrust::host_vector<int> h_cellId(numP * voln);
	for (int i = 0; i < numP; ++i)
		for (int j = 0; j < voln; ++j)
		{
			h_cellId[i + j * numP] = i;
		}
	thrust::copy(h_cellId.begin(), h_cellId.end(), d_cellId.begin());
	h_cellId.clear();


	// duplicate of histogram by dimension
	d_Hist.resize(numP * dim);

	// dummy key for reduction in calculating knn
	d_knn_dummy_key.resize(numP);

	// voln * numP stride used to replicate vector dim times
	d_numP_voln_rep_ptr = thrust::device_pointer_cast(thrust::raw_pointer_cast(&d_cellId[0]));

	tic();

	return 0;
}

int cuOMT_simple::gd_pre_calc()
{
	//tic();

	// fill volP with random numbers
	
	GPU_generate_RNM(d_volP, voln, dim, -.5f, .5f);

	return 0;
}

int cuOMT_simple::gd_calc_measure()
{
	// duplicate h, i.e repmat(h, [voln 1])
	thrust::gather(d_numP_voln_rep_ptr, d_numP_voln_rep_ptr + numP*voln, d_h.begin(), d_U_ptr);


	// PX+H
	//thrust::transform(d_U_ptr, d_U_ptr + numP * voln, d_PX_ptr, d_U_ptr, thrust::plus<float>());
	if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numP, voln, dim, alpha, d_P, numP, d_volP, voln, alpha, d_U, numP) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! Device matrix multiplication error (U <- PX + H)\n");
		return EXIT_FAILURE;
	}


	// find max parallelly
	thrust::reduce_by_key(thrust::device, d_sampleId.begin(), d_sampleId.begin() + numP * voln, thrust::make_zip_iterator(thrust::make_tuple(d_cellId.begin(), d_U_ptr)), d_voln_key.begin(),
		thrust::make_zip_iterator(thrust::make_tuple(d_ind.begin(), d_ind_val.begin())), thrust::equal_to<int>(), find_max());

	// calculate histogram
	dense_histogram(d_ind, d_g);


	return 0;
}


int cuOMT_simple::gd_update_h()
{
	// subtract current cell measures with target ones
	thrust::transform(d_g.begin(), d_g.end(), d_A.begin(), d_g.begin(), axmy<float>(1.0f / voln));

	/*update h from g*/
#ifdef USE_FANCY_GD
    thrust::transform(thrust::device, d_adam_m_ptr, d_adam_m_ptr + numP, d_g.begin(), d_adam_m_ptr, update_adam_m<float>(0.9f));
    thrust::transform(thrust::device, d_adam_v_ptr, d_adam_v_ptr + numP, d_g.begin(), d_adam_v_ptr, update_adam_v<float>(0.999f));
    thrust::transform(thrust::device, d_adam_m_ptr, d_adam_m_ptr + numP, d_adam_v_ptr, d_delta_h.begin(), adam_delta_h(lr));
    thrust::transform(thrust::device, d_h.begin(), d_h.begin() + numP, d_delta_h.begin(), d_h.begin(), thrust::plus<float>());
#else
    thrust::transform(thrust::device, d_g.begin(), d_g.end(), d_delta_h.begin(), axpb<float>(-lr, 0.0f));
    thrust::transform(thrust::device, d_h.begin(), d_h.begin() + numP, d_delta_h.begin(), d_h.begin(), thrust::plus<float>());
#endif

	thrust::transform(d_h.begin(), d_h.end(), d_h.begin(), axpb<float>(1.0f, thrust::transform_reduce(d_h.begin(), d_h.end(), axpb<float>(-1.0f / (float)numP, 0.0f), 0.0f, thrust::plus<float>())));


	/*terminate condition*/
	d_g_norm[0] = sqrt(thrust::transform_reduce(d_g.begin(), d_g.end(), square<float>(), 0.0f, thrust::plus<float>()));
	//if (iter % 100 == 0)
	if (!quiet_mode)
		std::cout << "[" << iter << "/" << maxIter << "] g norm: " << d_g_norm[0] << "/" << eps << std::endl;
	if (d_g_norm[0] < d_eps[0])
		return 0;
	else ++iter;


	return 1;

}

void cuOMT_simple::write_pushed_mu(const char* output)
{
	// calculate histogram
	dense_histogram(d_ind, d_g);
	thrust::transform(d_g.begin(), d_g.end(), d_g.begin(), axpb<float>(1.0f / voln, 0));

	_get_to_csv(output, thrust::raw_pointer_cast(&d_g[0]), 1, numP);
}

void cuOMT_simple::write_h(const char* output)
{
	_get_to_csv(output, thrust::raw_pointer_cast(&d_h[0]), 1, numP);
}

void cuOMT_simple::write_volP(const char* output)
{
	_get_to_csv(output, thrust::raw_pointer_cast(&d_volP[0]), voln, dim);
}

void cuOMT_simple::write_ind(const char* output)
{
	_get_to_csv(output, thrust::raw_pointer_cast(&d_ind[0]), 1, voln);
}
#ifdef USE_FANCY_GD
void cuOMT_simple::write_adam_m(const char* output)
{
    _get_to_csv(output, d_adam_m, 1, numP);
}

void cuOMT_simple::write_adam_v(const char* output)
{
    _get_to_csv(output, d_adam_v, 1, numP);
}
#endif
void cuOMT_simple::write_generated_P(const char* output)
{
	_get_to_csv(output, d_P, numP, dim);
}

int cuOMT_simple::gd_clean(void)
{
	/* Memory clean up */

	if (cudaFree(d_U) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (U)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_volP) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (volP)\n");
		return EXIT_FAILURE;
	}

#ifdef USE_FANCY_GD
    if (cudaFree(d_adam_m) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (adam_m)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_adam_v) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (adam_v)\n");
        return EXIT_FAILURE;
    }
#endif

	/* shut down*/
	if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	std::cout << "cuOMT finished successfully!\n(Press any key to continue...)" << std::endl;
	std::cin.get();
	return 0;

}

template <typename Vector1, typename Vector2>
void cuOMT_simple::dense_histogram(Vector1& input, Vector2& histogram)
{
	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector2::value_type IndexType; // histogram index type

													// copy input data (could be skipped if input is allowed to be modified)
													//thrust::device_vector<ValueType> data(input);

													// print the initial data
													//print_vector("initial data", data);

													// sort data to bring equal elements together
	thrust::device_vector<int> data(input);
	thrust::sort(data.begin(), data.end());

	// print the sorted data
	//print_vector("sorted data", input);

	// number of histogram bins is equal to the maximum value plus one
	int num_bins = data.back() + 1;

	// resize histogram storage
	//histogram.resize(num_bins);

	// find the end of each bin of values
	//thrust::counting_iterator<IndexType> search_begin(0);
	thrust::upper_bound(data.begin(), data.end(),
		thrust::make_counting_iterator<IndexType>(0), thrust::make_counting_iterator<IndexType>(num_bins),
		histogram.begin());

	// print the cumulative histogram
	//print_vector("cumulative histogram", histogram);

	// compute the histogram by taking differences of the cumulative histogram
	thrust::adjacent_difference(histogram.begin(), histogram.begin() + num_bins,
		histogram.begin());
	// fill outbounded historgram entry to 0
	if (num_bins < histogram.size())
		thrust::fill(histogram.begin() + num_bins, histogram.end(), 0.0f);

	// print the histogram
	//print_vector("histogram", histogram);
}

template <typename T>
void cuOMT_simple::print_matrix(T* A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";

		}
		std::cout << std::endl;

	}
	std::cout << std::endl;
}

template <typename T>
void cuOMT_simple::print_matrix_csv(T* A, int nr_rows_A, int nr_cols_A, const char* output) {
	std::ofstream file;
	file.open(output);
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			if (j != nr_cols_A - 1)
				file << std::setprecision(20) << A[j * nr_rows_A + i] << ",";
			else
				file << std::setprecision(20) << A[j * nr_rows_A + i] << "\n";
		}
	}
	file.close();
}

void cuOMT_simple::tic() {
	tictoc_stack.push(clock());
}

double cuOMT_simple::toc() {
	/*std::cout << "Time elapsed: "
	<< ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
	<< std::endl;*/
	auto t = ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	tictoc_stack.pop();
	return t;
}

template <typename T>
void cuOMT_simple::_get_to_csv(const char* output, T* d_vec, int row, int col)
{
	// compute centroid
	T* h_vec = 0;
	h_vec = (T *)malloc(row * col * sizeof(h_vec[0]));
	cublasGetVector(row * col, sizeof(h_vec[0]), d_vec, 1, h_vec, 1);
	print_matrix_csv(h_vec, row, col, output);

	// free junk
	free(h_vec);
}

template <typename T>
void cuOMT_simple::_set_from_csv(const char* input, T* d_vec, int row, int col)
{

	thrust::host_vector<T> h_vec(row * col);
	std::ifstream file(input);
	int count = 0;
	std::string line, value;
	while (file.good())
	{
		getline(file, line);
		std::stringstream          lineStream(line);
		while (std::getline(lineStream, value, ','))
		{
			T val;
			if (typeid(T) == typeid(float))
				val = std::stof(value);
			else if (typeid(T) == typeid(int))
				val = std::stoi(value);
			h_vec[count] = val;
			++count;
		}
	}
	file.close();
	auto d_vec_ptr = thrust::device_pointer_cast(d_vec);
	thrust::copy(h_vec.begin(), h_vec.end(), d_vec_ptr);



	h_vec.clear();


}

template <typename T>
void cuOMT_simple::_set_random_parameter(T* d_para, const int row, const int col)
{
	GPU_generate_RNM(d_para, row, col);
}

void cuOMT_simple::run_simple_omt(int argc, char* argv[])
{
	gd_init(argc, argv);
	gd_pre_calc();
	tic();

	//output

	while (iter < maxIter && !gd_calc_measure() && gd_update_h())
	{
		if (!no_output)
		{
			std::string output_h = std::string("h/") + std::to_string(iter) + std::string(".csv");
            write_h(output_h.c_str());

            std::string output_mu = std::string("pushed_mu/") + std::to_string(iter) + std::string(".csv");
            write_pushed_mu(output_mu.c_str());


            std::string output_volP = std::string("volP/") + std::to_string(iter) + std::string(".csv");
            write_volP(output_volP.c_str());

            std::string output_ind = std::string("ind/") + std::to_string(iter) + std::string(".csv");
            write_ind(output_ind.c_str());
		}

	}
	std::cout << "cuOMT loop takes " << toc() << "s" << std::endl;
	write_h("./h/h.csv");
	write_volP("./volP/volP_final.csv");
	write_ind("./ind/ind_final.csv");
	gd_clean();
}

cuOMT_simple::cuOMT_simple(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, bool _no_output, bool _quiet_mode)
	:dim(_dim), numP(_numP), voln(_voln), maxIter(_maxIter), lr(_lr), eps(_eps), no_output(_no_output), quiet_mode(_quiet_mode)
{

}


