#include "cuOMT.cuh"

int cuOMT::GPU_generate_RNM(float* P, const int nRowP, const int nColP, unsigned int seed)
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, seed);

	// Fill the array with random numbers on the device
	curandStatus_t info = curandGenerateUniform(prng, P, nRowP * nColP);
	if (info != CURAND_STATUS_SUCCESS)
		std::cerr << "Failed generating random matrix of dim " << nRowP << ", " << nColP << std::endl;
	return info;

}

int cuOMT::GPU_generate_RNM(float* P, const int nRowP, const int nColP)
{

	thrust::device_ptr<float> P_ptr(P);
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(nRowP * nColP),
		P_ptr,
		GenRandFloat());
	return 0;


}


int cuOMT::gd_init(int argc, char* argv[])
{
	/*set problem parameters*/
	d_eps.resize(1);
	d_eps[0] = eps;

	/*allocate device vectors*/
	d_P = 0;
	d_volP = 0;
	d_volP_data = 0;
	d_cache = 0;
	d_U = 0;
	d_PX = 0;
	d_Ind = 0;
	d_pos = 0;
	d_X = 0;
	d_pos_norm = 0;
	d_pos_Norm = 0;
	d_pos_NormT = 0;
	d_pos_stride = 0;
	d_pos_stride_data = 0;
	d_voln_dim_rep = 0;
	d_numP_dim_rep = 0;
	d_numP_numP_rep = 0;
	d_nb = 0;
	d_knn_range = 0;
	d_cellId_numP = 0;
	d_nb_vol = 0;
	d_nb_vol_key = 0;

	d_U_tilt = 0;
	d_PX_tilt = 0;


	printf("cuOMT running...\n");

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

	if (cudaMalloc((void **)&d_volP_data, voln * dim * sizeof(d_volP_data[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (allocate volP_data)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_cache, numP * sizeof(d_cache[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (allocate cache)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_U, numP * voln * sizeof(d_U[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (U)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_PX, numP * voln * sizeof(d_PX[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (PX)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_Ind, voln * dim * sizeof(d_Ind[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (Ind)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_pos, numP * dim * sizeof(d_pos[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (pos)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_X, numP * dim * sizeof(d_X[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (X)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_pos_norm, numP * sizeof(d_pos_norm[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (pos norm)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_pos_Norm, numP * numP * sizeof(d_pos_Norm[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (pos Norm)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_pos_NormT, numP * numP * sizeof(d_pos_NormT[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (pos NormT)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_pos_stride_data, numP * dim * sizeof(d_pos_stride_data[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (pos stride data)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_nb, numP * numP * sizeof(d_nb[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (nb)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_knn_range, numP * numP * sizeof(d_knn_range[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (knn_range)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_voln_dim_rep, voln * dim * sizeof(d_voln_dim_rep[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (voln_dim_rep)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_nb_vol, voln * nnb * sizeof(d_nb_vol[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (nb_vol)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_nb_vol_key, voln * nnb * sizeof(d_nb_vol_key[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (nb_vol_key)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_U_tilt, nnb * voln * sizeof(d_U_tilt[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (U_tilt)\n");
		return EXIT_FAILURE;
	}

	if (cudaMalloc((void **)&d_PX_tilt, nnb * voln * sizeof(d_PX_tilt[0])) != cudaSuccess)
	{
		fprintf(stderr, "!!!! device memory allocation error (PX_tilt)\n");
		return EXIT_FAILURE;
	}

	/*fill in data*/

	// set parameters from command line arguments: P, A
	d_P_ptr = thrust::device_pointer_cast(d_P);
	d_A.resize(numP);
	bool set_P(false), set_A(false);
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-P") == 0)
		{
			_set_from_csv<float>(argv[i + 1], d_P, numP, dim);
			set_P = true;
		}
		if (strcmp(argv[i], "-A") == 0)
		{
			_set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_A[0]), numP, 1);
			set_A = true;
		}
	}
	if (!set_P)
	{
		_set_random_parameter(d_P, numP, dim);
		thrust::transform(d_P_ptr, d_P_ptr + numP * dim, d_P_ptr, axpb<float>(1, -0.5));
	}
	if (!set_A)
	{
		thrust::fill(d_A.begin(), d_A.end(), 1.0f / numP);
	}

	// h
	d_h.resize(numP);
	thrust::fill(d_h.begin(), d_h.end(), 0);	

	// cache	
	d_cache_ptr = thrust::device_pointer_cast(d_cache);
	thrust::fill(d_cache_ptr, d_cache_ptr + numP, 0);

	// ind
	d_ind.resize(voln);
	thrust::fill(d_ind.begin(), d_ind.end(), -1);
	d_ind_val.resize(voln);
	thrust::fill(d_ind_val.begin(), d_ind_val.end(), 0.0f);

	// g
	d_g.resize(numP);


	// neighbor
	d_nb_ptr = thrust::device_pointer_cast(d_nb);

	// neighbor for each volP
	d_nb_vol_ptr = thrust::device_pointer_cast(d_nb_vol);
	d_nb_vol_key_ptr = thrust::device_pointer_cast(d_nb_vol_key);

	// position
	d_pos_ptr = thrust::device_pointer_cast(d_pos);
	thrust::copy(d_P_ptr, d_P_ptr + numP * dim, d_pos_ptr);

	/*---gradient descent---*/
	iter = 0;
	/*h_g_norm.resize(maxIter);
	h_time.resize(maxIter);*/
	d_g_norm.resize(1);

	d_U_ptr = thrust::device_pointer_cast(d_U);
	d_PX_ptr = thrust::device_pointer_cast(d_PX);
	d_delta_h.resize(numP);

	d_U_tilt_ptr = thrust::device_pointer_cast(d_U_tilt);
	d_PX_tilt_ptr = thrust::device_pointer_cast(d_PX_tilt);


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

	// sample_id_nnb
	d_sampleId_nnb.resize(nnb * voln);
	thrust::host_vector<int> h_sampleId_nnb(nnb * voln);
	for (int i = 0; i < nnb; ++i)
		for (int j = 0; j < voln; ++j)
		{
			h_sampleId_nnb[i + j * nnb] = j;
		}
	thrust::copy(h_sampleId_nnb.begin(), h_sampleId_nnb.end(), d_sampleId_nnb.begin());
	h_sampleId_nnb.clear();

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


	// cellId numP
	/*thrust::device_ptr<int> d_cellId_numP_ptr(d_cellId_numP);
	thrust::host_vector<int> h_cellId_numP(numP * numP);
	for (int i = 0; i < numP; ++i)
	{
	for (int j = 0; j < numP; ++j)
	{
	h_cellId_numP[i + numP * j] = i;
	}
	}
	thrust::copy(h_cellId_numP.begin(), h_cellId_numP.end(), d_cellId_numP_ptr);
	h_cellId_numP.clear();*/
	d_cellId_numP = thrust::raw_pointer_cast(&d_cellId[0]);

	// dummy key for reduction in calculating knn
	d_knn_dummy_key.resize(numP);

	// duplicate of histogram by dimension
	d_Hist.resize(numP * dim);

	// pos key 
	d_pos_key.resize(numP * dim);

	// pos value
	d_pos_val.resize(numP*dim);

	// pos stride
	/*thrust::device_ptr<int> d_pos_stride_ptr(d_pos_stride);
	thrust::host_vector<int> h_pos_stride(numP * dim);
	for (int i = 0; i < numP; ++i)
	{
	for (int j = 0; j < dim; ++j)
	{
	h_pos_stride[i + j*numP] = i;
	}
	}
	thrust::copy(h_pos_stride.begin(), h_pos_stride.end(), d_pos_stride_ptr);
	h_pos_stride.clear();*/
	d_pos_stride = thrust::raw_pointer_cast(&d_cellId[0]);

	// knn range
	/*thrust::device_ptr<int> d_knn_range_ptr(d_knn_range);
	thrust::host_vector<int> h_knn_range(numP * numP);
	for (int i = 0; i < numP; ++i)
	{
	for (int j = 0; j < numP; ++j)
	{
	h_knn_range[i + numP * j] = j;
	}
	}
	thrust::copy(h_knn_range.begin(), h_knn_range.end(), d_knn_range_ptr);
	h_knn_range.clear();*/
	d_knn_range = thrust::raw_pointer_cast(&d_sampleId[0]);

	// voln * dim stride used to replicate vector dim times
	d_voln_dim_rep_ptr = thrust::device_pointer_cast(d_voln_dim_rep);
	thrust::host_vector<int> h_voln_dim_rep(voln * dim);
	for (int i = 0; i < voln; ++i)
		for (int j = 0; j < dim; ++j)
			h_voln_dim_rep[i + j*voln] = i;
	thrust::copy(h_voln_dim_rep.begin(), h_voln_dim_rep.end(), d_voln_dim_rep_ptr);
	h_voln_dim_rep.clear();

	// voln * numP stride used to replicate vector dim times
	/*thrust::device_ptr<int> d_numP_voln_rep_ptr(d_numP_voln_rep);
	thrust::host_vector<int> h_numP_voln_rep(numP * voln);
	for (int i = 0; i < numP; ++i)
	for (int j = 0; j < voln; ++j)
	h_numP_voln_rep[i + j*numP] = i;
	thrust::copy(h_numP_voln_rep.begin(), h_numP_voln_rep.end(), d_numP_voln_rep_ptr);
	h_numP_voln_rep.clear();*/
	d_numP_voln_rep_ptr = thrust::device_pointer_cast(thrust::raw_pointer_cast(&d_cellId[0]));

	// numP * dim stride used to replicate vector dim times
	/*thrust::device_ptr<int> d_numP_dim_rep_ptr(d_numP_dim_rep);
	thrust::host_vector<int> h_numP_dim_rep(numP * dim);
	for (int i = 0; i < numP; ++i)
	for (int j = 0; j < dim; ++j)
	h_numP_dim_rep[i + j*numP] = i;
	thrust::copy(h_numP_dim_rep.begin(), h_numP_dim_rep.end(), d_numP_dim_rep_ptr);
	h_numP_dim_rep.clear();*/
	d_numP_dim_rep = thrust::raw_pointer_cast(&d_cellId[0]);

	// numP * numP stride
	/*thrust::device_ptr<int> d_numP_numP_rep_ptr(d_numP_numP_rep);
	thrust::host_vector<int> h_numP_numP_rep(numP * numP);
	for (int i = 0; i < numP; ++i)
	for (int j = 0; j < numP; ++j)
	h_numP_numP_rep[i + j*numP] = i;
	thrust::copy(h_numP_numP_rep.begin(), h_numP_numP_rep.end(), d_numP_numP_rep_ptr);
	h_numP_numP_rep.clear();*/
	d_numP_numP_rep = thrust::raw_pointer_cast(&d_cellId[0]);

	// nnb * voln replication
	/*thrust::device_vector<int> d_nnb_voln_rep(nnb * voln);
	thrust::host_vector<int> h_nnb_voln_rep(nnb * voln);
	for (int i = 0; i < nnb; ++i)
	{
	for (int j = 0; j < voln; ++j)
	h_nnb_voln_rep[i + nnb*j] = j;
	}
	thrust::copy(h_nnb_voln_rep.begin(), h_nnb_voln_rep.end(), d_nnb_voln_rep.begin());
	h_nnb_voln_rep.clear();*/
	d_nnb_voln_rep_ptr = thrust::device_pointer_cast(thrust::raw_pointer_cast(&d_sampleId_nnb[0]));

	// volP
	d_volP_ptr = thrust::device_pointer_cast(d_volP);

	

	// set initial h to be -0.5*|Pi|^2
	thrust::device_vector<int> d_numP_dim_stride(numP * dim);
	thrust::copy(d_cellId.begin(), d_cellId.begin() + numP * dim, d_numP_dim_stride.begin());
	thrust::device_vector<float> d_P_data(numP * dim);
	thrust::copy(d_P_ptr, d_P_ptr + numP * dim, d_P_data.begin());

	thrust::transform(d_P_data.begin(), d_P_data.end(), d_P_data.begin(), square<float>());
	thrust::sort_by_key(d_numP_dim_stride.begin(), d_numP_dim_stride.end(), d_P_data.begin());
	thrust::reduce_by_key(d_numP_dim_stride.begin(), d_numP_dim_stride.end(), d_P_data.begin(), d_knn_dummy_key.begin(), d_h.begin());

	thrust::transform(d_h.begin(), d_h.begin() + numP, d_h.begin(), axpb<float>(-0.5, 0));

	return 0;
}

int cuOMT::gd_pre_calc()
{
	tic();

	// fill volP with random numbers
	GPU_generate_RNM(d_volP, voln, dim);
	thrust::transform(d_volP_ptr, d_volP_ptr + voln * dim, d_volP_ptr, axpb<float>(1, -0.5));

	// calculate PX
	if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numP, voln, dim, alpha, d_P, numP, d_volP, voln, zero, d_PX, numP) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! Device matrix multiplication error (U <- PX)\n");
		return EXIT_FAILURE;
	}

	std::cout << "Pre-computing takes " << toc() << "s..." << std::endl;
	return 0;
}

int cuOMT::gd_loop()
{
	// begin loop

	if (true)
	//if ((iter < 200 && iter % 50 == 0) || (iter >= 200 && iter < 500 && iter % 80 == 0) || (iter >= 500 && iter % 200 == 0))
	{
		// duplicate h, i.e repmat(h, [voln 1])
		thrust::gather(d_numP_voln_rep_ptr, d_numP_voln_rep_ptr + numP*voln, d_h.begin(), d_U_ptr);

		// PX+H
		thrust::transform(d_U_ptr, d_U_ptr + numP * voln, d_PX_ptr, d_U_ptr, thrust::plus<float>());


		// find max parallelly
		thrust::reduce_by_key(thrust::device, d_sampleId.begin(), d_sampleId.begin() + numP * voln, thrust::make_zip_iterator(thrust::make_tuple(d_cellId.begin(), d_U_ptr)), d_voln_key.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(d_ind.begin(), d_ind_val.begin())), thrust::equal_to<int>(), find_max());

		// calculate histogram
		dense_histogram(d_ind, d_g);

		//// calculate pos every 100 steps
		////calculate the cell center
		//compute_pos(handle, d_volP, d_volP_data, voln, thrust::raw_pointer_cast(&d_ind[0]), dim, d_Ind, d_voln_dim_rep, numP, alpha, d_numP_dim_rep,
		//	thrust::raw_pointer_cast(&d_pos_key[0]),
		//	thrust::raw_pointer_cast(&d_pos_val[0]), thrust::raw_pointer_cast(&d_g[0]), thrust::raw_pointer_cast(&d_Hist[0]),
		//	d_pos);

		//// calculate neighbor
		//compute_knn(handle, d_pos, d_X, numP, dim, nnb, d_pos_norm, d_pos_Norm, d_pos_NormT, d_pos_stride,
		//	d_pos_stride_data, thrust::raw_pointer_cast(&d_knn_dummy_key[0]), d_numP_numP_rep, alpha, beta,
		//	neg_2, zero, d_knn_range, d_cellId_numP, voln, thrust::raw_pointer_cast(&d_ind[0]), d_nb);
	}
	else
	{
		// distribute neighbor info to each volP
		thrust::gather(d_nnb_voln_rep_ptr, d_nnb_voln_rep_ptr + voln*nnb, d_ind.begin(), d_nb_vol_key_ptr);
		thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nnb * voln), d_nb_vol_key_ptr, d_nb_vol_key_ptr, xmodnpby(nnb, numP));
		thrust::gather(d_nb_vol_key_ptr, d_nb_vol_key_ptr + nnb*voln, d_nb_ptr, d_nb_vol_ptr);

		// H_tilt <- h
		thrust::gather(d_nb_vol_ptr, d_nb_vol_ptr + nnb * voln, d_h.begin(), d_U_tilt_ptr);

		// key for PX_tilt
		thrust::transform(d_nnb_voln_rep_ptr, d_nnb_voln_rep_ptr + nnb*voln, d_nb_vol_ptr, d_nb_vol_key_ptr, axpbypc<int>(numP, 1, 0));

		// PX_tilt <- PX
		thrust::gather(d_nb_vol_key_ptr, d_nb_vol_key_ptr + nnb * voln, d_PX_ptr, d_PX_tilt_ptr);

		// PX_tilt + H_tilt
		thrust::transform(d_U_tilt_ptr, d_U_tilt_ptr + nnb * voln, d_PX_tilt_ptr, d_U_tilt_ptr, thrust::plus<float>());

		// find max
		thrust::reduce_by_key(d_sampleId_nnb.begin(), d_sampleId_nnb.begin() + nnb * voln, thrust::make_zip_iterator(thrust::make_tuple(d_nb_vol_ptr, d_U_tilt_ptr)), d_voln_key.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(d_ind.begin(), d_ind_val.begin())), thrust::equal_to<int>(), find_max());

		// calculate histogram
		dense_histogram(d_ind, d_g);
	}



	// subtract current cell measures with target ones
	thrust::transform(d_g.begin(), d_g.end(), d_A.begin(), d_g.begin(), axmy<float>(1.0f / voln));

	/*update h from g*/
	thrust::transform(thrust::device, d_cache_ptr, d_cache_ptr + numP, d_g.begin(), d_cache_ptr, update_cache<float>());
	thrust::transform(thrust::device, d_g.begin(), d_g.end(), d_cache_ptr, d_delta_h.begin(), delta_h(lr));
	thrust::transform(thrust::device, d_h.begin(), d_h.begin() + numP, d_delta_h.begin(), d_h.begin(), thrust::plus<float>());

	/*normalize h*/
	thrust::transform(d_h.begin(), d_h.begin() + numP, d_h.begin(), axpb<float>(1, -thrust::reduce(d_h.begin(), d_h.begin() + numP, 0.0f, mean(numP))));

	/*output result*/
	if ((iter < 600))
		_get_to_csv((std::string("data/test_exp/mu/mu_") + std::to_string(iter)).c_str(), thrust::raw_pointer_cast(&d_g[0]), numP, 1);

	/*terminate condition*/
	//h_time[iter] = toc();
	//h_g_norm[iter] = sqrt(thrust::transform_reduce(d_g.begin(), d_g.end(), square<float>(), 0.0f, thrust::plus<float>()));
	//std::cout << "[" << iter << "/" << maxIter << "] g norm: " << h_g_norm[iter] << "/" << eps << " (" << h_time[iter] << "s elapsed)..." << std::endl;

	d_g_norm[0] = sqrt(thrust::transform_reduce(d_g.begin(), d_g.end(), square<float>(), 0.0f, thrust::plus<float>()));
	std::cout << "[" << iter << "/" << maxIter << "] g norm: " << d_g_norm[0] << "/" << eps << std::endl;
	if (d_g_norm[0] < d_eps[0])
		return 0;
	else ++iter;


	return 1;
}

int cuOMT::gd_clean(void)
{
	/* Memory clean up */

	if (cudaFree(d_U) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (U)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_PX) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (U)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_volP) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (volP)\n");
		return EXIT_FAILURE;
	}


	if (cudaFree(d_cache) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (cache)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_Ind) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (Ind)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_pos) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (pos)\n");
		return EXIT_FAILURE;
	}


	if (cudaFree(d_pos_norm) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (pos_norm)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_pos_Norm) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (pos_Norm)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_pos_NormT) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (pos_NormT)\n");
		return EXIT_FAILURE;
	}

	if (cudaFree(d_pos_stride_data) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (d_pos_stride_data)\n");
		return EXIT_FAILURE;
	}




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
void cuOMT::dense_histogram(Vector1& input, Vector2& histogram)
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
void cuOMT::print_matrix(T* A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";

		}
		std::cout << std::endl;

	}
	std::cout << std::endl;
}

template <typename T>
void cuOMT::print_matrix_csv(T* A, int nr_rows_A, int nr_cols_A, const char* output) {
	std::ofstream file;
	file.open(output);
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			if (j != nr_cols_A - 1)
				file << A[j * nr_rows_A + i] << ",";
			else
				file << A[j * nr_rows_A + i] << "\n";
		}
	}
}

void cuOMT::tic() {
	tictoc_stack.push(clock());
}

double cuOMT::toc() {
	/*std::cout << "Time elapsed: "
	<< ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
	<< std::endl;*/
	auto t = ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	tictoc_stack.pop();
	return t;
}

int cuOMT::compute_pos(cublasHandle_t &handle, float* d_volP, float* d_volP_data, int voln, int* d_ind, int dim, int* d_Ind, int* d_voln_dim_rep, int numP,
	const float* alf, int* d_numP_dim_rep, int* d_pos_key, float* d_pos_val, float* d_hist, float* d_Hist, float* d_pos)
{
	thrust::device_ptr<int> d_ind_ptr(d_ind);
	thrust::device_ptr<int> d_Ind_ptr(d_Ind);
	thrust::device_ptr<float> d_volP_ptr(d_volP);
	thrust::device_ptr<float> d_volP_data_ptr(d_volP_data);
	thrust::device_ptr<float> d_pos_ptr(d_pos);
	thrust::device_ptr<float> d_hist_ptr(d_hist);
	thrust::device_ptr<float> d_Hist_ptr(d_Hist);
	thrust::device_ptr<int> d_pos_key_ptr(d_pos_key);
	thrust::device_ptr<float> d_pos_val_ptr(d_pos_val);
	thrust::device_ptr<int> d_voln_dim_rep_ptr(d_voln_dim_rep);
	thrust::device_ptr<int> d_numP_dim_rep_ptr(d_numP_dim_rep);

	// reset looping variables
	thrust::fill(d_pos_key_ptr, d_pos_key_ptr + numP * dim, -1);
	thrust::fill(d_pos_val_ptr, d_pos_val_ptr + numP * dim, 0.0f);

	// copy volP to volP_data
	thrust::copy(d_volP_ptr, d_volP_ptr + voln * dim, d_volP_data_ptr);

	//set Ind to be replicate of ind
	thrust::gather(d_voln_dim_rep_ptr, d_voln_dim_rep_ptr + voln * dim, d_ind_ptr, d_Ind_ptr);

	// add constant to different dimensions
	thrust::transform(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(voln * dim), d_Ind_ptr, d_Ind_ptr, add_dim_step(voln, numP));

	// sort d_volP by d_Ind
	thrust::stable_sort_by_key(d_Ind_ptr, d_Ind_ptr + voln * dim, d_volP_data_ptr);

	// summing up sample coordinates within the same cell
	thrust::reduce_by_key(d_Ind_ptr, d_Ind_ptr + voln * dim, d_volP_data_ptr, d_pos_key_ptr, d_pos_val_ptr);

	// scatter coordinate sum back to each cell
	thrust::scatter_if(d_pos_val_ptr, d_pos_val_ptr + numP * dim, d_pos_key_ptr, d_pos_key_ptr, d_pos_ptr, nonneg_scatter<int>());

	// build Hist by duplicate histogram dim times
	thrust::gather(d_numP_dim_rep_ptr, d_numP_dim_rep_ptr + numP * dim, d_hist_ptr, d_Hist_ptr);

	// devide by counts of each cell
	thrust::transform(d_pos_ptr, d_pos_ptr + numP * dim, d_Hist_ptr, d_pos_ptr, safe_divide());


	/*float* h_pos = 0;
	h_pos = (float *)malloc(numP * dim * sizeof(h_pos[0]));
	cublasGetVector(numP * dim, sizeof(h_pos[0]), d_pos, 1, h_pos, 1);
	print_matrix_csv(h_pos, numP, dim, "pos.csv");*/

	return 0;
}

int cuOMT::compute_knn(cublasHandle_t &handle, float* d_pos, float *d_X, int numP, int dim, const int k, float* d_pos_norm, float* d_pos_Norm, float* d_pos_NormT,
	int* d_pos_stride, int* d_pos_stride_data, int* d_knn_dummy_key, int* d_numP_numP_rep, const float* alpha, const float* beta, const float* neg_2, const float* zero,
	int* d_knn_range, int* d_cellId_numP, const int voln, int* d_ind, int *d_nb)
{
	thrust::device_ptr<float> d_pos_ptr(d_pos);
	thrust::device_ptr<float> d_X_ptr(d_X);
	thrust::device_ptr<float> d_pos_norm_ptr(d_pos_norm);
	thrust::device_ptr<float> d_pos_Norm_ptr(d_pos_Norm);
	thrust::device_ptr<float> d_pos_NormT_ptr(d_pos_NormT);
	thrust::device_ptr<int> d_pos_stride_ptr(d_pos_stride);
	thrust::device_ptr<int> d_pos_stride_data_ptr(d_pos_stride_data);
	thrust::device_ptr<int> d_knn_dummy_key_ptr(d_knn_dummy_key);
	thrust::device_ptr<int> d_knn_range_ptr(d_knn_range);
	thrust::device_ptr<int> d_cellId_numP_ptr(d_cellId_numP);
	thrust::device_ptr<int> d_nb_ptr(d_nb);
	thrust::device_ptr<int> d_ind_ptr(d_ind);
	thrust::device_ptr<int> d_numP_numP_rep_ptr(d_numP_numP_rep);


	// copy vectors that will be sorted for the use in next iteration
	thrust::copy(d_pos_stride_ptr, d_pos_stride_ptr + numP * dim, d_pos_stride_data_ptr);
	thrust::copy(d_pos_ptr, d_pos_ptr + numP * dim, d_X_ptr);
	thrust::copy(d_cellId_numP_ptr, d_cellId_numP_ptr + numP * numP, d_nb_ptr);

	// zip iterators
	auto zip_iter_first = thrust::make_zip_iterator(thrust::make_tuple(d_nb_ptr, d_pos_Norm_ptr));

	// compute distance matrix d(X,Y) = |X|^2 + |Y|^2 - 2XY^T
	thrust::sort_by_key(d_pos_stride_data_ptr, d_pos_stride_data_ptr + numP * dim, d_X_ptr);
	thrust::transform(d_X_ptr, d_X_ptr + numP * dim, d_X_ptr, square<float>());
	thrust::reduce_by_key(d_pos_stride_data_ptr, d_pos_stride_data_ptr + numP * dim, d_X_ptr, d_knn_dummy_key_ptr, d_pos_norm_ptr);

	thrust::gather(d_numP_numP_rep_ptr, d_numP_numP_rep_ptr + numP*numP, d_pos_norm_ptr, d_pos_Norm_ptr);


	/*float* h_pos_Norm = 0;
	h_pos_Norm = (float *)malloc(numP * numP * sizeof(h_pos_Norm[0]));
	std::cout << "\n d_pos_Norm: " << std::endl;
	cublasGetVector(numP*numP, sizeof(h_pos_Norm[0]), d_pos_Norm, 1, h_pos_Norm, 1);
	print_matrix_csv(h_pos_Norm, numP, numP, "pos_Norm0.csv");*/

	if (cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, numP, numP, alpha, d_pos_Norm, numP, zero, d_pos_Norm, numP, d_pos_NormT, numP)
		!= CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! cuBLAS geam error (pos_Norm transpose <- pos_Norm)\n");
		return EXIT_FAILURE;
	}


	if (cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, numP, numP, alpha, d_pos_Norm, numP, beta, d_pos_NormT, numP, d_pos_Norm, numP)
		!= CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! cuBLAS geam error (pos_Norm + pos_Norm^T)\n");
		return EXIT_FAILURE;
	}

	if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numP, numP, dim, neg_2, d_pos, numP, d_pos, numP, beta, d_pos_Norm, numP) != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! cuBLAS gemm error (pos_Norm -2*dX*dX^T)\n");
		return EXIT_FAILURE;
	}

	// vector sorting
	thrust::stable_sort_by_key(zip_iter_first, zip_iter_first + numP * numP, d_knn_range_ptr, tuple_smaller());
	thrust::stable_sort_by_key(d_knn_range_ptr, d_knn_range_ptr + numP * numP, zip_iter_first);

	// get back id's
	//thrust::copy(zip_iter_first, zip_iter_last, thrust::make_zip_iterator(thrust::make_tuple(d_nb_ptr, d_pos_Norm_ptr)));

	return 0;
}

template <typename T>
void cuOMT::_get_to_csv(const char* output, T* d_vec, int row, int col)
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
void cuOMT::_set_from_csv(const char* input, T* d_vec, int row, int col)
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
	cudaDeviceSynchronize();
}

template <typename T>
void cuOMT::_set_random_parameter(T* d_para, const int row, const int col)
{
	GPU_generate_RNM(d_para, row, col);
}

cuOMT::cuOMT(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, const int _nnb)
	:dim(_dim), numP(_numP), voln(_voln), maxIter(_maxIter), eps(_eps), lr(_lr), nnb(_nnb)
{
}


