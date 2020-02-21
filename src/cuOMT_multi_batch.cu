#include "cuOMT_multi_batch.cuh"
#include "math_constants.h"
#include <string>
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

int cuOMT_multi_batch::gd_mul_bat_init(int argc, char* argv[])
{
    gd_bat_init(argc, argv);

    //allocate memory
    d_t_g.resize(numTP);
    d_t_g_sum.resize(numTP);
    d_t_h.resize(numTP);
    d_t_delta_h.resize(numTP);
    d_t_A.resize(numTP);
    d_t_ind.resize(voln);
    d_t_ind_val.resize(voln);

    if (h_TP.max_size() < numTP * dim)
    {
        std::cerr << "Error: size of host vector (TP)" << numTP * dim << "exceeds the length of maximum possible size" << h_TP.max_size() << std::endl;
        return EXIT_FAILURE;
    }
    h_TP.resize(numTP * dim);

#ifdef USE_FANCY_GD
    d_t_adam_m = 0;
    d_t_adam_v = 0;
    if (cudaMalloc((void **)&d_t_adam_m, numTP * sizeof(d_t_adam_m[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_adam_m)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_t_adam_v, numTP * sizeof(d_t_adam_v[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_adam_v)\n");
        return EXIT_FAILURE;
    }

    d_t_adam_m_ptr = thrust::device_pointer_cast(d_t_adam_m);
    d_t_adam_v_ptr = thrust::device_pointer_cast(d_t_adam_v);
    
#endif

    bool set_TP(false), set_Pool_dir(false), set_TA(false), set_Th(false), set_t_adam_v(false), set_t_adam_m(false);
    //get list of random number pool files
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-TP") == 0)
        {
            std::ifstream file(argv[i + 1]);
            int count = 0;
            std::string line, value;
            while (file.good())
            {
                getline(file, line);
                std::stringstream          lineStream(line);
                while (std::getline(lineStream, value, ','))
                {
                    float val;
                    val = std::stof(value);
                    h_TP[count] = val;
                    ++count;
                }
            }
            file.close();
            set_TP = true;
        }
        if (strcmp(argv[i], "-Pool_dir") == 0)
        {
            std::string path = argv[i + 1];
             //TODO: add folder traverse method 

            set_Pool_dir = true;
        }
        if (strcmp(argv[i], "-TA") == 0)
        {
            _set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_t_A[0]), 1, numTP);
            set_TA = true;

            std::cout << "Read target measure successfully." << std::endl;
        }
        if (strcmp(argv[i], "-Th") == 0)
        {
            _set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_t_h[0]), 1, numTP);
            set_Th = true;

            std::cout << "Read Th successfully." << std::endl;
        }
#ifdef USE_FANCY_GD
        if (strcmp(argv[i], "-t_adam_v") == 0)
        {
            _set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_t_adam_v[0]), 1, numTP);
            set_t_adam_v = true;

            std::cout << "Read t_adam_v successfully." << std::endl;
        }
        if (strcmp(argv[i], "-t_adam_m") == 0)
        {
            _set_from_csv<float>(argv[i + 1], thrust::raw_pointer_cast(&d_t_adam_m[0]), 1, numTP);
            set_t_adam_m = true;

            std::cout << "Read t_adam_m successfully." << std::endl;
        }
#endif
    }
    if (!set_TP)
    {
        std::cout << "Warning: host vector (TP) is not set during initialization" << std::endl;
        thrust::fill(h_TP.begin(), h_TP.end(), 0.1f);
    }
    if (!set_Pool_dir)
        std::cout << "Warning: random pool directory root is not set during initialization" << std::endl;
    if (!set_TA)
        thrust::fill(d_t_A.begin(), d_t_A.end(), 1.0f / numTP);
    if (!set_Th)
        thrust::fill(d_t_h.begin(), d_t_h.end(), 0.0f);

#ifdef USE_FANCY_GD
    if (!set_t_adam_m)
        thrust::fill(d_t_adam_m_ptr, d_t_adam_m_ptr + numTP, 0.0f);
    if (!set_t_adam_v)
        thrust::fill(d_t_adam_v_ptr, d_t_adam_v_ptr + numTP, 0.0f);
#endif // USE_FANCY_GD


    return 0;
}

int cuOMT_multi_batch::gd_bat_calc_measure()
{
    //iterate over P batch
    thrust::fill(d_t_ind_val.begin(), d_t_ind_val.end(), -1e30); // assume PX+H is not smaller than -1e9


    thrust::device_ptr<float> d_P_ptr = thrust::device_pointer_cast(d_P);

    for (int p_iter = 0; p_iter < numBatP; ++p_iter)
    {
        // duplicate h, i.e repmat(h, [voln 1])
        thrust::gather(d_numP_voln_rep_ptr, d_numP_voln_rep_ptr + numP * voln, d_t_h.begin() + p_iter*numP, d_U_ptr);

        // prepare P: copy corresponding host P to device P        
        thrust::copy(h_TP.begin() + p_iter * numP*dim, h_TP.begin() + (p_iter + 1) * numP*dim, d_P_ptr);

        // PX+H
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numP, voln, dim, alpha, d_P, numP, d_volP, voln, alpha, d_U, numP);
        if (status!= CUBLAS_STATUS_SUCCESS)
        {
            std::cout << _cudaGetErrorEnum(status) <<std::endl;
            fprintf(stderr, "!!!! Device matrix multiplication error (U <- PX + H)\n");
            return EXIT_FAILURE;
        }
        
        // find max parallelly
        thrust::reduce_by_key(thrust::device, d_sampleId.begin(), d_sampleId.begin() + numP * voln, thrust::make_zip_iterator(thrust::make_tuple(d_cellId.begin(), d_U_ptr)), d_voln_key.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(d_ind.begin(), d_ind_val.begin())), thrust::equal_to<int>(), find_max());

        // cast d_ind to global cell index
        thrust::transform(d_ind.begin(), d_ind.end(), d_ind.begin(), axpb<int>(1, p_iter*numP));

        // compare current px+h to history, update max value
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_t_ind.begin(), d_t_ind_val.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_t_ind.end(), d_t_ind_val.end())),
            thrust::make_zip_iterator(thrust::make_tuple(d_ind.begin(), d_ind_val.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_t_ind.begin(), d_t_ind_val.begin())),
            find_max());

    }
    // calculate histogram
    dense_histogram(d_t_ind, d_t_g);

    return 0;
}

int cuOMT_multi_batch::gd_bat_update_h()
{
    // subtract current cell measures with target ones
    thrust::transform(d_t_g.begin(), d_t_g.end(), d_t_A.begin(), d_t_g.begin(), axmy<float>(1.0f / voln));

    /*update h from g*/
#ifdef USE_FANCY_GD
    thrust::transform(thrust::device, d_t_adam_m_ptr, d_t_adam_m_ptr + numTP, d_t_g.begin(), d_t_adam_m_ptr, update_adam_m<float>(0.9f));
    thrust::transform(thrust::device, d_t_adam_v_ptr, d_t_adam_v_ptr + numTP, d_t_g.begin(), d_t_adam_v_ptr, update_adam_v<float>(0.999f));
    thrust::transform(thrust::device, d_t_adam_m_ptr, d_t_adam_m_ptr + numTP, d_t_adam_v_ptr, d_t_delta_h.begin(), adam_delta_h(lr));
    thrust::transform(thrust::device, d_t_h.begin(), d_t_h.begin() + numTP, d_t_delta_h.begin(), d_t_h.begin(), thrust::plus<float>());
#else
    thrust::transform(thrust::device, d_t_g.begin(), d_t_g.end(), d_t_delta_h.begin(), axpb<float>(-lr, 0.0f));
    thrust::transform(thrust::device, d_t_h.begin(), d_t_h.begin() + numP, d_t_delta_h.begin(), d_t_h.begin(), thrust::plus<float>());
#endif

    
    /*normalize h*/
    thrust::transform(d_t_h.begin(), d_t_h.end(), d_t_h.begin(), 
        axpb<float>(1.0f, thrust::transform_reduce(d_t_h.begin(), d_t_h.end(), axpb<float>(-1.0f / (float)numTP, 0.0f), 0.0f, thrust::plus<float>())));


    /*terminate condition*/
    d_g_norm[0] = sqrt(thrust::transform_reduce(d_t_g.begin(), d_t_g.end(), square<float>(), 0.0f, thrust::plus<float>()));
    //if (iter % 100 == 0)
    if (!quiet_mode)
        std::cout << "[" << iter << "/" << maxIter << "] g norm: " << d_g_norm[0] << "/" << eps << std::endl;
    if (d_g_norm[0] < d_eps[0])
        return 0;
    else ++iter;


    return 1;

}


void cuOMT_multi_batch::run_cuOMT_mul_bat_gd(int argc, char* argv[])
{
    tic();
    gd_mul_bat_init(argc, argv);

    d_t_g_sum.resize(numTP);

    double best_g_norm = 1e10;
    double curr_best_g_norm = 1e10;
    int not_converge = 1;
    int steps = 0;
    int dyn_numBat = numBat;
    int count_bad_iter = 0;

    // record results
    const char* output = (std::string("g_log.csv")).c_str();
    std::ofstream file;
    file.open(output);
    while (not_converge && steps <= maxIter)
    {
        thrust::fill(d_t_g_sum.begin(), d_t_g_sum.end(), 0.0f);
        for (int count = 0; count < dyn_numBat; ++count)
        {
            // 2.generate volP online
            gd_bat_pre_calc(count);
            // 3.calculate measure
            gd_bat_calc_measure();
            // 4.repeate 2-3 and aggregate measures
            thrust::transform(d_t_g_sum.begin(), d_t_g_sum.end(), d_t_g.begin(), d_t_g_sum.begin(), thrust::plus<float>());
        }
        thrust::transform(d_t_g_sum.begin(), d_t_g_sum.end(), d_t_g.begin(), axpb<float>(1.0f / (float)dyn_numBat, 0.0f));
        // 5.update h
        not_converge = gd_bat_update_h();
        // 6.repeat 2-5 until converge
        file << d_g_norm[0] << ",";

        // record best norm
        // if (d_g_norm[0] < best_g_norm && !no_output)
        if (!no_output)
        {
            std::string output_h = std::string("h/") + std::to_string(steps) + std::string(".csv");
            _get_to_csv(output_h.c_str(), thrust::raw_pointer_cast(&d_t_h[0]), 1, numTP);

            std::string output_mu = std::string("pushed_mu/") + std::to_string(steps) + std::string(".csv");
            thrust::transform(d_t_g_sum.begin(), d_t_g_sum.end(), d_t_g_sum.begin(), axpb<float>(1.0f / (voln*dyn_numBat), 0));
            _get_to_csv(output_mu.c_str(), thrust::raw_pointer_cast(&d_g_sum[0]), 1, numTP);

            std::string output_volP = std::string("volP/") + std::to_string(steps) + std::string(".csv");
            write_volP(output_volP.c_str());

            std::string output_ind = std::string("ind/") + std::to_string(steps) + std::string(".csv");
            write_ind(output_ind.c_str());

#ifdef USE_FANCY_GD
            std::string output_adam_m = std::string("adam_m/") + std::to_string(steps) + std::string(".csv");
            _get_to_csv(output_adam_m.c_str(), d_t_adam_m, 1, numTP);
            std::string output_adam_v = std::string("adam_v/") + std::to_string(steps) + std::string(".csv");
            _get_to_csv(output_adam_v.c_str(), d_t_adam_v, 1, numTP);
#endif
            best_g_norm = d_g_norm[0];
        }

        // dynamically change amount of MC samples
        if (d_g_norm[0] < curr_best_g_norm)
        {
            curr_best_g_norm = d_g_norm[0];
            count_bad_iter = 0;
        }
        else
            count_bad_iter++;

        if (count_bad_iter > 20)
        {
            dyn_numBat *= 2;
            std::cout << "(MC samples amounts increased to " << std::to_string(dyn_numBat * voln) << "...)" << std::endl;
            //reset parameters
            count_bad_iter = 0;
            curr_best_g_norm = 1e10;

        }
        steps++;
    }
    file.close();

    write_h("./h/h_final.csv");
    write_volP("./volP/volP_final.csv");
    write_ind("./ind/ind_final.csv");

    std::cout << "MC-OMT computation takes " << toc() << "s..." << std::endl;

    // shut down
    h_TP.clear();
    d_t_g.clear();
    d_t_h.clear();
    d_g_sum.clear();
    gd_clean();
}

