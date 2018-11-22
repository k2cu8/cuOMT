#include "cuOMT_batched.cuh"

void cuOMT_batched::load_RN_from_pool(float* P, int count, const int num, const int dim)
{
	thrust::device_ptr<float> P_ptr(P);
	thrust::copy(h_rn_pool.begin() + num*count*dim, h_rn_pool.begin() + num*count*dim + num*dim, P_ptr);
}

int cuOMT_batched::gd_bat_init(int argc, char* argv[])
{
	gd_init(argc, argv);

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-Pool") == 0)
		{
			h_rn_pool.resize(voln*numBat*dim);

			std::ifstream file(argv[i+1]);
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

					h_rn_pool[count] = val;
					++count;
				}
			}
			file.close();

			std::cout << "h_pool: " << std::endl;
			thrust::copy(h_rn_pool.begin(), h_rn_pool.begin() + 20, std::ostream_iterator<float>(std::cout, " "));
			std::cout << std::endl;

		}
	}
	return 0;
}

int cuOMT_batched::gd_bat_pre_calc(int count)
{
	// fill volP with random numbers
    if (h_rn_pool.size() == 0)
    {
        //GPU_generate_RNM(d_volP, voln, dim, -.5f, .5f);
        curand_RNG_sobol(d_volP, voln, dim, count*voln);
    }
	else
	{
		// or fill volP with RAM stored numbers
		load_RN_from_pool(d_volP, count, voln, dim);
	}

	return 0;
}

void cuOMT_batched::write_dyn_pushed_mu(const char* output, int nb)
{
    // calculate histogram
    thrust::transform(d_g_sum.begin(), d_g_sum.end(), d_g_sum.begin(), axpb<float>(1.0f / (voln*nb), 0));

    _get_to_csv(output, thrust::raw_pointer_cast(&d_g_sum[0]), 1, numP);
}


void cuOMT_batched::run_bat_gd(int argc, char* argv[])
{
	// 1.initiliaze
	gd_bat_init(argc, argv);
	d_g_sum.resize(numP);
	
    double best_g_norm = 1e10;
	int not_converge = 1;
	int steps = 0;

	// record results
	const char* output = (std::string("test_g/g_bat.csv")).c_str();
	std::ofstream file;
	file.open(output);
	while (not_converge && steps <= maxIter)
	{
		thrust::fill(d_g_sum.begin(), d_g_sum.end(), 0.0f);
		for (int count = 0; count < numBat; ++count)
		{
			// 2.generate volP online
			gd_bat_pre_calc(count);
			// 3.calculate measure
			gd_calc_measure();
			// 4.repeate 2-3 and aggregate measures
			thrust::transform(d_g_sum.begin(), d_g_sum.end(), d_g.begin(), d_g_sum.begin(), thrust::plus<float>());
		}
		thrust::transform(d_g_sum.begin(), d_g_sum.end(), d_g.begin(), axpb<float>(1 / (float)numBat, 0));
		// 5.update h
		not_converge = gd_update_h();
		// 6.repeat 2-5 until converge
		file << d_g_norm[0] << ",";

		if (d_g_norm[0] < best_g_norm)
		{
			std::string output_mu = std::string("pushed_mu/") + std::to_string(steps) + std::string(".csv");
			write_pushed_mu(output_mu.c_str());

            std::string output_h = std::string("h/") + std::to_string(steps) + std::string(".csv");
            write_h(output_h.c_str());

            best_g_norm = d_g_norm[0];
		}

		steps++;
	}
	file.close();

	write_h("h_final.csv");
	write_generated_P("generated_P.csv");

	// shut down
	d_g_sum.clear();
	gd_clean();
}

void cuOMT_batched::run_dyn_bat_gd(int argc, char* argv[])
{
    tic();
    // 1.initiliaze
    gd_bat_init(argc, argv);
    d_g_sum.resize(numP);

    double best_g_norm = 1e10;
    double curr_best_g_norm = 1e10;
    int not_converge = 1;
    int steps = 0;
    int dyn_numBat = numBat;
    int count_bad_iter = 0;

    // record results
    const char* output = (std::string("test_g/g_bat.csv")).c_str();
    std::ofstream file;
    file.open(output);
    while (not_converge && steps <= maxIter)
    {
        thrust::fill(d_g_sum.begin(), d_g_sum.end(), 0.0f);
        for (int count = 0; count < dyn_numBat; ++count)
        {
            // 2.generate volP online
            gd_bat_pre_calc(count);
            // 3.calculate measure
            gd_calc_measure();
            // 4.repeate 2-3 and aggregate measures
            thrust::transform(d_g_sum.begin(), d_g_sum.end(), d_g.begin(), d_g_sum.begin(), thrust::plus<float>());
        }
        thrust::transform(d_g_sum.begin(), d_g_sum.end(), d_g.begin(), axpb<float>(1 / (float)dyn_numBat, 0));
        // 5.update h
        not_converge = gd_update_h();
        // 6.repeat 2-5 until converge
        file << d_g_norm[0] << ",";

        // record best norm
        if (d_g_norm[0] < best_g_norm)
        //if (true)
        {
            std::string output_h = std::string("h/") + std::to_string(steps) + std::string(".csv");
            write_h(output_h.c_str());

            std::string output_mu = std::string("pushed_mu/") + std::to_string(steps) + std::string(".csv");
            write_dyn_pushed_mu(output_mu.c_str(),dyn_numBat);
#ifdef USE_FANCY_GD
            std::string output_adam_m = std::string("adam_m/") + std::to_string(steps) + std::string(".csv");
            write_adam_m(output_adam_m.c_str());
            std::string output_adam_v = std::string("adam_v/") + std::to_string(steps) + std::string(".csv");
            write_adam_v(output_adam_v.c_str());
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

    write_h("h_final.csv");
    write_generated_P("generated_P.csv");

    std::cout << "MC-OMT computation takes " << toc() << "s..." << std::endl;

    // shut down
    d_g_sum.clear();
    gd_clean();
}
