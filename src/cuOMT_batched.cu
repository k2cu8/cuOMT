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
	if(h_rn_pool.size() == 0)
		GPU_generate_RNM(d_volP, voln, dim, -50.0f, 50.0f);
	else
	{
		// or fill volP with RAM stored numbers
		load_RN_from_pool(d_volP, count, voln, dim);
	}

	return 0;
}

void cuOMT_batched::run_bat_gd(int argc, char* argv[])
{
	// 1.initiliaze
	gd_bat_init(argc, argv);
	d_g_sum.resize(numP);
	

	int not_converge = 1;
	int steps = 0;

	// record results
	const char* output = (std::string("data/test_g/g_bat.csv")).c_str();
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

		if (steps < 300 || steps % 300 == 0)
		{
			std::string output_mu = std::string("data/skeleton/pushed_mu/") + std::to_string(steps) + std::string(".csv");
			write_pushed_mu(output_mu.c_str());
		}

		steps++;
	}
	file.close();

	write_h("data/skeleton/h.csv");
	write_generated_P("data/skeleton/generated_P.csv");

	// shut down
	d_g_sum.clear();
	gd_clean();
}

