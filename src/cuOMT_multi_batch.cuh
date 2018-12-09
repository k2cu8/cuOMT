#include "cuOMT_batched.cuh"

class cuOMT_multi_batch :public cuOMT_batched
{
protected:
    //number of total P
    const int numTP;

    int numBatP;
    //host vectors
    thrust::host_vector<float> h_TP; //total P stored in host memory 

    //device vectors
    thrust::device_vector<float> d_t_h; //total h
    thrust::device_vector<float> d_t_delta_h; //total delat h
    thrust::device_vector<float> d_t_g; //total g
    thrust::device_vector<float> d_t_g_sum; //total g sum
    thrust::device_vector<int> d_t_ind; //total ind
    thrust::device_vector<float> d_t_ind_val; //total ind value 
    thrust::device_vector<float> d_t_A; //target measure
#ifdef USE_FANCY_GD
    float *d_t_adam_m;
    float *d_t_adam_v;
    thrust::device_ptr<float> d_t_adam_m_ptr;
    thrust::device_ptr<float> d_t_adam_v_ptr;
#endif



    //random pool dirs
    std::vector<std::string> m_rn_pool_files;




protected:
    /*multi-batch init*/
    int gd_mul_bat_init(int argc, char* argv[]);

    //calculate measure per batch
    int gd_bat_calc_measure();

    int gd_bat_update_h();

public:
    cuOMT_multi_batch(const int _dim, const int _numP, const int _voln, const int _maxIter, const float _eps, const float _lr, const int _numBat, const int _numTP) :
        cuOMT_batched(_dim,_numP,_voln,_maxIter,_eps,_lr,_numBat), numTP(_numTP)
    {
        if (numTP % numP != 0)
            std::cerr << "Error: batch size " << numP << " is not a factor of " << numTP << std::endl;
        else
            numBatP = numTP / numP;
    };
 
    ~cuOMT_multi_batch() {};

    void run_cuOMT_mul_bat_gd(int argc, char* argv[]);
};