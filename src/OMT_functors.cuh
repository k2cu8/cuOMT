#include <thrust/functional.h>
#include <thrust/random.h>

//thrust functors
template <typename T>
struct square
{
	__host__ __device__
		T operator()(const T& x) const
	{
		return x * x;
	}
};

struct int_mod
{
	const int m;
	int_mod (const int _m) : m(_m) {}
	__host__ __device__
		int operator()(const int & x) const
	{
		return x % m;
	}
};

template <typename T>
struct axpb
{
	const T a;
	const T b;
	axpb(const T _a, const T _b) : a(_a), b(_b) {}
	__host__ __device__
		T operator () (const T &x) const
	{
		return a*x + b;
	}
};

template <typename T>
struct axmy
{
	const T a;
	axmy(const T _a) : a(_a) {}
	__host__ __device__
		T operator () (const T &x, const T &y) const
	{
		return a*x - y;
	}
};

template <typename T>
struct axpbypc
{
	const T a;
	const T b;
	const T c;
	axpbypc(const T _a, const T _b, const T _c) : a(_a), b(_b), c(_c) {}
	__host__ __device__
		T operator () (const T &x, const T &y) const
	{
		//printf("%f\n", a*x + b*y + c);
		return a*x + b*y + c;
	}
};

struct delta_h
{
	const float lr;
	delta_h(const float _lr) : lr(_lr) {}
	__host__ __device__
		float operator () (const float &g, const float &cache) const
	{
        if (cache != 0)
            return -lr * g / sqrt(cache);
        else
            return 0;
	}
};

struct mean
{
	const int numP;
	mean(const int _numP) : numP(_numP) {}
	__host__ __device__
		float operator() (const float &x, const float &y) const
	{
		return (x + y) / numP;
	}
};

template <typename T>
struct update_cache
{
	__host__ __device__
		T operator()(const T &cache, const T &g) const
	{
		return cache + g*g;
	}

};

template <typename T>
struct update_adam_m
{
    const float beta1;
    update_adam_m(const float _beta1) : beta1(_beta1) {}
    __host__ __device__
        T operator()(const T &m, const T &g) const
    {
        return beta1*m + (1-beta1) * g;
    }

};

template <typename T>
struct update_adam_v
{
    const float beta2;
    update_adam_v(const float _beta2) : beta2(_beta2) {}
    __host__ __device__
        T operator()(const T &v, const T &g) const
    {
        return beta2 * v + (1 - beta2) * g * g;
    }

};

struct adam_delta_h
{
    const float lr;
    adam_delta_h(const float _lr) : lr(_lr) {}
    __host__ __device__
        float operator () (const float &m, const float &v) const
    {
        return -lr * m / (sqrt(v) + 1e-8);
    }
};

struct find_max
{
	__host__ __device__
		thrust::tuple<int, float> operator()(const thrust::tuple<int, float> &a, const thrust::tuple<int, float> &b) const
	{
		if (thrust::get<1>(a) > thrust::get<1>(b))
			return a;
		else
			return b;
	}
};

struct tuple_smaller
{
	__host__ __device__
		bool operator()(const thrust::tuple<int, float> &a, const thrust::tuple<int, float> &b) const
	{
		if (thrust::get<1>(a) < thrust::get<1>(b))
			return true;
		else
			return false;
	}
};

struct tuple_greater
{
	__host__ __device__
		bool operator()(const thrust::tuple<int, float> &a, const thrust::tuple<int, float> &b) const
	{
		if (thrust::get<1>(a) > thrust::get<1>(b))
			return true;
		else
			return false;
	}
};

struct tuple_get_0
{
	__host__ __device__
		int operator()(thrust::tuple<int, float> &a) const
	{
		return thrust::get<0>(a);
	}
};

struct safe_divide
{
	__host__ __device__
		float operator()(float &x, float &y)
	{
		return (y == 0) ? x : x / y;
	}
};

template <typename T1, typename T2>
struct type_transfer
{
	__host__ __device__
		T2 operator()(T1 &x)
	{
		return (T2)x;
	}
};

template <typename T>
struct nonneg_scatter
{
	__host__ __device__
		bool operator()(T &x)
	{
		return x < 0 ? false : true;
	}
};

template <typename T>
struct truncate
{
	const int t;
	const int stride;
	truncate(const int _t, const int _stride) : t(_t), stride(_stride)  {}
	__host__ __device__
		T operator()(const int &id, const T &x) const
	{
		if (id % stride < t)
			return x;
		else
			return (T) -1;
	}
};

struct add_dim_step
{
	const int voln;
	const int numP;
	add_dim_step(const int _voln, const int _numP) : voln(_voln), numP(_numP) {}
	__host__ __device__
		int operator()(const int &n, const int &x) const
	{
		return x + (n / voln) * numP;
	}
};

struct xmodnpby
{
	const int b;
	const int n;
	xmodnpby(const int _n, const int _b) : n(_n), b(_b) {}
	__host__ __device__
		int operator()(const int &x, const int &y) const
	{
		return (x%n) + b*y;
	}
};



struct GenRand
{
	const int MAX;
	GenRand(const int _max) : MAX(_max) {}
	__device__
		float operator () (int idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_int_distribution<int> uniDist(0, MAX-1);
		randEng.discard(idx);
		return uniDist(randEng);
	}
};


struct GenRandFloat
{
	__device__
		float operator () (int idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_real_distribution<float> uniDist(0, 1);
		randEng.discard(idx);
		return uniDist(randEng);
	}
};

struct GenRandFloatUni
{
	const float a, b;
	GenRandFloatUni(const float _a, const float _b) : a(_a), b(_b) {}
	__device__
		float operator () (int idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_real_distribution<float> uniDist(a, b);
		randEng.discard(idx);
		return uniDist(randEng);
	}
};