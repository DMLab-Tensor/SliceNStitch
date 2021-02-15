#pragma once

#include "loader.hpp"
#include <Eigen/Dense>
#include <any>
#include <chrono>
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

class Config;
class SpTensor_List;
class SpTensor_Hash;
class SpTensor_dX;
class DataStream;

class TensorStream;
TensorStream* generateTensorStream(DataStream& paperX, const Config& config); // Generate the tensor stream

class TensorStream {
public:
    TensorStream(DataStream& paperX, const Config& config);
    virtual ~TensorStream(void);

    void updateTensor(const DataStream::Event& e);
    void updateFactor(void);

    double elapsedTime(void) const; // sec

    double find_reconst(const std::vector<int>& coord) const;
    double density(void) const;

    /* Load matrix */
    void saveFactor(std::string fileName) const;

    /* Get errors */
    double rmse(void) const;
    double fitness(void) const;
    double fitness_latest(void) const;
    double error(const std::vector<int>& coord) const; // Error of the given entry

    void updateAtA(void); // Update the AtA when _use_AtA is false

protected:
    const Config* _config;

    std::vector<int> _compute_order;

    virtual void _updateAlgorithm(void) {} // It will change the current updateAlgorithm later

    double _norm_frobenius_reconst(void) const;
    double _innerprod_X_X_reconst(void) const;

    SpTensor_Hash* _X = nullptr;
    SpTensor_dX* _dX = nullptr;
    DataStream* _paperX;

    Eigen::ArrayXd _lambda;
    std::vector<Eigen::MatrixXd> _A;
    std::vector<Eigen::ArrayXXd> _AtA;

    bool _use_AtA = true;

    long long _nextTime;

    std::chrono::nanoseconds _elapsed_time; // Elapsed time

    void _rand_init_A(void); // Randomly initialize factor matrices

    void _als_base(void); // Base code for ALS
    void _unnormalize_A(void); // Unnormalize the factor matrices

    /* Basic factor update algorithms */
    void _als(void);
    void _recurrent_als(void);
};

/* General ALS until the convergence */
class TensorStream_ALS : public TensorStream {
public:
    TensorStream_ALS(DataStream& paperX, const Config& config)
        : TensorStream(paperX, config)
    {
    }
    virtual ~TensorStream_ALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override
    {
        _als();
    }
};

/* ALS using previous factor matrices as a initialization point until the convergence */
class TensorStream_RecurrentALS : public TensorStream {
public:
    TensorStream_RecurrentALS(DataStream& paperX, const Config& config)
        : TensorStream(paperX, config)
    {
    }
    virtual ~TensorStream_RecurrentALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override
    {
        _recurrent_als();
    }
};

/* ALS using previous factor matrices as a initialization point with fixed # of iterations */
class TensorStream_RecurrentALS_iter : public TensorStream_RecurrentALS {
public:
    TensorStream_RecurrentALS_iter(DataStream& paperX, const Config& config);
    virtual ~TensorStream_RecurrentALS_iter(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
    int _numIter;
};

/* SelectiveALS */
class TensorStream_SelectiveALS : public TensorStream {
public:
    TensorStream_SelectiveALS(DataStream& paperX, const Config& config)
        : TensorStream(paperX, config)
    {
    }
    virtual ~TensorStream_SelectiveALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
};

class TensorStream_coordSelectiveALS : public TensorStream {
public:
    TensorStream_coordSelectiveALS(DataStream& paperX, const Config& config);
    virtual ~TensorStream_coordSelectiveALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
    Eigen::ArrayXd _sqSumProd; // Product of sqSum
    std::vector<Eigen::MatrixXd> _Aprev;
    std::vector<Eigen::ArrayXXd> _AprevtA;
    double _higherBound;
};

/* SamplingALS */
class TensorStream_SamplingALS : public TensorStream {
public:
    TensorStream_SamplingALS(DataStream& paperX, const Config& config);
    virtual ~TensorStream_SamplingALS(void) {}

    /*
        Save sampled entries and their reconstructed value.
        deltidxLists: save changed index, numSel: number of changed idx in each mode.
    */
    void Sampling(const std::vector<std::vector<int>>& deltaIdxLists, const std::vector<int>& numSel);

    typedef std::unordered_map<std::vector<int>, double> samples;
    typedef std::vector<std::vector<samples>> totalSamples;

protected:
    int _numSample; // # of sampling per selected slice
    std::vector<std::vector<int>> _numCand; // Array for sampling candidates
    std::vector<Eigen::ArrayXXd> _AprevtA; // For A_{t-1}^TA_t
    totalSamples _sampledIdx; // Save sampled entries
};

class TensorStream_baseSamplingALS : public TensorStream_SamplingALS {
public:
    TensorStream_baseSamplingALS(DataStream& paperX, const Config& config)
        : TensorStream_SamplingALS(paperX, config)
    {
    }
    virtual ~TensorStream_baseSamplingALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override;

    /* Compute the mttkrp result of one entry */
    void mkpRow(const std::vector<int>& rawIdx, int currMode,
        Eigen::MatrixXd& mkp, int rdx, bool changed, double reconVal = 0.0);
};

/* SamplingALS with coordinate descent */
class TensorStream_coordSamplingALS : public TensorStream_SamplingALS {
public:
    TensorStream_coordSamplingALS(DataStream& paperX, const Config& config);
    virtual ~TensorStream_coordSamplingALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
    Eigen::ArrayXd _sqSumProd; // Product of sqSum
    std::vector<Eigen::MatrixXd> _Aprev;
    double _higherBound;
};

class TensorStream_hybridALS : public TensorStream_baseSamplingALS {
public:
    TensorStream_hybridALS(DataStream& paperX, const Config& config);
    virtual ~TensorStream_hybridALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
    std::vector<std::vector<int>> selIdxLists;
    std::vector<std::vector<int>> samIdxLists;
};

class TensorStream_coordHybridALS : public TensorStream_coordSamplingALS {
public:
    TensorStream_coordHybridALS(DataStream& paperX, const Config& config);
    virtual ~TensorStream_coordHybridALS(void) {}
    double prodAtA(int numMode, int currMode, const std::vector<Eigen::ArrayXXd>& squareVec,
        int currRow, int currCol);
    double prodA(int numMode, int currMode,
        const std::vector<Eigen::MatrixXd>& factorVec, const std::vector<int>& currIdx, int currCol);
    void updateA_and_AtA(double newVal, int rank, int currMode, int currRow, int currCol);

protected:
    virtual void _updateAlgorithm(void) override;
    std::vector<std::vector<int>> selIdxLists;
    std::vector<std::vector<int>> samIdxLists;
};
