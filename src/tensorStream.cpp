#include "tensorStream.hpp"
#include "constants.hpp"
#include "loader.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

TensorStream* generateTensorStream(DataStream& paperX, const Config& config)
{
    TensorStream* ts;
    if (config.algo() == "ALS") {
        ts = new TensorStream_ALS(paperX, config);
    } else if (config.algo() == "SNS_MAT") {
        ts = new TensorStream_RecurrentALS_iter(paperX, config);
    } else if (config.algo() == "SNS_VEC") {
        ts = new TensorStream_SelectiveALS(paperX, config);
    } else if (config.algo() == "SNS_VEC+") {
        ts = new TensorStream_coordSelectiveALS(paperX, config);
    } else if (config.algo() == "SNS_RND") {
        ts = new TensorStream_hybridALS(paperX, config);
    } else if (config.algo() == "SNS_RND+") {
        ts = new TensorStream_coordHybridALS(paperX, config);
    } else {
        ts = new TensorStream(paperX, config);
    }
    return ts;
}

TensorStream::TensorStream(DataStream& paperX, const Config& config)
{
    _paperX = &paperX;
    _config = &config;

    const int numMode = _config->numMode();
    const int unitNum = _config->unitNum();
    const int unitSize = _config->unitSize();
    const int rank = _config->rank();

    std::vector<int> dimension = _config->nonTempDim(); // Dimension of the tensor X
    dimension.push_back(unitNum);

    // Set the computing order
    {
        _compute_order.reserve(numMode);
        _compute_order.push_back(numMode - 1);
        for (int m = 0; m < numMode - 1; ++m) {
            _compute_order.push_back(m);
        }
    }

    _elapsed_time = std::chrono::nanoseconds::zero();

    /* Initialize X */
    {
        _X = new SpTensor_Hash(dimension);

        std::vector<int> coord(numMode);

        while (true) {
            auto e = _paperX->pop();
            if (e == nullptr) {
                break;
            }

            for (int m = 0; m < numMode - 1; ++m) {
                coord[m] = e->nonTempCoord[m];
            }
            coord[numMode - 1] = e->newUnitIdx;
            _X->insert(coord, e->val);
        }
    }

    /* Initialize dX */
    _dX = new SpTensor_dX(dimension);

    /* Initialize lambda, A, AtA */
    {
        _A.resize(numMode);
        _AtA.resize(numMode);

        _als();
        _unnormalize_A(); // Unnormalization
    }
}

TensorStream::~TensorStream(void)
{
    delete _dX;
    delete _X;
}

void TensorStream::updateTensor(const DataStream::Event& e)
{
    const int unitNum = _config->unitNum();
    const int unitSize = _config->unitSize();
    const int numMode = _config->numMode();

    // Clear dX
    _dX->clear();

    // Eventwise
    std::vector<int> coord(numMode);
    for (int m = 0; m < numMode - 1; ++m) {
        coord[m] = e.nonTempCoord[m];
    }
    coord[numMode - 1] = e.newUnitIdx;

    if (e.newUnitIdx != -1) {
        _dX->insert(coord, e.val);
        _X->insert(coord, e.val);
    }

    if (e.newUnitIdx != unitNum - 1) {
        coord[numMode - 1] += 1;
        _dX->insert(coord, -e.val);
        _X->insert(coord, -e.val);
    }
}

void TensorStream::saveFactor(std::string fileName) const
{
    const int numMode = _config->numMode();
    // Save factor matrix
    for (int m = 0; m < numMode; ++m) {
        std::ofstream outFile("factor_matrix/mode_" + std::to_string(m) + "_" + fileName);
        outFile << std::setprecision(std::numeric_limits<double>::digits10 + 2) << _A[m];
        outFile.close();
    }
}

void TensorStream::updateFactor(void)
{
    int _updatePeriod = _config->updatePeriod();
    if (_updatePeriod == 1) {
        const unsigned long long numDelta = _dX->numNnz();
        assert(numDelta >= 0);
        if (numDelta == 0)
            return;
    }

    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    _updateAlgorithm();
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    _elapsed_time += end - begin;

    /* Normalization
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    std::vector<Eigen::ArrayXd> lambdas(numMode);

    // Update lambda (column-wise L_infinity-norm)
    for (int m = 0; m < numMode; ++m) {
        lambdas[m] = _A[m].cwiseAbs().colwise().maxCoeff().array();
        for (int r = 0; r < rank; ++r) { // If lambda is 0, then set to 1
            if (abs(lambdas[m][r]) < TENSOR_MACHINE_EPSILON) {
                lambdas[m][r] = 1;
            }
        }

        // Normalize A
        _A[m] = (_A[m].array().rowwise() / lambdas[m].transpose()).matrix();
    }

    for (int r = 0; r < rank; ++r) {
        _lambda[r] = 1.0;
        for (int m = 0; m < numMode; ++m) {
            _lambda[r] *= lambdas[m][r];
        }
    }

    // Unnormalize A
    _unnormalize_A(); */
}

void TensorStream::updateAtA(void)
{
    if (!_use_AtA) {
        const int numMode = _config->numMode();

        for (int m = 0; m < numMode; m++) {
            _AtA[m] = (_A[m].transpose() * _A[m]).array();
        }
    }
}

double TensorStream::elapsedTime(void) const
{
    return std::chrono::duration<double>(_elapsed_time).count();
}

double TensorStream::density() const
{
    const std::vector<int>& dimension = _X->dimension();
    double _density = (double)(_X->numNnz());
    const int& _numMode = _config->numMode();

    for (int m = 0; m < _numMode; m++)
        _density /= (double)dimension[m];

    return _density;
}

double TensorStream::find_reconst(const std::vector<int>& coord) const
{
    const int rank = _config->rank();
    const int numMode = _config->numMode();

    Eigen::ArrayXd prodMatrix = Eigen::ArrayXd::Ones(rank);
    for (int m = 0; m < numMode; ++m) {
        prodMatrix *= _A[m].row(coord[m]).array();
    }
    prodMatrix *= _lambda;

    return prodMatrix.sum();
}

double TensorStream::rmse(void) const
{
    const std::vector<int>& dimension = _X->dimension();
    const double normX = _X->norm_frobenius();
    const double normX_hat = _norm_frobenius_reconst();
    const double innerprod = _innerprod_X_X_reconst();
    const double normresidual_square = abs(pow(normX, 2) + pow(normX_hat, 2) - 2 * innerprod);

    double error_square = normresidual_square;
    for (int const& modeSize : dimension) {
        error_square /= modeSize;
    }
    return sqrt(error_square);
}

double TensorStream::fitness(void) const
{
    const double normX = _X->norm_frobenius();
    const double normX_hat = _norm_frobenius_reconst();
    const double innerprod = _innerprod_X_X_reconst();
    const double normresidual = sqrt(abs(pow(normX, 2) + pow(normX_hat, 2) - 2 * innerprod));

    return 1 - normresidual / normX;
}

double TensorStream::fitness_latest(void) const
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();

    const double normX_latest = _X->norm_frobenius_latest();

    // Compute the latest normX_hat
    double normX_hat_latest_square = 0;
    {
        const Eigen::MatrixXd lambda_mat = _lambda.matrix();

        Eigen::ArrayXXd coefMatrix = (lambda_mat * lambda_mat.transpose()).array();
        for (int m = 0; m < numMode - 1; ++m) {
            coefMatrix *= _AtA[m];
        }
        {
            const auto A_temporal = _A[numMode - 1].row(dimension[numMode - 1] - 1);
            coefMatrix *= (A_temporal.transpose() * A_temporal).array();
        }

        normX_hat_latest_square = abs(coefMatrix.sum());
    }

    // Compute the latest inner product
    double innerprod_latest = 0;
    {
        const SpTensor_Hash::coord_map& coord_map = _X->elems()[numMode - 1][dimension[numMode - 1] - 1];
        for (auto const& it : coord_map) {
            const std::vector<int> coord_vec = it.first;
            const double& val_real = it.second;
            const double val_reconst = find_reconst(coord_vec);
            innerprod_latest += val_real * val_reconst;
        }
    }

    const double normresidual = sqrt(abs(pow(normX_latest, 2) + normX_hat_latest_square - 2 * innerprod_latest));

    return 1 - normresidual / normX_latest;
}

double TensorStream::error(const std::vector<int>& coord) const
{
    const double val = _X->find(coord);
    const double val_reconst = find_reconst(coord);

    //return abs((val - val_reconst) / val);
    return abs(val - val_reconst);
}

double TensorStream::_norm_frobenius_reconst(void) const
{
    const int numMode = _config->numMode();
    const Eigen::MatrixXd lambda_mat = _lambda.matrix();

    Eigen::ArrayXXd coefMatrix = (lambda_mat * lambda_mat.transpose()).array();
    for (int m = 0; m < numMode; ++m) {
        coefMatrix *= _AtA[m];
    }
    return sqrt(abs(coefMatrix.sum()));
}

double TensorStream::_innerprod_X_X_reconst(void) const
{
    double innerprod = 0;

    const std::vector<SpTensor_Hash::coord_map>& elems = _X->elems()[0];
    for (auto const& coord_map : elems) {
        for (auto const& it : coord_map) {
            const std::vector<int> coord_vec = it.first;
            const double& val_real = it.second;
            const double val_reconst = find_reconst(coord_vec);
            innerprod += val_real * val_reconst;
        }
    }

    return innerprod;
}

inline void TensorStream::_rand_init_A(void)
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int unitNum = _config->unitNum();
    const int rank = _config->rank();

    std::mt19937* rng = RNG::Instance()->rng();
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int m = 0; m < numMode - 1; ++m) {
        _A[m] = Eigen::MatrixXd::NullaryExpr(dimension[m], rank, [&]() { return dis(*rng); });
    }
    _A[numMode - 1] = Eigen::MatrixXd::NullaryExpr(unitNum, rank, [&]() { return dis(*rng); });

    for (int m = 0; m < numMode; ++m) {
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }

    _lambda = Eigen::ArrayXd::Ones(rank);
}

inline void TensorStream::_als_base(void)
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<SpTensor_Hash::row_vector>& elems = _X->elems();

    for (int const& m : _compute_order) {
        const int numIdx = dimension[m];

        // Initialize V
        Eigen::MatrixXd V;
        {
            Eigen::ArrayXXd V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];
            }
            V = V_arr.matrix();
        }

        // Compute mttkrp
        Eigen::MatrixXd mkp = Eigen::MatrixXd::Zero(numIdx, rank);
        {
            int idx = 0;
            for (auto const& coord_map : elems[m]) {
                for (auto const& it : coord_map) {
                    const std::vector<int> coord_vec = it.first;
                    const double val = it.second;

                    Eigen::ArrayXd mkp_row = Eigen::ArrayXd::Constant(rank, val);
                    for (int n = 0; n < numMode; ++n) {
                        if (n == m) {
                            continue;
                        }
                        mkp_row *= _A[n].row(coord_vec[n]).array();
                    }
                    mkp.row(idx) += mkp_row.transpose().matrix();
                }
                idx++;
            }
        }

        // Solve mkp / V using householder QR
        _A[m] = V.transpose().householderQr().solve(mkp.transpose()).transpose();

        // Update lambda (column-wise L_infinity-norm)
        _lambda = _A[m].cwiseAbs().colwise().maxCoeff().array();
        for (int r = 0; r < rank; ++r) { // If lambda is 0, then set to 1
            if (abs(_lambda[r]) < TENSOR_MACHINE_EPSILON) {
                _lambda[r] = 1;
            }
        }

        // Normalize A
        _A[m] = (_A[m].array().rowwise() / _lambda.transpose()).matrix();

        // Update AtA
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }
}

inline void TensorStream::_unnormalize_A(void)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    // Multiply Nth root of lambda into each factor matrix
    const Eigen::ArrayXd nrootLambda = _lambda.pow(1.0 / numMode);

    _lambda = Eigen::ArrayXd::Ones(rank);
    for (int m = 0; m < numMode; ++m) {
        _A[m] = (_A[m].array().rowwise() * nrootLambda.transpose()).matrix();
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }
}

void TensorStream::_als(void)
{
    _rand_init_A(); // Randomly initialize A
    _recurrent_als();
}

void TensorStream::_recurrent_als(void)
{
    // Run ALS until fit change is lower than tolerance
    double fitold = 0;
    for (int i = 0; i < ALS_MAX_ITERS; ++i) {
        _als_base();

        const double fitnew = fitness();
        if (i > 0 && abs(fitold - fitnew) < ALS_FIT_CHANGE_TOL) {
            break;
        }
        fitold = fitnew;
    }
}

TensorStream_RecurrentALS_iter::TensorStream_RecurrentALS_iter(DataStream& paperX, const Config& config)
    : TensorStream_RecurrentALS(paperX, config)
{
    _numIter = _config->findAlgoSettings<int>("numIter");
}

void TensorStream_RecurrentALS_iter::_updateAlgorithm(void)
{
    for (int iter = 0; iter < _numIter; ++iter) {
        _als_base();
    }
}

void TensorStream_SelectiveALS::_updateAlgorithm(void)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<SpTensor_Hash::row_vector>& elems = _X->elems();
    const std::vector<std::vector<int>>& nnzIdxLists = _dX->idxLists();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();

    std::vector<Eigen::ArrayXXd> AtA_prev(numMode);
    for (int m = 0; m < numMode; ++m) {
        Eigen::MatrixXd A_prev = _A[m](nnzIdxLists[m], Eigen::all);
        AtA_prev[m] = (A_prev.transpose() * A_prev).array();
    }

    for (int const& m : _compute_order) {
        const int numIdx = nnzIdxLists[m].size();
        if (numIdx == 0) {
            continue;
        }

        // Initialize V
        Eigen::MatrixXd V;
        {
            Eigen::ArrayXXd V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];
            }
            V = V_arr.matrix();
        }

        // Compute mttkrp
        Eigen::MatrixXd mkp = Eigen::MatrixXd::Zero(numIdx, rank);
        {
            int rdx = 0;
            for (int const& idx : nnzIdxLists[m]) {
                const SpTensor_Hash::coord_map& targetIterable = (m == numMode - 1) ? elemsdX[m][idx] : elems[m][idx];
                for (auto const& it : targetIterable) {
                    const std::vector<int> coord_vec = it.first;
                    const double val = it.second;

                    Eigen::ArrayXd mkp_row = Eigen::ArrayXd::Constant(rank, val);
                    for (int n = 0; n < numMode; ++n) {
                        if (n == m) {
                            continue;
                        }
                        mkp_row *= _A[n].row(coord_vec[n]).array();
                    }

                    mkp.row(rdx) += mkp_row.transpose().matrix();
                }
                rdx++;
            }
        }

        // Solve mkp / V using householder QR
        Eigen::MatrixXd A_new = V.transpose().householderQr().solve(mkp.transpose()).transpose();
        if (m == numMode - 1) {
            A_new += _A[m](nnzIdxLists[m], Eigen::all);
        }

        _A[m](nnzIdxLists[m], Eigen::all) = A_new;
        Eigen::ArrayXXd AtA_new = (A_new.transpose() * A_new).array();

        // Update AtA
        _AtA[m] += AtA_new - AtA_prev[m];
        AtA_prev[m] = AtA_new;
    }
}

TensorStream_coordSelectiveALS::TensorStream_coordSelectiveALS(DataStream& paperX, const Config& config)
    : TensorStream(paperX, config)
{
    int rank = _config->rank();
    int numMode = _config->numMode();
    _higherBound = _config->findAlgoSettings<double>("clipping"); // Set the bound of clipping

    // Initialize square sum
    for (int m = 0; m < numMode; ++m) {
        if (m == 0) {
            _sqSumProd = _AtA[m].matrix().diagonal().array();
        } else {
            _sqSumProd *= _AtA[m].matrix().diagonal().array();
        }
    }

    // Initialize Aprev
    _Aprev.reserve(numMode);
    for (int m = 0; m < numMode; ++m)
        _Aprev.push_back(_A[m]);

    // Allocate memory for previous matrix
    _AprevtA.reserve(numMode);
    for (int m = 0; m < numMode; ++m) {
        _AprevtA.push_back(Eigen::ArrayXXd(rank, rank));
    }
}

void TensorStream_coordSelectiveALS::_updateAlgorithm(void)
{
    const std::vector<std::vector<int>>& deltaIdxLists = _dX->idxLists();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();
    const std::vector<SpTensor_Hash::row_vector>& elemsX = _X->elems();

    double firstTerm, secondTerm, thirdTerm;
    double prodaa, prodba, proda;
    double newVal, prevVal, prevNorm;
    int rank = _config->rank();
    int numMode = _config->numMode();
    std::vector<int> numSel(numMode);

    // Copy previous factor matrix (only part that will be changed) and AtAprev
    std::copy(_AtA.begin(), _AtA.end(), _AprevtA.begin());
    for (int m = 0; m < numMode; ++m) {
        _Aprev[m](deltaIdxLists[m], Eigen::all) = _A[m](deltaIdxLists[m], Eigen::all);
    }

    for (int m = 0; m < numMode; ++m) {
        numSel[m] = deltaIdxLists[m].size();
    }

    // Update in column(same rank) unit
    for (int const& m : _compute_order) {
        for (int r = 0; r < rank; ++r) {
            for (int di = 0; di < numSel[m]; ++di) {
                int deltaIdx = deltaIdxLists[m][di];

                // Compute the first and third term in slide 200513.pptx, page 4.
                firstTerm = 0.0;
                secondTerm = 0.0;
                thirdTerm = 0.0;

                // Update third term
                for (int s = 0; s < rank; ++s) {
                    prodaa = 1.0;
                    if (s == r)
                        continue;

                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        prodaa *= _AtA[l](s, r);
                    }
                    thirdTerm += _A[m](deltaIdx, s) * prodaa;
                }

                // Update first term
                if (m == numMode - 1) {
                    for (int s = 0; s < rank; ++s) {
                        prodba = 1.0;

                        // update first term
                        for (int l = 0; l < numMode; ++l) {
                            if (l == m)
                                continue;

                            prodba *= _AprevtA[l](s, r);
                        }
                        firstTerm += _Aprev[m](deltaIdx, s) * prodba;
                    }
                }

                // Get the second term
                const std::unordered_map<std::vector<int>, double>& currMap = (m == numMode - 1) ? elemsdX[m][deltaIdx] : elemsX[m][deltaIdx];
                for (const auto& idxIt : currMap) {
                    proda = 1.0;
                    const std::vector<int>& currIdx = idxIt.first;
                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        proda *= _A[l](currIdx[l], r);
                    }
                    secondTerm += idxIt.second * proda;
                }

                // Update an entry of factor matrix
                prevVal = _A[m](deltaIdx, r);
                newVal = firstTerm + secondTerm - thirdTerm;
                newVal /= _sqSumProd[r];
                newVal *= _AtA[m](r, r);

                // Clipping
                if (abs(newVal) >= _higherBound) {
                    newVal = std::signbit(newVal) ? (-1.0) * _higherBound : _higherBound;
                }

                _A[m](deltaIdx, r) = newVal;

                // Update AtA and AprevtA
                prevNorm = _AtA[m](r, r);
                for (int s = 0; s < rank; ++s) {
                    if (s != r) {
                        double tempDelta = (newVal - prevVal) * _A[m](deltaIdx, s);
                        _AtA[m](r, s) += tempDelta;
                        _AtA[m](s, r) += tempDelta;
                    } else {
                        _AtA[m](s, s) += newVal * newVal - prevVal * prevVal;
                    }

                    _AprevtA[m](s, r) += _Aprev[m](deltaIdx, s) * (newVal - prevVal);
                }

                // Update sqSumProd
                _sqSumProd[r] *= (_AtA[m](r, r) / prevNorm);
            }
        }
    }
}

TensorStream_SamplingALS::TensorStream_SamplingALS(DataStream& paperX, const Config& config)
    : TensorStream(paperX, config)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<int>& dimension = _X->dimension(); // Dimension of the tensor X

    _numSample = _config->findAlgoSettings<int>("numSample"); //Set the number of samples

    // Assign maximum candidates(number of entries) for sampling
    _numCand.reserve(numMode);
    {
        std::vector<int> v(numMode);
        for (int m = 0; m < numMode; ++m) {
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                v[n] = dimension[n];
            }
            _numCand.push_back(v);
        }
    }

    // Allocate memory for previous matrix
    _AprevtA.reserve(numMode);
    for (int m = 0; m < numMode; ++m) {
        _AprevtA.push_back(Eigen::ArrayXXd(rank, rank));
    }

    // Set size for totalSamples
    _sampledIdx.resize(numMode);
}

void TensorStream_SamplingALS::Sampling(const std::vector<std::vector<int>>& deltaIdxLists,
    const std::vector<int>& numSel)
{
    int numMode = _config->numMode();
    int currIdx;

    // Sample random entries in the changed slice
    for (int const& m : _compute_order) {
        _sampledIdx[m].clear();
        _sampledIdx[m].resize(numSel[m]);

        for (int i = 0; i < numSel[m]; ++i) {
            currIdx = deltaIdxLists[m][i];
            pickIdx(_numCand[m], m, currIdx, _numSample, _sampledIdx[m][i]);
            // Save reconstructed value
            for (auto& it : _sampledIdx[m][i]) {
                it.second = find_reconst(it.first);
            }
        }
    }
}

void TensorStream_baseSamplingALS::_updateAlgorithm()
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<int>& dimension = _X->dimension();
    const std::vector<SpTensor_Hash::row_vector>& elemsX = _X->elems();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();
    const std::vector<std::vector<int>>& nnzIdxLists = _dX->idxLists();

    // Aprev.transpose() * A
    std::copy(_AtA.begin(), _AtA.end(), _AprevtA.begin());

    std::vector<int> numNnzIdx(numMode);
    for (int m = 0; m < numMode; ++m) {
        numNnzIdx[m] = nnzIdxLists[m].size();
    }

    // Sample entries
    Sampling(nnzIdxLists, numNnzIdx);

    // Run the update codes
    {
        Eigen::ArrayXXd V_arr(rank, rank);
        Eigen::ArrayXXd Vprime_arr(rank, rank);
        Eigen::MatrixXd V(rank, rank);
        Eigen::MatrixXd Vprime(rank, rank);
        Eigen::ArrayXd mkp_row(rank);
        Eigen::ArrayXXd AtABef(rank, rank);

        // Update factor matrices
        for (int const& m : _compute_order) {
            int numIdx = numNnzIdx[m];
            Eigen::MatrixXd Aprev = _A[m](nnzIdxLists[m], Eigen::all);
            AtABef = (Aprev.transpose() * Aprev).array();

            // Initialize V, Vprime
            V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            Vprime_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];
                Vprime_arr *= _AprevtA[n]; // Can be optimized by utilizing V_arr
            }
            V = V_arr.matrix();
            Vprime = Vprime_arr.matrix();

            // Compute mttkrp of sampledX and A
            Eigen::MatrixXd mkp = Eigen::MatrixXd::Zero(numIdx, rank);
            {
                int rdx = 0;
                for (int i = 0; i < numNnzIdx[m]; ++i) {
                    int fixedIdx = nnzIdxLists[m][i];

                    // Sample random entries in the current slice
                    // Compute mttkrp for corresponding row
                    for (auto const& it : _sampledIdx[m][i]) {
                        if (elemsdX[m][fixedIdx].find(it.first) != elemsdX[m][fixedIdx].end())
                            continue;

                        mkpRow(it.first, m, mkp, rdx, false, it.second);
                    }

                    for (auto const& it : elemsdX[m][fixedIdx]) {
                        mkpRow(it.first, m, mkp, rdx, true);
                    }

                    rdx++;
                }
            }

            // Update A
            Eigen::MatrixXd A_new(numIdx, rank);
            if (m == _compute_order[0]) {
                A_new = _A[m](nnzIdxLists[m], Eigen::all) + V.transpose().householderQr().solve(mkp.transpose()).transpose();
                _A[m](nnzIdxLists[m], Eigen::all) = A_new;
            } else {
                A_new = V.transpose().householderQr().solve((Aprev * Vprime + mkp).transpose()).transpose();
                _A[m](nnzIdxLists[m], Eigen::all) = A_new;
            }

            // Update AtA, AprevtA
            _AtA[m] += (A_new.transpose() * A_new).array() - AtABef;
            _AprevtA[m] += (Aprev.transpose() * A_new).array() - AtABef;
        }
    }
}

void TensorStream_baseSamplingALS::mkpRow(const std::vector<int>& rawIdx, int currMode,
    Eigen::MatrixXd& mkp, int rdx, bool changed, double reconVal)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<int>& dimension = _X->dimension();

    double currVal;
    if (changed)
        currVal = _dX->find(rawIdx);
    else
        currVal = _X->find(rawIdx) - reconVal;

    Eigen::ArrayXd mkp_row = Eigen::ArrayXd::Constant(rank, currVal);
    for (int n = 0; n < numMode; ++n) {
        if (n == currMode) {
            continue;
        }
        mkp_row *= _A[n].row(rawIdx[n]).array();
    }
    mkp.row(rdx) += mkp_row.transpose().matrix();
}

TensorStream_coordSamplingALS::TensorStream_coordSamplingALS(DataStream& paperX, const Config& config)
    : TensorStream_SamplingALS(paperX, config)
{
    int rank = _config->rank();
    int numMode = _config->numMode();
    _higherBound = _config->findAlgoSettings<double>("clipping"); // Set the bound of clipping

    // Initialize square sum
    for (int m = 0; m < numMode; ++m) {
        if (m == 0) {
            _sqSumProd = _AtA[m].matrix().diagonal().array();
        } else {
            _sqSumProd *= _AtA[m].matrix().diagonal().array();
        }
    }

    // Initialize Aprev
    _Aprev.reserve(numMode);
    for (int m = 0; m < numMode; ++m)
        _Aprev.push_back(_A[m]);
}

void TensorStream_coordSamplingALS::_updateAlgorithm(void)
{
    const std::vector<std::vector<int>>& deltaIdxLists = _dX->idxLists();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();

    int rank = _config->rank();
    int numMode = _config->numMode();
    double firstTerm, secondTerm, thirdTerm;
    double prodba, prodaa, proda;
    double prevVal, newVal, prevNorm;

    std::vector<int> numSel(numMode);

    // Copy previous factor matrix (only part that will be changed) and AtAprev
    std::copy(_AtA.begin(), _AtA.end(), _AprevtA.begin());
    for (int m = 0; m < numMode; ++m) {
        _Aprev[m](deltaIdxLists[m], Eigen::all) = _A[m](deltaIdxLists[m], Eigen::all);
    }

    for (int m = 0; m < numMode; ++m) {
        numSel[m] = deltaIdxLists[m].size();
    }

    // Sample random entries in the changed slice
    Sampling(deltaIdxLists, numSel);

    // Update in column(same rank) unit
    for (int r = 0; r < rank; ++r) {
        for (int m = 0; m < numMode; ++m) {
            for (int di = 0; di < numSel[m]; ++di) {
                int deltaIdx = deltaIdxLists[m][di];

                // Compute the first and third term in slide 200513.pptx, page 4.
                firstTerm = 0.0;
                secondTerm = 0.0;
                thirdTerm = 0.0;
                for (int s = 0; s < rank; ++s) {
                    prodba = 1.0;
                    prodaa = 1.0;

                    // update first term
                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        prodba *= _AprevtA[l](s, r);
                    }
                    firstTerm += _Aprev[m](deltaIdx, s) * prodba;

                    // update third term
                    if (s == r)
                        continue;

                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        prodaa *= _AtA[l](s, r);
                    }
                    thirdTerm += _A[m](deltaIdx, s) * prodaa;
                }

                // Get the second term
                // Handle sampled entries
                for (const auto& samIt : _sampledIdx[m][di]) {
                    if (elemsdX[m][deltaIdx].find(samIt.first) != elemsdX[m][deltaIdx].end()) {
                        continue;
                    }

                    proda = 1.0;
                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        proda *= _A[l](samIt.first[l], r);
                    }
                    secondTerm += (_X->find(samIt.first) - samIt.second) * proda;
                }

                // Handle changed entries
                for (const auto& chaIdxIt : elemsdX[m][deltaIdx]) {
                    proda = 1.0;
                    const std::vector<int>& chaIdx = chaIdxIt.first;
                    for (int l = 0; l < numMode; ++l) {
                        if (l == m)
                            continue;

                        proda *= _A[l](chaIdx[l], r);
                    }
                    secondTerm += chaIdxIt.second * proda;
                }

                // Update an entry of factor matrix
                prevVal = _A[m](deltaIdx, r);
                newVal = firstTerm + secondTerm - thirdTerm;
                newVal /= _sqSumProd[r];
                newVal *= _AtA[m](r, r);

                // Clipping
                if (abs(newVal) >= _higherBound) {
                    newVal = std::signbit(newVal) ? (-1.0) * _higherBound : _higherBound;
                }

                _A[m](deltaIdx, r) = newVal;

                // Update AtA and AprevtA
                prevNorm = _AtA[m](r, r);
                for (int s = 0; s < rank; ++s) {
                    if (s != r) {
                        double tempDelta = (newVal - prevVal) * _A[m](deltaIdx, s);
                        _AtA[m](r, s) += tempDelta;
                        _AtA[m](s, r) += tempDelta;
                    } else {
                        _AtA[m](s, s) += newVal * newVal - prevVal * prevVal;
                    }

                    _AprevtA[m](s, r) += _Aprev[m](deltaIdx, s) * (newVal - prevVal);
                }

                // Update sqSumProd
                _sqSumProd[r] *= (_AtA[m](r, r) / prevNorm);
            }
        }
    }
}

TensorStream_hybridALS::TensorStream_hybridALS(DataStream& paperX, const Config& config)
    : TensorStream_baseSamplingALS(paperX, config)
{
    int numMode = _config->numMode();
    selIdxLists.resize(numMode);
    samIdxLists.resize(numMode);
}

void TensorStream_hybridALS::_updateAlgorithm(void)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<int>& dimension = _X->dimension();
    const std::vector<SpTensor_Hash::row_vector>& elemsX = _X->elems();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();
    const std::vector<std::vector<int>>& deltaIdxLists = _dX->idxLists();

    // Figure out the rows to sample and select
    std::vector<int> numSamIdx(numMode);
    for (int m = 0; m < numMode; ++m) {
        selIdxLists[m].clear();
        samIdxLists[m].clear();

        for (const int i : deltaIdxLists[m]) {
            if (elemsX[m][i].size() <= _numSample) {
                selIdxLists[m].push_back(i);
            } else {
                samIdxLists[m].push_back(i);
            }
        }
        numSamIdx[m] = samIdxLists[m].size();
    }

    // Aprev.transpose() * A
    std::copy(_AtA.begin(), _AtA.end(), _AprevtA.begin());

    // Sample random entries in the changed slice
    Sampling(samIdxLists, numSamIdx);

    // Run the update codes
    {
        Eigen::ArrayXXd V_arr(rank, rank);
        Eigen::ArrayXXd Vprime_arr(rank, rank);
        Eigen::MatrixXd V(rank, rank);
        Eigen::MatrixXd Vprime(rank, rank);
        Eigen::ArrayXd mkp_row(rank);
        Eigen::ArrayXXd AtABef(rank, rank);

        // Update factor matrices
        for (int const& m : _compute_order) {
            int numSel = selIdxLists[m].size();
            int numSam = numSamIdx[m];

            // Initialize V, Vprime
            V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            Vprime_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];

                if (numSam > 0) {
                    Vprime_arr *= _AprevtA[n]; // Can be optimized by utilizing V_arr
                }
            }
            V = V_arr.matrix();
            Vprime = Vprime_arr.matrix();

            // Compute mttkrp of selective ALS
            Eigen::MatrixXd mkpSel = Eigen::MatrixXd::Zero(numSel, rank);
            {
                // Handle selective entries
                for (int i = 0; i < numSel; i++) {
                    int fixedIdx = selIdxLists[m][i];
                    for (auto const& it : elemsX[m][fixedIdx]) {
                        mkpRow(it.first, m, mkpSel, i, false, 0.0);
                    }
                }
            }

            // Compute mttkrp of sampling ALS
            Eigen::MatrixXd mkpSam = Eigen::MatrixXd::Zero(numSam, rank);
            {
                for (int i = 0; i < numSam; i++) {
                    int fixedIdx = samIdxLists[m][i];
                    for (auto const& it : _sampledIdx[m][i]) {
                        if (elemsdX[m][fixedIdx].find(it.first) != elemsdX[m][fixedIdx].end())
                            continue;

                        mkpRow(it.first, m, mkpSam, i, false, it.second);
                    }

                    for (auto const& it : elemsdX[m][fixedIdx]) {
                        mkpRow(it.first, m, mkpSam, i, true);
                    }
                }
            }

            // Update A corresponds to selective ALS
            if (numSam > 0) {
                Eigen::MatrixXd Asam_prev = _A[m](samIdxLists[m], Eigen::all);
                AtABef = (Asam_prev.transpose() * Asam_prev).array();
                Eigen::MatrixXd Asam_new(numSam, rank);

                // Rows updated by sampling ALS
                if (m == _compute_order[0]) {
                    Asam_new = _A[m](samIdxLists[m], Eigen::all) + V.transpose().householderQr().solve(mkpSam.transpose()).transpose();
                } else {
                    Asam_new = V.transpose().householderQr().solve((Asam_prev * Vprime + mkpSam).transpose()).transpose();
                }
                _A[m](samIdxLists[m], Eigen::all) = Asam_new;

                // Update AtA, AprevtA
                _AtA[m] += (Asam_new.transpose() * Asam_new).array() - AtABef;
                _AprevtA[m] += (Asam_prev.transpose() * Asam_new).array() - AtABef;
            }

            if (numSel > 0) {
                Eigen::MatrixXd Asel_prev = _A[m](selIdxLists[m], Eigen::all);
                AtABef = (Asel_prev.transpose() * Asel_prev).array();
                Eigen::MatrixXd Asel_new(numSel, rank);

                // Rows updated by selective ALS
                Asel_new = V.transpose().householderQr().solve(mkpSel.transpose()).transpose();
                _A[m](selIdxLists[m], Eigen::all) = Asel_new;

                // Update AtA, AprevtA
                _AtA[m] += (Asel_new.transpose() * Asel_new).array() - AtABef;
                _AprevtA[m] += (Asel_prev.transpose() * Asel_new).array() - AtABef;
            }
        }
    }
}

TensorStream_coordHybridALS::TensorStream_coordHybridALS(DataStream& paperX, const Config& config)
    : TensorStream_coordSamplingALS(paperX, config)
{
    int numMode = _config->numMode();
    selIdxLists.resize(numMode);
    samIdxLists.resize(numMode);
}

double TensorStream_coordHybridALS::prodAtA(int numMode, int currMode,
    const std::vector<Eigen::ArrayXXd>& squareVec, int currRow, int currCol)
{
    double output = 1.0;

    // update first term
    for (int l = 0; l < numMode; ++l) {
        if (l == currMode)
            continue;

        output *= squareVec[l](currRow, currCol);
    }
    return output;
}

double TensorStream_coordHybridALS::prodA(int numMode, int currMode,
    const std::vector<Eigen::MatrixXd>& factorVec, const std::vector<int>& currIdx, int currCol)
{
    double output = 1.0;

    // update first term
    for (int l = 0; l < numMode; ++l) {
        if (l == currMode)
            continue;

        output *= factorVec[l](currIdx[l], currCol);
    }
    return output;
}

void TensorStream_coordHybridALS::updateA_and_AtA(double newVal, int rank, int currMode, int currRow, int currCol)
{
    // Update an entry of factor matrix
    double prevVal = _A[currMode](currRow, currCol);
    newVal /= _sqSumProd[currCol];
    newVal *= _AtA[currMode](currCol, currCol);

    // Clipping
    if (abs(newVal) >= _higherBound) {
        newVal = std::signbit(newVal) ? (-1.0) * _higherBound : _higherBound;
    }

    _A[currMode](currRow, currCol) = newVal;

    // Update AtA and AprevtA
    double prevNorm = _AtA[currMode](currCol, currCol);
    for (int s = 0; s < rank; ++s) {
        if (s != currCol) {
            double tempDelta = (newVal - prevVal) * _A[currMode](currRow, s);
            _AtA[currMode](currCol, s) += tempDelta;
            _AtA[currMode](s, currCol) += tempDelta;
        } else {
            _AtA[currMode](s, s) += newVal * newVal - prevVal * prevVal;
        }

        _AprevtA[currMode](s, currCol) += _Aprev[currMode](currRow, s) * (newVal - prevVal);
    }

    // Update sqSumProd
    _sqSumProd[currCol] *= (_AtA[currMode](currCol, currCol) / prevNorm);
}

void TensorStream_coordHybridALS::_updateAlgorithm(void)
{
    const std::vector<std::vector<int>>& deltaIdxLists = _dX->idxLists();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();
    const std::vector<SpTensor_Hash::row_vector>& elemsX = _X->elems();

    int rank = _config->rank();
    int numMode = _config->numMode();
    double firstTerm, secondTerm, thirdTerm;
    double prodba, prodaa, proda;
    double prevVal, newVal, prevNorm;

    // Figure out the rows to sample and select
    std::vector<int> numSamIdx(numMode);
    for (int m = 0; m < numMode; ++m) {
        selIdxLists[m].clear();
        samIdxLists[m].clear();

        for (const int i : deltaIdxLists[m]) {
            if (elemsX[m][i].size() <= _numSample) {
                selIdxLists[m].push_back(i);
            } else {
                samIdxLists[m].push_back(i);
            }
        }
        numSamIdx[m] = samIdxLists[m].size();
    }

    // Copy previous factor matrix (only part that will be changed) and AtAprev
    std::copy(_AtA.begin(), _AtA.end(), _AprevtA.begin());
    for (int m = 0; m < numMode; ++m) {
        _Aprev[m](deltaIdxLists[m], Eigen::all) = _A[m](deltaIdxLists[m], Eigen::all);
    }

    // Sample random entries in the changed slice
    Sampling(samIdxLists, numSamIdx);

    // Update in column(same rank) unit
    for (const int& m : _compute_order) {
        for (int r = 0; r < rank; ++r) {
            int numSel = selIdxLists[m].size();
            int numSam = numSamIdx[m];

            // Handle selected entries
            for (int di = 0; di < numSel; ++di) {
                int selIdx = selIdxLists[m][di];

                // Compute the first and third term in slide 200513.pptx, page 4.
                firstTerm = 0.0;
                secondTerm = 0.0;
                thirdTerm = 0.0;

                // Update the first term
                if (m == numMode - 1) {
                    for (int s = 0; s < rank; ++s) {
                        firstTerm += _Aprev[m](selIdx, s) * prodAtA(numMode, m, _AprevtA, s, r);
                    }
                }

                // Update the third term
                for (int s = 0; s < rank; ++s) {
                    if (s == r)
                        continue;

                    thirdTerm += _A[m](selIdx, s) * prodAtA(numMode, m, _AtA, s, r);
                }

                // Get the second term
                const std::unordered_map<std::vector<int>, double>& currSlice = (m == numMode - 1) ? elemsdX[m][selIdx] : elemsX[m][selIdx];
                for (const auto& idxIt : currSlice) {
                    secondTerm += idxIt.second * prodA(numMode, m, _A, idxIt.first, r);
                }

                // Update an entry of factor matrix
                newVal = firstTerm + secondTerm - thirdTerm;
                updateA_and_AtA(newVal, rank, m, selIdx, r);
            }

            // Handle sampled entries
            for (int di = 0; di < numSam; ++di) {
                int samIdx = samIdxLists[m][di];

                // Compute the first and third term in slide 200513.pptx, page 4.
                firstTerm = 0.0;
                secondTerm = 0.0;
                thirdTerm = 0.0;

                // Update the first and the third term
                for (int s = 0; s < rank; ++s) {
                    firstTerm += _Aprev[m](samIdx, s) * prodAtA(numMode, m, _AprevtA, s, r);
                    if (s == r)
                        continue;

                    thirdTerm += _A[m](samIdx, s) * prodAtA(numMode, m, _AtA, s, r);
                }

                // Update the second term
                for (const auto& samIt : _sampledIdx[m][di]) {
                    if (elemsdX[m][samIdx].find(samIt.first) != elemsdX[m][samIdx].end()) {
                        continue;
                    }

                    secondTerm += (_X->find(samIt.first) - samIt.second) * prodA(numMode, m, _A, samIt.first, r);
                }

                for (const auto& chaIt : elemsdX[m][samIdx]) {
                    secondTerm += chaIt.second * prodA(numMode, m, _A, chaIt.first, r);
                }

                // Update an entry of factor matrix
                newVal = firstTerm + secondTerm - thirdTerm;
                updateA_and_AtA(newVal, rank, m, samIdx, r);
            }
        }
    }
}
