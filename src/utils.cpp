#include "utils.hpp"

RNG* RNG::_instance = nullptr;
bool RNG::_isRandom = true;

RNG::RNG(void)
{
    if (_isRandom) {
        _rd = new std::random_device();
        _rng = new std::mt19937((*_rd)());
    } else {
        _rng = new std::mt19937(0);
    }
}

RNG::~RNG(void)
{
    delete _rd;
    delete _rng;
}

void pickIdx(const std::vector<int>& dimension, int currMode,
    int currIdx, int k, std::unordered_map<std::vector<int>, double>& sampledIdx)
{
    std::mt19937* rng = RNG::Instance()->rng();
    const int ndim = dimension.size();
    std::vector<int> r = dimension;

    /*
        It uses Robert Floyd's algorithm for sampling without replacement.
        We slightly modified the algorithm to sample the vectors.
    */

    // Define the lambda function which increases r by carry in lexicographical order
    auto r_increase = [ndim, currMode, &r, &dimension](int carry) {
        for (int m = 0; m < ndim; ++m) {
            if (m == currMode)
                continue;

            if (carry == 0) {
                break;
            }
            r[m] += carry;
            int remainder = (r[m] % dimension[m] + dimension[m]) % dimension[m];
            carry = (r[m] - remainder) / dimension[m];
            r[m] = remainder;
        }
    };

    {
        // Set r as the maximum possible index (= N - 1) in lexicographical order
        for (int m = 0; m < ndim; ++m) {
            if (m == currMode) {
                r[m] = currIdx;
                continue;
            }

            r[m] -= 1;
        }

        // Set r as N - k - 1 in lexicographical order
        r_increase(-k);
    }

    std::vector<int> v(ndim);
    v[currMode] = currIdx;
    for (int i = 0; i < k; ++i) {
        // r += 1 in lexicographical order
        r_increase(1);

        // Sample one vector
        for (int m = ndim - 1; m >= 0; --m) {
            if (m == currMode)
                continue;

            v[m] = std::uniform_int_distribution<>(0, r[m])(*rng);
            if (v[m] < r[m]) {
                for (int n = 0; n < m; ++n) {
                    if (n == currMode)
                        continue;

                    v[n] = std::uniform_int_distribution<>(0, dimension[n] - 1)(*rng);
                }
                break;
            }
        }

        if (sampledIdx.find(v) != sampledIdx.end()) {
            sampledIdx[r] = 0.0;
        } else {
            sampledIdx[v] = 0.0;
        }
    }
}

void pickIdx(const std::vector<int>& dimension, int k, std::unordered_set<std::vector<int>>& sampledIdx)
{
    std::mt19937* rng = RNG::Instance()->rng();
    const int ndim = dimension.size();
    std::vector<int> r = dimension;

    /*
        It uses Robert Floyd's algorithm for sampling without replacement.
        We slightly modified the algorithm to sample the vectors.
    */

    // Define the lambda function which increases r by carry in lexicographical order
    auto r_increase = [ndim, &r, &dimension](int carry) {
        for (int m = 0; m < ndim; ++m) {
            if (carry == 0) {
                break;
            }
            r[m] += carry;
            int remainder = (r[m] % dimension[m] + dimension[m]) % dimension[m];
            carry = (r[m] - remainder) / dimension[m];
            r[m] = remainder;
        }
    };

    {
        // Set r as the maximum possible index (= N - 1) in lexicographical order
        for (int m = 0; m < ndim; ++m) {
            r[m] -= 1;
        }

        // Set r as N - k - 1 in lexicographical order
        r_increase(-k);
    }

    std::vector<int> v(ndim);
    for (int i = 0; i < k; ++i) {
        // r += 1 in lexicographical order
        r_increase(1);

        // Sample one vector
        for (int m = ndim - 1; m >= 0; --m) {
            v[m] = std::uniform_int_distribution<>(0, r[m])(*rng);
            if (v[m] < r[m]) {
                for (int n = 0; n < m; ++n) {
                    v[n] = std::uniform_int_distribution<>(0, dimension[n] - 1)(*rng);
                }
                break;
            }
        }

        // Currently, it cannot treat the case that r is already inside the set
        if (!sampledIdx.insert(v).second) {
            sampledIdx.insert(r);
        }
    }
}
