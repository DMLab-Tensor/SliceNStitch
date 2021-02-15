#pragma once

#include <functional>
#include <random>
#include <unordered_set>
#include <vector>

namespace std {
/* New hash function for the std::vector<int> */
template <>
struct hash<vector<int>> {
    size_t operator()(vector<int> const& vec) const
    {
        size_t seed = vec.size();
        for (auto const& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
}

/* Singleton class for generating the random number */
class RNG {
public:
    /* Set whether this RNG is deterministic or not. It should be called once before calling the Instance function. */
    static void SetRandom(bool isRandom)
    {
        _isRandom = isRandom;
    }

    /* Get the instance */
    static RNG* Instance(void)
    {
        if (_instance == nullptr) {
            _instance = new RNG();
        }
        return _instance;
    }

    /* Destroy the instance */
    static void Destroy(void) // Destroy the instance
    {
        delete _instance;
    }

    std::mt19937* rng(void)
    {
        return _rng;
    }

private:
    RNG(void);
    ~RNG(void);

    static RNG* _instance;

    static bool _isRandom;
    std::random_device* _rd;
    std::mt19937* _rng;
};

/*
    Random index generator
    sample k vectors inside the given dimension, return those vectors
*/
void pickIdx(const std::vector<int>& dimension, int currMode, int currIdx,
    int k, std::unordered_map<std::vector<int>, double>& sampledIdx);

void pickIdx(const std::vector<int>& dimension, int k, std::unordered_set<std::vector<int>>& sampledIdx);
