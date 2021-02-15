#include "tensor.hpp"
#include "constants.hpp"
#include <cmath>
#include <iterator>
#include <sstream>
#include <string>

SpTensor::SpTensor(const std::vector<int>& dimension)
{
    _dimension = dimension;
    _numMode = _dimension.size();
}

SpTensor_Hash::SpTensor_Hash(const std::vector<int>& dimension)
    : SpTensor(dimension)
{
    _elems.resize(_numMode);
    for (int i = 0; i < _numMode; ++i) {
        _elems[i].resize(_dimension[i]);
    }
}

double SpTensor_Hash::find(const std::vector<int>& coord) const
{
    const coord_map& cmap = _elems[0][coord[0]];
    const coord_map::const_iterator& it = cmap.find(coord);

    return (it != cmap.end()) ? it->second : 0;
}

void SpTensor_Hash::insert(const std::vector<int>& coord, double value)
{
    const double preValue = find(coord);
    const double newValue = preValue + value;
    const bool prevZero = abs(preValue) < TENSOR_MACHINE_EPSILON;
    const bool newZero = abs(newValue) < TENSOR_MACHINE_EPSILON;

    if (abs(newValue) < TENSOR_MACHINE_EPSILON) { // Treated as zero
        if (!prevZero) {
            for (int m = 0; m < _numMode; ++m) {
                _elems[m][coord[m]].erase(coord);
            }
            _numNnz--;
        }
    } else {
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]][coord] = newValue;
        }

        if (prevZero)
            _numNnz++;
    }
}

void SpTensor_Hash::set(const std::vector<int>& coord, double value)
{
    if (abs(value) < TENSOR_MACHINE_EPSILON) { // Treated as zero
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]].erase(coord);
        }
    } else {
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]][coord] = value;
        }
    }
}

void SpTensor_Hash::clear(void)
{
    _numNnz = 0;
    for (int m = 0; m < _numMode; ++m) {
        for (int i = 0; i < _dimension[m]; ++i) {
            _elems[m][i].clear();
        }
    }
}

double SpTensor_Hash::norm_frobenius(void) const
{
    double square_sum = 0;

    const row_vector& elems_vec = _elems[0];
    for (int i = 0; i < _dimension[0]; ++i) {
        for (auto const& it : elems_vec[i]) {
            const double val = it.second;
            square_sum += pow(val, 2);
        }
    }

    return sqrt(square_sum);
}

double SpTensor_Hash::norm_frobenius_latest(void) const
{
    double square_sum = 0;

    const row_vector& elems_vec = _elems[_numMode - 1];
    for (auto const& it : elems_vec[_dimension[_numMode - 1] - 1]) {
        const double val = it.second;
        square_sum += pow(val, 2);
    }

    return sqrt(square_sum);
}

SpTensor_dX::SpTensor_dX(const std::vector<int>& dimension)
    : SpTensor_Hash(dimension)
{
    _idxLists.resize(_numMode); // Initialize index list
    _idxMaps.resize(_numMode); // Initialize index map
}

void SpTensor_dX::insert(const std::vector<int>& coord, double value)
{
    const double preValue = find(coord);
    const double newValue = preValue + value;
    const bool prevZero = abs(preValue) < TENSOR_MACHINE_EPSILON;
    const bool newZero = abs(newValue) < TENSOR_MACHINE_EPSILON;

    if (!prevZero && newZero) { // Treated as zero
        for (int m = 0; m < _numMode; ++m) {
            int currCoord = coord[m];
            _elems[m][currCoord].erase(coord);

            // Reduce counts for each index of each mode
            const int count = _idxMaps[m][currCoord] - 1;
            if (count == 0)
                _idxMaps[m].erase(currCoord);
            else
                _idxMaps[m][currCoord] = count;
        }

        _numNnz--;
    } else if (!newZero) {
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]][coord] = newValue;
        }

        if (prevZero) {
            for (int m = 0; m < _numMode; ++m) {
                int currCoord = coord[m];
                auto got = _idxMaps[m].find(currCoord);

                if (got == _idxMaps[m].end())
                    _idxMaps[m][currCoord] = 1;
                else
                    _idxMaps[m][currCoord] = got->second + 1;
            }

            _numNnz++;
        }
    }
}

std::vector<std::vector<int>>& SpTensor_dX::idxLists(void)
{
    for (int m = 0; m < _numMode; ++m) {
        std::vector<int>& currList = _idxLists[m];
        const std::unordered_map<int, int>& currMap = _idxMaps[m];

        currList.reserve(currMap.size());
        for (auto it = currMap.begin(); it != currMap.end(); ++it)
            currList.push_back(it->first);
    }

    return _idxLists;
}

void SpTensor_dX::clear(void)
{
    if (_idxLists[0].size() > 0) {
        for (int m = 0; m < _numMode; ++m) {
            _idxLists[m].clear();
            _idxMaps[m].clear();
        }
    }

    SpTensor_Hash::clear();
}

SpTensor_List::SpTensor_List(const std::vector<int>& nonTempDim, long long tempDim)
    : SpTensor(nonTempDim)
{
    _numMode += 1;
    _order_time = _numMode - 1;
    const long long& numTime = tempDim;
    _elems.resize(numTime);
}

void SpTensor_List::insert(const std::vector<int>& coord, const long long& timeIdx, double value)
{
    const nnzEntry nnz = std::make_pair(coord, value);
    _elems[timeIdx].push_back(nnz);
}
