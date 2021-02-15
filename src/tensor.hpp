#pragma once

#include "utils.hpp"
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

class SpTensor {
public:
    SpTensor(const std::vector<int>& dimension);

    // Getters
    const std::vector<int>& dimension(void) const { return _dimension; }
    int numMode(void) const { return _numMode; }

protected:
    std::vector<int> _dimension;
    int _numMode;
};

class SpTensor_Hash : public SpTensor {
public:
    typedef std::unordered_map<std::vector<int>, double> coord_map;
    typedef std::vector<coord_map> row_vector;

    SpTensor_Hash(const std::vector<int>& dimension);

    double find(const std::vector<int>& coord) const;
    virtual void insert(const std::vector<int>& coord, double value);
    void set(const std::vector<int>& coord, double value); // Override the existed value

    const std::vector<row_vector>& elems(void) const { return _elems; }

    virtual void clear(void);
    const unsigned long long numNnz(void) { return _numNnz; }

    double norm_frobenius(void) const;
    double norm_frobenius_latest(void) const;

protected:
    unsigned long long _numNnz = 0;
    std::vector<row_vector> _elems; // tree-like
};

class SpTensor_dX : public SpTensor_Hash {
public:
    SpTensor_dX(const std::vector<int>& dimension);

    virtual void insert(const std::vector<int>& coord, double value) override;

    std::vector<std::vector<int>>& idxLists(void);

    virtual void clear(void) override;

private:
    std::vector<std::vector<int>> _idxLists;
    std::vector<std::unordered_map<int, int>> _idxMaps;
};

class SpTensor_List : public SpTensor {
public:
    typedef std::pair<std::vector<int>, double> nnzEntry;
    typedef std::list<nnzEntry> nnzEntry_list;

    SpTensor_List(const std::vector<int>& nonTempDim, long long tempDim);

    const nnzEntry_list& findAt(long long time) const { return _elems[time]; };
    virtual void insert(const std::vector<int>& coord, const long long& timeIdx, double value);

private:
    std::vector<nnzEntry_list> _elems;

    int _order_time;
};
