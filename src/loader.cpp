#include "loader.hpp"
#include "tensor.hpp"
#include "tensorStream.hpp"
#include "utils.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <yaml-cpp/yaml.h>

DataStream::DataStream(const std::string& dataPath, const int modeNum, const int unitNum, const int unitSize, const long long startTime)
    : _file(dataPath)
    , _modeNum(modeNum)
    , _unitNum(unitNum)
    , _unitSize(unitSize)
    , _pts(unitNum)
{
    const long long endTime = startTime + unitNum * unitSize;

    // Load initial data from the file
    while (true) {
        std::istream::pos_type sol = _file.tellg(); // start of the line

        Event* e = parseLine();
        if (e == nullptr) {
            break;
        } else if (e->tempIdx < startTime) {
            delete e;
            continue;
        } else if (e->tempIdx >= endTime) {
            _file.seekg(sol); // Return back to the start of the line
            delete e;
            break;
        }

        e->newUnitIdx = (e->tempIdx - startTime) / _unitSize;
        e->remainderIdx = e->tempIdx % _unitSize;

        _events.push_front(e);
        _q.push(e);
    }

    // Set the pointers for each unit
    for (int i = 0; i < _unitNum; ++i) {
        _pts[i] = _events.end();
    }

    int processIdx = _unitNum - 1;

    for (auto itr = _events.begin(); itr != _events.end(); ++itr) {
        while (processIdx > (*itr)->newUnitIdx && processIdx >= 0) {
            _pts[processIdx] = itr;
            --processIdx;
        }

        if (processIdx < 0) {
            break;
        }
    }
}

DataStream::~DataStream(void)
{
    _file.close();

    for (Event* e : _events) {
        delete e;
    }
}

void DataStream::update(long long time)
{
    // Remove unnecessary nodes from the list
    {
        auto itr = _pts[0];
        if (itr != _events.end()) {
            ++itr;
            while (itr != _events.end()) {
                delete *itr;
                itr = _events.erase(itr);
            }
        }
    }

    // Parse from the list
    {
        const int remainderIdx_curr = time % _unitSize;

        for (int u = 0; u < _unitNum; ++u) {
            auto itr = _pts[u];
            while (itr != _events.begin()) {
                --itr;
                if ((*itr)->newUnitIdx == u && (*itr)->remainderIdx == remainderIdx_curr) {
                    --(*itr)->newUnitIdx;
                    _q.push(*itr);
                    _pts[u] = itr;
                }
            }
        }
    }

    // Load the data from the file until the given time
    while (true) {
        std::istream::pos_type sol = _file.tellg(); // start of the line

        Event* e = parseLine();
        if (e == nullptr) {
            break;
        } else if (e->tempIdx > time) {
            _file.seekg(sol); // Return back to the start of the line
            delete e;
            break;
        }

        e->newUnitIdx = _unitNum - 1;
        e->remainderIdx = e->tempIdx % _unitSize;

        _events.push_front(e);
        _q.push(e);
    }
}

DataStream::Event* DataStream::pop(void)
{
    if (_q.empty()) {
        return nullptr;
    }

    Event* e = _q.front();
    _q.pop();

    return e;
}

DataStream::Event* DataStream::parseLine(void)
{
    if (_file.eof()) {
        return nullptr;
    }

    // Read one line from the file
    std::string line;
    std::getline(_file, line);

    if (line.empty()) {
        return nullptr;
    }

    Event* e = new Event; // It will be the output event
    {
        e->nonTempCoord.reserve(_modeNum - 1);

        std::string str_buf;
        std::stringstream ss(line);
        for (int colIdx = 0; colIdx < _modeNum - 1; ++colIdx) {
            std::getline(ss, str_buf, ',');
            e->nonTempCoord[colIdx] = std::stoi(str_buf);
        }

        std::getline(ss, str_buf, ',');
        e->tempIdx = std::stoll(str_buf);

        std::getline(ss, str_buf, ',');
        e->val = std::stod(str_buf);
    }
    return e;
}

Config::Config(const std::string& configPath)
{
    const YAML::Node config = YAML::LoadFile(configPath);
    _dataPath = config["data"]["filePath"].as<std::string>();
    _nonTempDim = config["data"]["nonTempDim"].as<std::vector<int>>();

    _numMode = _nonTempDim.size() + 1;
    _timeLen = config["data"]["tempDim"].as<long long>();

    _unitNum = config["tensor"]["unitNum"].as<int>();
    _unitSize = config["tensor"]["unitSize"].as<int>();
    _rank = config["tensor"]["rank"].as<int>();

    _outputPath = config["test"]["outputPath"].as<std::string>();
    _startTime = config["test"]["startTime"].as<long long>();
    _numEpoch = config["test"]["numEpoch"].as<long long>();
    _checkPeriod = config["test"]["checkPeriod"].as<int>();
    _updatePeriod = config["test"]["updatePeriod"].as<int>();

    // If randomness is required, then change the seed number into current time
    const bool isRandom = config["test"]["random"].as<bool>();
    RNG::SetRandom(isRandom);

    // Check the data
    if ((long long)_unitNum * (long long)_unitSize + _numEpoch > timeLen()) {
        std::cerr << "Error: Not sufficient data. Too short time-length" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Algorithm
    _algo = config["algorithm"]["name"].as<std::string>();
    const YAML::Node algo_settings = config["algorithm"]["settings"];
    {
        if (_algo == "SNS_MAT") {
            _algo_settings["numIter"] = algo_settings["numIter"].as<int>();
        } else if (_algo == "SNS_VEC+") {
            _algo_settings["clipping"] = algo_settings["clipping"].as<double>();
        } else if (_algo == "SNS_RND") {
            _algo_settings["numSample"] = algo_settings["numSample"].as<int>();
        } else if (_algo == "SNS_RND+") {
            _algo_settings["numSample"] = algo_settings["numSample"].as<int>();
            _algo_settings["clipping"] = algo_settings["clipping"].as<double>();
        } else if (_algo != "SNS_VEC" && _algo != "ALS") {
            std::cerr << "Error: This algorithm is not allowed" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

Config::~Config(void)
{
    RNG::Destroy();
}

DataStream* Config::loadDataStream(void) const
{
    return new DataStream(_dataPath, _numMode, _unitNum, _unitSize, _startTime);
}
