#pragma once

#include <any>
#include <fstream>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <vector>

class DataStream {
public:
    struct Event {
        std::vector<int> nonTempCoord;
        long long tempIdx;
        double val;
        int newUnitIdx;
        int remainderIdx;
    };

    DataStream(const std::string& dataPath, const int modeNum, const int unitNum, const int unitSize, const long long startTime);
    ~DataStream(void);

    void update(long long time);
    Event* pop(void);

private:
    std::ifstream _file;
    const int _modeNum;
    const int _unitNum;
    const int _unitSize;

    std::list<Event*> _events;
    std::vector<std::list<Event*>::iterator> _pts;

    std::queue<Event*> _q;

    Event* parseLine(void);
};

class SpTensor_List;
enum class Init;

class Config {
public:
    Config(const std::string& configPath);
    ~Config(void);

    DataStream* loadDataStream(void) const; // Load CSV format dataStream

    // Getters
    const std::string& dataPath(void) const { return _dataPath; }
    const std::vector<int>& nonTempDim(void) const { return _nonTempDim; }

    int numMode(void) const { return _numMode; }
    long long timeLen(void) const { return _timeLen; }

    int unitNum(void) const { return _unitNum; }
    int unitSize(void) const { return _unitSize; }
    int rank(void) const { return _rank; }
    Init init(void) const { return _init; }

    std::string algo(void) const { return _algo; }
    const std::map<std::string, std::any>& algo_settings(void) const { return _algo_settings; }

    template <typename T>
    const T findAlgoSettings(const std::string& settingName) const
    {
        return std::any_cast<T>(_algo_settings.find(settingName)->second);
    }

    long long startTime(void) const { return _startTime; }
    long long numEpoch(void) const { return _numEpoch; }
    int checkPeriod(void) const { return _checkPeriod; }
    int updatePeriod(void) const { return _updatePeriod; }
    std::string outputPath(void) const { return _outputPath; }

private:
    std::string _dataPath;
    std::vector<int> _nonTempDim;

    int _numMode;
    long long _timeLen;

    int _unitNum;
    int _unitSize;
    int _rank;

    std::string _algo;
    std::map<std::string, std::any> _algo_settings;

    long long _startTime;
    long long _numEpoch;
    int _checkPeriod;

    int _updatePeriod;
    std::string _outputPath;

    Init _init;
};
