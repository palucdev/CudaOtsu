#include <string>
#include "ExecutionTimestamp.h"

#pragma once
class BinarizationResult
{
private:
    ExecutionTimestamp *executionTimestamp;
    unsigned int method;
    std::string binarizedImagePath;

public:
    BinarizationResult();
    ~BinarizationResult();
};
