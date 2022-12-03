#include "../model/BinarizationResult.h"
#include "../model/RunConfiguration.h"

class AbstractBinarizer
{
public:
    virtual BinarizationResult *binarize(RunConfiguration *runConfig) = 0;
private:
    virtual MethodImplementation getBinarizerType() = 0;
};