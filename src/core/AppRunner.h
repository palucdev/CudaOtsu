#include "../model/RunConfiguration.h"
#include "../model/BinarizationResult.h"
#include <map>

class AppRunner {
private:
	static const int DEFAULT_THREADS_NUMBER = 512;
	static const int DEFAULT_BLOCKS_NUMBER = 512;
	static const int DEFAULT_CPU_THREADS = 16;

    RunConfiguration* runConfig;
    std::map<MethodImplementation, BinarizationResult *> binarizationResults;
    void printHelp();
    int parseIntInputParam(const char *param, int defaultValue);
public:
    AppRunner();
    ~AppRunner();
    RunConfiguration* getRunConfig();
    void loadInputConfiguration(int argc, char **argv);
    void runBinarization();
};