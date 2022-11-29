#include "AppRunner.h"
#include "../utils/CudaUtil.h"
#include "../utils/ImageFileUtil.h"
#include "../utils/RunConfigurationBuilder.h"

AppRunner::AppRunner() {}

void AppRunner::loadInputConfiguration(int argc, char **argv)
{

    RunConfigurationBuilder configBuilder = RunConfigurationBuilder();

    std::string fullFilePath;
    int cudaDeviceId;

    if (argc <= 3)
    {
        printHelp();
        CudaUtil::getAvailableGpuNames();
    }
    else
    {
        fullFilePath = argv[1];
        configBuilder.forFileInPath(fullFilePath);
        configBuilder.withThreadsPerBlock(parseIntInputParam(argv[2], DEFAULT_THREADS_NUMBER));
        configBuilder.withNumberOfBlocks(parseIntInputParam(argv[3], DEFAULT_BLOCKS_NUMBER));
        configBuilder.withCpuThreads(parseIntInputParam(argv[4], DEFAULT_CPU_THREADS));
        configBuilder.withHistograms(false);

        for (int argumentIndex = 5; argumentIndex < argc; argumentIndex++)
        {
            std::string flag(argv[argumentIndex]);

            if (flag == "-h")
            {
                configBuilder.withHistograms(true);
                continue;
            }

            if (flag == "-d")
            {
                int nextArgument = argumentIndex + 1;
                if (nextArgument < argc)
                {
                    cudaDeviceId = std::atoi(argv[nextArgument]);

                    bool gpuSetSuccess = CudaUtil::setGpu(cudaDeviceId);

                    if (!gpuSetSuccess)
                    {
                        CudaUtil::getAvailableGpuNames();
                    }

                    argumentIndex = nextArgument;
                    continue;
                }
            }

            if (flag == "--cpu")
            {
                configBuilder.withAlgorithmToRun(CPU);
                continue;
            }

            if (flag == "--cpu-openmp")
            {
                configBuilder.withAlgorithmToRun(CPU_OpenMP);
                continue;
            }

            if (flag == "--gpu")
            {
                configBuilder.withAlgorithmToRun(GPU);
                continue;
            }

            if (flag == "--gpu-sm")
            {
                configBuilder.withAlgorithmToRun(GPU_SharedMemory);
                continue;
            }

            if (flag == "--gpu-mono")
            {
                configBuilder.withAlgorithmToRun(GPU_MonoKernel);
                continue;
            }

            if (flag == "--run-all")
            {
                configBuilder.withAlgorithmToRun(ALL);
                continue;
            }
        }
    }

     this->runConfig = configBuilder
                          .forImage(ImageFileUtil::loadPngFile(fullFilePath.c_str()))
                          .build();
}

RunConfiguration* AppRunner::getRunConfig()
{
    return this->runConfig;
}

void AppRunner::printHelp()
{
    std::string helpMessage = "";
    helpMessage.append("Help:\n");
    helpMessage.append("<program> filePath cudaThreadsNumber cudaBlocksNumber [optional flags]\n");
    helpMessage.append("\tFlags:\n");
    helpMessage.append("\t\t -h show histogram values for each binarizer run\n");
    helpMessage.append("\t\t -d <deviceName> choose GPU device by given name (defaults to 0)\n");
    helpMessage.append("\t\t --cpu run CPU version of algorithm\n");
    helpMessage.append("\t\t --cpu-openmp run CPU with OpenMP version of algorithm\n");
    helpMessage.append("\t\t --gpu run GPU reference version of algorithm\n");
    helpMessage.append("\t\t --gpu-sm run GPU version of algorithm with shared memory optimization\n");
    helpMessage.append("\t\t --gpu-mono run GPU version of algorithm with single kernel arch on single block\n");
    helpMessage.append("\t\t --run-all run all implemented versions of Otsu algorithm (CPU and GPU)\n");

    printf(helpMessage.c_str());
}

int AppRunner::parseIntInputParam(const char *param, int defaultValue)
{
    return std::atoi(param) > 0 ? std::atoi(param) : defaultValue;
}