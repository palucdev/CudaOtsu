// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <time.h>
#include <omp.h>

// Memory leaks checking
// #include "vld.h"

#include "libs/lodepng.h"
#include "core/AppRunner.h"
#include "utils/ImageFileUtil.h"
#include "utils/CudaUtil.h"
#include "utils/RunConfigurationBuilder.h"
#include "model/PngImage.h"
#include "model/RunConfiguration.h"
#include "core/binarizers/OtsuBinarizer.h"
#include "core/binarizers/OtsuOpenMPBinarizer.h"
#include "core/binarizers/CudaOtsuBinarizer.cuh"
#include "core/binarizers/SMCudaOtsuBinarizer.cuh"
#include "core/binarizers/MonoCudaOtsuBinarizer.cuh"

std::string getConfigurationInfo(int threadsPerBlock, int numBlocks)
{
	std::vector<std::string> params;
	params.push_back(std::to_string(threadsPerBlock));
	params.push_back(std::to_string(numBlocks));
	params.push_back(CudaUtil::getDeviceName(CudaUtil::getCurrentDevice()));

	return ImageFileUtil::joinString(params, ',');
}

void runCpuOpenmpImplementation(RunConfiguration* runConfig)
{

	printf("\nSetting OpenMP threads num to %d threads\n", runConfig->getCpuThreads());
	omp_set_dynamic(0);
	omp_set_num_threads(runConfig->getCpuThreads());

	clock_t time;
	time = clock();

	std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), "cpu-openmp_binarized_");

	PngImage *cpuBinarizedImage = OtsuOpenMPBinarizer::binarize(runConfig->getLoadedImage(), runConfig->getCpuThreads());

	time = clock() - time;

	printf("\nCPU-OpenMP binarization taken %f seconds\n", ((double)time / CLOCKS_PER_SEC));

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

	delete cpuBinarizedImage;
}

std::string runGpuImplementation(RunConfiguration* runConfig)
{

	CudaOtsuBinarizer *cudaBinarizer = new CudaOtsuBinarizer(
		runConfig->getThreadsPerBlock(),
		runConfig->getNumberOfBlocks(),
		runConfig->shouldDrawHistograms());

	PngImage *gpuBinarizedImage = cudaBinarizer->binarize(runConfig->getLoadedImage());

	std::string gpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), "gpu_binarized_");

	ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename.c_str());

	std::string csvTimesLog = cudaBinarizer->getBinarizerExecutionInfo(runConfig->getFullFilePath());
	std::string configLog = getConfigurationInfo(
		runConfig->getThreadsPerBlock(),
		runConfig->getNumberOfBlocks());

	delete gpuBinarizedImage;
	delete cudaBinarizer;

	return csvTimesLog + "," + configLog;
}

std::string runGpuSharedMemoryImplementation(RunConfiguration* runConfig)
{

	SMCudaOtsuBinarizer *smCudaBinarizer = new SMCudaOtsuBinarizer(
		runConfig->getThreadsPerBlock(),
		runConfig->getNumberOfBlocks(),
		runConfig->shouldDrawHistograms());

	PngImage *sharedMemoryGpuBinarizedImage = smCudaBinarizer->binarize(runConfig->getLoadedImage());

	std::string smGpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), "gpu_shared_memory_binarized_");

	ImageFileUtil::savePngFile(sharedMemoryGpuBinarizedImage, smGpuBinarizedFilename.c_str());

	std::string csvTimesLog = smCudaBinarizer->getBinarizerExecutionInfo(runConfig->getFullFilePath());
	std::string configLog = getConfigurationInfo(runConfig->getThreadsPerBlock(), runConfig->getNumberOfBlocks());

	delete sharedMemoryGpuBinarizedImage;
	delete smCudaBinarizer;

	return csvTimesLog + "," + configLog;
}

void runGpuMonoKernelImplementation(RunConfiguration* runConfig)
{

	MonoCudaOtsuBinarizer *monoCudaBinarizer = new MonoCudaOtsuBinarizer(runConfig->getThreadsPerBlock(), runConfig->shouldDrawHistograms());

	PngImage *monoKernelGpuBinarizedImage = monoCudaBinarizer->binarize(runConfig->getLoadedImage());

	std::string monoKernelGpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), "gpu_mono_binarized_");

	ImageFileUtil::savePngFile(monoKernelGpuBinarizedImage, monoKernelGpuBinarizedFilename.c_str());

	delete monoKernelGpuBinarizedImage;
	delete monoCudaBinarizer;
}

// CudaOtsu filepath/dirpath threadsPerBlock numBlocks
int main(int argc, char **argv)
{
	std::vector<std::string> binarizerTimestamps;
	const char *timestampsFile = "times.csv";

	AppRunner *appRunner = new AppRunner();

	appRunner->loadInputConfiguration(argc, argv);

	RunConfiguration* runConfig = appRunner->getRunConfig();

	runConfig->print();

	if (runConfig->hasLoadedImage())
	{
		// To refactor

		if (runConfig->shouldRunAlgorithm(CPU))
		{
			OtsuBinarizer* cpuBinarizer = new OtsuBinarizer(runConfig->getLoadedImage());
			cpuBinarizer->binarize(runConfig)->printResult();
		}

		if (runConfig->shouldRunAlgorithm(CPU_OpenMP))
		{
			runCpuOpenmpImplementation(runConfig);
		}

		if (runConfig->shouldRunAlgorithm(GPU))
		{
			std::string csvTimeLog = runGpuImplementation(runConfig);
			binarizerTimestamps.push_back(csvTimeLog);
		}

		if (runConfig->shouldRunAlgorithm(GPU_SharedMemory))
		{
			std::string csvTimeLog = runGpuSharedMemoryImplementation(runConfig);
			binarizerTimestamps.push_back(csvTimeLog);
		}

		if (runConfig->shouldRunAlgorithm(GPU_MonoKernel))
		{
			runGpuMonoKernelImplementation(runConfig);
		}
	} else {
		printf("\nFile not loaded");
	}

	ImageFileUtil::saveCsvFile(binarizerTimestamps, timestampsFile);

	return 0;
}
