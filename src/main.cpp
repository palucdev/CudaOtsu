// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

// Memory leaks checking
// #include "vld.h"

#include "libs/lodepng.h"
#include "utils/ImageFileUtil.h"
#include "model/PngImage.h"
#include "OtsuBinarizer.h"
#include "CudaOtsuBinarizer.cuh"

// CudaOtsu filepath/dirpath threadsPerBlock numBlocks
int main(int argc, char **argv)
{
	std::string fullFilePath;
	int threadsPerBlock;
	int numBlocks;

	if (argc > 1) {
		fullFilePath = argv[1];
		if (argc > 3) {
			threadsPerBlock =  (int)argv[2];
			numBlocks = (int)argv[3];
		} else {
			threadsPerBlock = 256;
			numBlocks = 256;
		}
	}

	PngImage* loadedImage = ImageFileUtil::loadPngFile(fullFilePath.c_str());

	if (loadedImage != nullptr) {
		std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "cpu_binarized_");

		PngImage* cpuBinarizedImage = OtsuBinarizer::binarize(loadedImage);

		ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

		delete cpuBinarizedImage;

		CudaOtsuBinarizer* cudaBinarizer = new CudaOtsuBinarizer(threadsPerBlock, numBlocks);

		PngImage* gpuBinarizedImage = cudaBinarizer->binarize(loadedImage);

		std::string gpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_binarized_");

		ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename.c_str());

		delete gpuBinarizedImage;
	}

	delete loadedImage;

	system("pause");
	return 0;
}
