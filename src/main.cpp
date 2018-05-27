// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "vld.h"

#include "libs/lodepng.h"
#include "utils/ImageFileUtil.h"
#include "model/PngImage.h"
#include "OtsuBinarizer.h"
#include "CudaOtsuBinarizer.cuh"

int main(int argc, char **argv)
{
	std::string fullFilePath;

	while (1) {
		std::cout << "\nEnter full path to .png image file" << std::endl;
		std::cin >> fullFilePath;

		if (fullFilePath == "exit") {
			break;
		}

		PngImage* loadedImage = ImageFileUtil::loadPngFile(fullFilePath.c_str());

		if (loadedImage != nullptr) {
			std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "cpu_binarized_");

			PngImage* cpuBinarizedImage = OtsuBinarizer::binarize(loadedImage);

			ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

			delete cpuBinarizedImage;

			PngImage* gpuBinarizedImage = CudaOtsuBinarizer::binarize(loadedImage);

			std::string gpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_binarized_");

			ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename.c_str());
			
			delete gpuBinarizedImage;
		}

		delete loadedImage;
	}

	system("pause");
	return 0;
}