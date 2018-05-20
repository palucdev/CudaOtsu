// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "lodepng.h"

// CUDA Runtime
#include <cuda_runtime.h>

extern "C" void cudaCalculateHistogram();
extern "C" void cudaBinarize();

void decodePNG(const char*  filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height) {
	std::vector<unsigned char> png;

	unsigned imageWidth;
	unsigned imageHeight;

	unsigned error = lodepng::load_file(png, filename);
	if (!error) error = lodepng::decode(image, imageWidth, imageHeight, png);

	if (error) std::cout << lodepng_error_text(error) << std::endl;

	*width = imageWidth;
	*height = imageHeight;
}

void encodePNG(const char* filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height) {
	std::vector<unsigned char> png;

	unsigned error = lodepng::encode(png, image, *width, *height);
	if (!error) lodepng::save_file(png, filename);

	if (error) std::cout << lodepng_error_text(error) << std::endl;
}

int main(int argc, char **argv)
{
	//Kernel configuration, where a two-dimensional grid and
	//three-dimensional blocks are configured.

	std::vector<unsigned char> image; // raw pixels

	const char* filename = "assets/example_grey.png";
	const char* testCopyFilename = "assets/example_grey_copy.png";
	unsigned width = 0;
	unsigned height = 0;

	decodePNG(filename, image, &width, &height);

	encodePNG(testCopyFilename, image, &width, &height);
	
	cudaCalculateHistogram();
	cudaDeviceSynchronize();
	cudaBinarize();
	cudaDeviceSynchronize();


	system("pause");

	return 0;
}