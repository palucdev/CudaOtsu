#include "ImageFileUtil.h"
#include "lodepng.h"
#include <iostream>
#include <vector>


ImageFileUtil::ImageFileUtil() {}

void ImageFileUtil::loadPngFile(const char* filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height) {
	std::vector<unsigned char> png;

	unsigned imageWidth;
	unsigned imageHeight;

	unsigned error = lodepng::load_file(png, filename);
	if (!error) error = lodepng::decode(image, imageWidth, imageHeight, png);

	if (error) std::cout << lodepng_error_text(error) << std::endl;

	*width = imageWidth;
	*height = imageHeight;
}

void ImageFileUtil::savePngFile(const char* filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height) {
	std::vector<unsigned char> png;

	unsigned error = lodepng::encode(png, image, *width, *height);
	if (!error) lodepng::save_file(png, filename);

	if (error) std::cout << lodepng_error_text(error) << std::endl;
}

