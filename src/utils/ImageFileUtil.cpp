#include "../utils/ImageFileUtil.h"
#include "../libs/lodepng.h"
#include <iostream>


ImageFileUtil::ImageFileUtil() {}

PngImage* ImageFileUtil::loadPngFile(const char* filename) {
	std::vector<unsigned char> png;
	std::vector<unsigned char> rawImage;

	unsigned imageWidth;
	unsigned imageHeight;

	unsigned error = lodepng::load_file(png, filename);
	if (!error) {
		error = lodepng::decode(rawImage, imageWidth, imageHeight, png);
	}

	if (error) {
		std::cout << lodepng_error_text(error) << std::endl;
		return nullptr;
	}

	return new PngImage(filename, imageWidth, imageHeight, rawImage);
}

void ImageFileUtil::savePngFile(PngImage* pngImage, const char* newFileName = nullptr) {
	std::vector<unsigned char> png;

	unsigned error = lodepng::encode(png, pngImage->getRawPixelData(), pngImage->getWidth(), pngImage->getHeight());

	const char* filename = newFileName != nullptr ? newFileName : pngImage->getFilename();

	if (!error) lodepng::save_file(png, filename);

	if (error) {
		std::cout << lodepng_error_text(error) << std::endl;
	}
}

