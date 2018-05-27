#include "ImageFileUtil.h"
#include "../libs/lodepng.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <fstream>

ImageFileUtil::ImageFileUtil() {}

PngImage* ImageFileUtil::loadPngFile(const char* filename) {
	std::vector<unsigned char> png;
	std::vector<unsigned char> rawImage;

	unsigned imageWidth;
	unsigned imageHeight;

	if (!fileExists(filename)) {
		std::cout << "Cannot find or open file: " << filename << std::endl;
		return nullptr;
	}

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

std::string ImageFileUtil::addPrefix(std::string fullFilePath, const char* prefix) {
	std::vector<std::string> pathParts;

	const char osPathDelimiter = getOsPathDelimiter();
	
	pathParts = splitString(fullFilePath, osPathDelimiter);

	std::string newPathPart = pathParts.back();
	pathParts.pop_back();
	pathParts.push_back(prefix + newPathPart);

	return joinString(pathParts, osPathDelimiter);
}

std::vector<std::string> ImageFileUtil::splitString(std::string stringToSplit, const char delimiter) {
	std::vector<std::string> parts;
	std::istringstream f(stringToSplit);
	std::string part;
	while (std::getline(f, part, delimiter)) {
		parts.push_back(part);
	}
	
	return parts;
}

std::string ImageFileUtil::joinString(std::vector<std::string> strings, const char delimiter) {
	std::string resultString = strings.front();
	for (std::vector<std::string>::size_type i = 1; i != strings.size(); i++) {
		resultString.append(delimiter + strings[i]);
	}

	return resultString;
}

bool ImageFileUtil::fileExists(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

const char ImageFileUtil::getOsPathDelimiter() {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	return '\\';
#else
	return "/";
#endif
}

