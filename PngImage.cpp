#include "PngImage.h"



PngImage::PngImage(const char* filename, unsigned width, unsigned height, std::vector<unsigned char> rawPixelData)
{
	this->filename_ = filename;
	this->width_ = width;
	this->height_ = height;
	this->rawPixelData_ = rawPixelData;
	this->totalPixels_ = rawPixelData.size();
}

PngImage::~PngImage(){}

const char * PngImage::getFilename()
{
	return filename_;
}

unsigned PngImage::getWidth()
{
	return width_;
}

void PngImage::setWidth(unsigned width)
{
	this->width_ = width;
}

unsigned PngImage::getHeight()
{
	return height_;
}

void PngImage::setHeight(unsigned height)
{
	this->height_ = height;
}

long int PngImage::getTotalPixels()
{
	return totalPixels_;
}

std::vector<unsigned char> PngImage::getRawPixelData()
{
	return rawPixelData_;
}
