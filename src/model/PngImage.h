#include <vector>
#pragma once
class PngImage
{
public:
	PngImage(const char* filename, unsigned width, unsigned height, std::vector<unsigned char> rawPixelData);
	~PngImage();
	const char* getFilename();
	unsigned getWidth();
	void setWidth(unsigned width);
	unsigned getHeight();
	void setHeight(unsigned height);
	long int getTotalPixels();
	std::vector<unsigned char> getRawPixelData();
	static const unsigned int MAX_PIXEL_VALUE = 256;
	static const unsigned char COLOR_BLACK = 0;
	static const unsigned char COLOR_WHITE = 255;
private:
	const char* filename_;
	unsigned width_;
	unsigned height_;
	long int totalPixels_;
	std::vector<unsigned char> rawPixelData_;
};

