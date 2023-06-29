#pragma once

#include <mat.h>

struct Image {
	unsigned char *data;
	int width, height;
	ncnn::Mat::PixelType pixel_type;
};
