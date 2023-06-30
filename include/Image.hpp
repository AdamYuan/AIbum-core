#pragma once

#include <ncnn/mat.h>

namespace aibum {

struct Image {
	unsigned char *data;
	int width, height;
	ncnn::Mat::PixelType pixel_type;
};

} // namespace aibum