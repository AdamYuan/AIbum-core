#pragma once

#include <mat.h>

namespace aibum {

struct Image {
	const unsigned char *data;
	int width, height;
	ncnn::Mat::PixelType pixel_type;
};

} // namespace aibum
