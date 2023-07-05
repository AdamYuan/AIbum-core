#pragma once

#include <mat.h>

namespace aibum {

template <int TargetWidth, int TargetHeight, ncnn::Mat::PixelType PixelType>
inline ncnn::Mat Image2Mat(const Image &image) {
	return ncnn::Mat::from_pixels_resize(image.data,
	                                     image.pixel_type == PixelType
	                                         ? image.pixel_type
	                                         : (image.pixel_type | (PixelType << ncnn::Mat::PIXEL_CONVERT_SHIFT)),
	                                     image.width, image.height, TargetWidth, TargetHeight);
}

template <ncnn::Mat::PixelType PixelType>
inline ncnn::Mat Image2Mat(const Image &image, int target_width, int target_height) {
	return ncnn::Mat::from_pixels_resize(image.data,
	                                     image.pixel_type == PixelType
	                                         ? image.pixel_type
	                                         : (image.pixel_type | (PixelType << ncnn::Mat::PIXEL_CONVERT_SHIFT)),
	                                     image.width, image.height, target_width, target_height);
}

} // namespace aibum
