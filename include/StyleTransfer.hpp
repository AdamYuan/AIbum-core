#pragma once

#include "Image.hpp"
#include <ncnn/mat.h>
#include <ncnn/net.h>

namespace aibum {

class StyleTransfer {
public:
	explicit StyleTransfer(const char *model_path);
	explicit StyleTransfer(const unsigned char *model_bin);
	ncnn::Mat Transfer(const Image &image, int target_width, int target_height);

private:
	ncnn::Net m_net;
};

} // namespace aibum
