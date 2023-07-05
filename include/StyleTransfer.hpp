#pragma once

#include "Image.hpp"
#include <mat.h>
#include <net.h>

namespace aibum {

class StyleTransfer {
public:
#ifndef AIBUM_CORE_WASM
	explicit StyleTransfer(const char *model_path);
#endif
	explicit StyleTransfer(const unsigned char *model_bin);
	ncnn::Mat Transfer(const Image &image, int target_width, int target_height);

private:
	ncnn::Net m_net;
};

} // namespace aibum
