#pragma once

#include "Image.hpp"
#include <mat.h>
#include <net.h>

namespace aibum {

class StyleTransfer {
public:
	inline StyleTransfer() = default;
#ifndef AIBUM_CORE_WASM
	void LoadFromFile(const char *model_path);
	inline explicit StyleTransfer(const char *model_path) { LoadFromFile(model_path); }
#endif
	void LoadFromMemory(const unsigned char *model_bin);
	inline explicit StyleTransfer(const unsigned char *model_bin) { LoadFromMemory(model_bin); }

	ncnn::Mat Transfer(const Image &image, int target_width, int target_height) const;
	inline void Clear() { m_net.clear(); }

private:
	ncnn::Net m_net;
};

} // namespace aibum
