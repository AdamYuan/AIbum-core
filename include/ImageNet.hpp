#pragma once

#include <Image.hpp>
#include <net.h>

namespace aibum {

struct Tag {
	int index;
	float score;
};

class ImageNet {
public:
	inline ImageNet() = default;
#ifndef AIBUM_CORE_WASM
	void LoadFromFile(const char *model_path);
	inline explicit ImageNet(const char *model_path) { LoadFromFile(model_path); }
#endif
	void LoadFromMemory(const unsigned char *model_bin);
	inline explicit ImageNet(const unsigned char *model_bin) { LoadFromMemory(model_bin); }

	std::vector<Tag> GetTags(const Image &image, int count) const;
	inline void Clear() { m_net.clear(); }

private:
	ncnn::Net m_net;
};

} // namespace aibum
