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
	ImageNet();
	std::vector<Tag> GetTags(const Image &image, int count) const;
	inline void Clear() { m_net.clear(); }

private:
	ncnn::Net m_net;
};

} // namespace aibum
