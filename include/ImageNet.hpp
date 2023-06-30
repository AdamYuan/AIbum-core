#pragma once

#include <Image.hpp>
#include <ncnn/net.h>

namespace aibum {

struct Tag {
	int index;
	float score;
};

class ImageNet {
public:
	ImageNet();
	std::vector<Tag> GetTags(const Image &image, int count) const;

private:
	ncnn::Net m_net;
};

} // namespace aibum
