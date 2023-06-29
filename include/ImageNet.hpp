#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>

namespace aibum {

struct Tag {
	int index;
	float score;
};

class ImageNet {
public:
	ImageNet();
	std::vector<Tag> GetTags(const cv::Mat &image, unsigned count);

private:
	ncnn::Net m_net;
};

} // namespace aibum
