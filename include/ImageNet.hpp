#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>

namespace aibum {

struct ImageTag {
	int index;
	float score;
};

class ImageNet {
public:
	ImageNet(const char *param, const char *model);
	std::vector<ImageTag> GetTags(const cv::Mat &image, unsigned max_count);

private:
	ncnn::Net m_net;
};

} // namespace aibum
