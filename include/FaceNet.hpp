#pragma once

#include <array>
#include <net.h>
#include <opencv2/opencv.hpp>

namespace aibum {

class FaceNet {
public:
	FaceNet(const char *param, const char *model);
	std::array<float, 128> GetFeature(const cv::Mat &image);

private:
	ncnn::Net m_net;
};

} // namespace aibum
