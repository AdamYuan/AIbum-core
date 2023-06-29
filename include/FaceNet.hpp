#pragma once

#include <Image.hpp>
#include <SCRFD.hpp>
#include <array>
#include <net.h>

namespace aibum {

struct Face {
	int x, y, w, h;
	float feature[128];
};

struct FaceFeature {
	float feature[128];
};

class FaceNet {
public:
	FaceNet();
	FaceFeature GetFeature(const Image &image) const;
	std::vector<Face> GetFaces(const SCRFD &scrfd, const Image &image) const;

private:
	ncnn::Net m_net;

	FaceFeature get_feature_rgb_112_112(const ncnn::Mat &image) const;
};

} // namespace aibum
