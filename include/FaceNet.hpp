#pragma once

#include <Image.hpp>
#include <SCRFD.hpp>
#include <array>
#include <net.h>

namespace aibum {

using FaceFeature = std::array<float, 128>;

struct Face {
	int x, y, w, h;
	FaceFeature feature;
};

class FaceNet {
public:
	FaceNet();
	FaceFeature GetFeature(const Image &image) const;
	std::vector<Face> GetFaces(const SCRFD &scrfd, const Image &image) const;
	inline void Clear() { m_net.clear(); }

private:
	ncnn::Net m_net;

	FaceFeature get_feature_rgb_112_112(const ncnn::Mat &image) const;
};

} // namespace aibum
