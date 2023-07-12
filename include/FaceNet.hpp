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
	inline FaceNet() = default;
#ifndef AIBUM_CORE_WASM
	void LoadFromFile(const char *model_path);
	inline explicit FaceNet(const char *model_path) { LoadFromFile(model_path); }
#endif
	void LoadFromMemory(const unsigned char *model_bin);
	inline explicit FaceNet(const unsigned char *model_bin) { LoadFromMemory(model_bin); }

	FaceFeature GetFeature(const Image &image) const;
	std::vector<Face> GetFaces(const SCRFD &scrfd, const Image &image) const;
	inline void Clear() { m_net.clear(); }

private:
	ncnn::Net m_net;

	FaceFeature get_feature_rgb_112_112(const ncnn::Mat &image) const;
};

} // namespace aibum
