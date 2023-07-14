#pragma once

#include <Image.hpp>
#include <net.h>
#include <vector>

namespace aibum {

struct FaceBox {
	int x, y, w, h;
	struct {
		float x, y;
	} landmarks[5];
};

class SCRFD {
public:
	inline SCRFD() = default;
#ifndef AIBUM_CORE_WASM
	void LoadFromFile(const char *model_path);
	inline explicit SCRFD(const char *model_path) { LoadFromFile(model_path); }
#endif
	void LoadFromMemory(const unsigned char *model_bin);
	inline explicit SCRFD(const unsigned char *model_bin) { LoadFromMemory(model_bin); }

	std::vector<FaceBox> Detect(const Image &image) const;
	inline void Clear() { m_net.clear(); }

private:
	struct BBox {
		float x, y, width, height, prob;
		struct {
			float x, y;
		} landmarks[5];
	};

	ncnn::Net m_net;

	static std::vector<BBox> generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
	                                            const ncnn::Mat &bbox_blob, const ncnn::Mat &kps_blob,
	                                            float prob_threshold);
	static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales);
	static std::vector<int> nms_sorted_bboxes(const std::vector<BBox> &faceobjects, float nms_threshold);
};

} // namespace aibum
