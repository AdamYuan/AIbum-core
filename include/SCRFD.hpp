#pragma once

#include <Image.hpp>
#include <net.h>
#include <vector>

namespace aibum {

struct FaceBox {
	int x, y, w, h;
};

class SCRFD {
public:
	SCRFD();

	std::vector<FaceBox> Detect(const Image &image) const;

private:
	struct BBox {
		float x, y, width, height, prob;
	};

	ncnn::Net m_net;

	static std::vector<BBox> generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
	                                            const ncnn::Mat &bbox_blob, float prob_threshold);
	static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales);
	static std::vector<int> nms_sorted_bboxes(const std::vector<BBox> &faceobjects, float nms_threshold);
};

} // namespace aibum
