#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace aibum {

struct FaceBox {
	int x, y, size;
};

class SCRFD {
private:
	struct BBox {
		float x, y, width, height, prob;
	};

public:
	SCRFD();

	std::vector<FaceBox> Detect(const cv::Mat &image) const;

private:
	ncnn::Net m_net;

	static std::vector<BBox> generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
	                                            const ncnn::Mat &bbox_blob, float prob_threshold);
	static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales);
	static std::vector<int> nms_sorted_bboxes(const std::vector<BBox> &faceobjects, float nms_threshold);
};

} // namespace aibum
