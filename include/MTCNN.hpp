#pragma once

#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace aibum {

struct FaceBox {
	int x, y, size;
};

class MTCNN {
private:
	struct Bbox {
		float score;
		int x1, y1, x2, y2;
		float area, regre_coord[4];
	};

	inline static constexpr float kNMSThresholds[3] = {0.5f, 0.7f, 0.7f};
	inline static constexpr float kMeanValues[3] = {127.5, 127.5, 127.5};
	inline static constexpr float kNormValues[3] = {0.0078125, 0.0078125, 0.0078125};
	inline static constexpr int kMinDetSize = 12;
	inline static constexpr float kThresholds[3] = {0.8f, 0.8f, 0.6f};
	inline static constexpr int kMinSize = 40;
	inline static constexpr float kPreFactor = 0.709f;

public:
	MTCNN(const char *p_net_param, const char *p_net_model, const char *r_net_param, const char *r_net_model,
	      const char *o_net_param, const char *o_net_model);

	std::vector<FaceBox> Detect(const cv::Mat &image) const;

private:
	ncnn::Net m_p_net, m_r_net, m_o_net;

	static std::vector<Bbox> make_bbox(const ncnn::Mat &score, const ncnn::Mat &location, float scale);
	static void nms(std::vector<Bbox> *p_bboxes, float overlap_threshold,
	                bool min_model = false); // union_model by default
	static void refine(std::vector<Bbox> *p_bboxes, int height, int width);
	std::vector<Bbox> run_p_net(const ncnn::Mat &image) const;
	std::vector<Bbox> run_r_net(const ncnn::Mat &image, std::vector<Bbox> &&prev_bboxes) const;
	std::vector<Bbox> run_o_net(const ncnn::Mat &image, std::vector<Bbox> &&prev_bboxes) const;
};

} // namespace aibum
