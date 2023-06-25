#include <MTCNN.hpp>

namespace aibum {

MTCNN::MTCNN(const char *model_dir) {
	m_p_net.load_param((std::string{model_dir} + "det1.param").c_str());
	m_p_net.load_model((std::string{model_dir} + "det1.bin").c_str());

	m_r_net.load_param((std::string{model_dir} + "det2.param").c_str());
	m_r_net.load_model((std::string{model_dir} + "det2.bin").c_str());

	m_o_net.load_param((std::string{model_dir} + "det3.param").c_str());
	m_o_net.load_model((std::string{model_dir} + "det3.bin").c_str());
}

std::vector<FaceBox> MTCNN::Detect(const cv::Mat &image) const {
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	ncnn_img.substract_mean_normalize(kMeanValues, kNormValues);

	std::vector<MTCNN::Bbox> bboxes = run_p_net(ncnn_img);
	if (bboxes.empty())
		return {};
	// printf("bbox.size() = %ld\n", bboxes.size());
	nms(&bboxes, kNMSThresholds[0]);
	// printf("bbox.size() = %ld\n", bboxes.size());
	refine(&bboxes, ncnn_img.h, ncnn_img.w);
	// printf("bbox.size() = %ld\n", bboxes.size());

	bboxes = run_r_net(ncnn_img, std::move(bboxes));
	if (bboxes.empty())
		return {};
	// printf("bbox.size() = %ld\n", bboxes.size());
	nms(&bboxes, kNMSThresholds[1]);
	// printf("bbox.size() = %ld\n", bboxes.size());
	refine(&bboxes, ncnn_img.h, ncnn_img.w);
	// printf("bbox.size() = %ld\n", bboxes.size());

	bboxes = run_o_net(ncnn_img, std::move(bboxes));
	if (bboxes.empty())
		return {};
	// printf("bbox.size() = %ld\n", bboxes.size());
	refine(&bboxes, ncnn_img.h, ncnn_img.w);
	// printf("bbox.size() = %ld\n", bboxes.size());
	nms(&bboxes, kNMSThresholds[2], true);
	// printf("bbox.size() = %ld\n", bboxes.size());

	std::vector<FaceBox> ret;
	ret.reserve(bboxes.size());
	for (const auto &bbox : bboxes)
		ret.push_back({bbox.x1, bbox.y1, bbox.x2 - bbox.x1});
	return ret;
}

std::vector<MTCNN::Bbox> MTCNN::make_bbox(const ncnn::Mat &score, const ncnn::Mat &location, float scale) {
	std::vector<MTCNN::Bbox> ret;

	constexpr int kStride = 2;
	constexpr int kCellSize = 12;
	// score p
	const float *p = score.channel(1); // score.data + score.cstep;
	Bbox bbox{};
	float inv_scale = 1.0f / scale;
	for (int row = 0; row < score.h; ++row) {
		for (int col = 0; col < score.w; ++col, ++p) {
			if (*p > kThresholds[0]) {
				bbox.score = *p;
				bbox.x1 = (int)std::round(float(kStride * col + 1) * inv_scale);
				bbox.y1 = (int)std::round(float(kStride * row + 1) * inv_scale);
				bbox.x2 = (int)std::round(float(kStride * col + 1 + kCellSize) * inv_scale);
				bbox.y2 = (int)std::round(float(kStride * row + 1 + kCellSize) * inv_scale);
				bbox.area = float(bbox.x2 - bbox.x1) * float(bbox.y2 - bbox.y1);
				const int index = row * score.w + col;
				for (int channel = 0; channel < 4; channel++) {
					bbox.regre_coord[channel] = location.channel(channel)[index];
				}
				ret.push_back(bbox);
			}
		}
	}
	return ret;
}

void MTCNN::nms(std::vector<Bbox> *p_bboxes, float overlap_threshold, bool min_model) {
	auto bboxes = *p_bboxes;
	if (bboxes.empty())
		return;
	std::sort(bboxes.begin(), bboxes.end(), [](const Bbox &l, const Bbox &r) { return l.score < r.score; });
	std::vector<int> v_pick;
	int n_pick = 0;
	// TODO: WTF is this ? just sort it and iterate it

	std::multimap<float, int> v_scores;
	const int num_boxes = (int)bboxes.size();
	v_pick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i)
		v_scores.insert(std::pair<float, int>(bboxes[i].score, i));

	while (!v_scores.empty()) {
		int last = v_scores.rbegin()->second;
		v_pick[n_pick++] = last;
		for (auto it = v_scores.begin(); it != v_scores.end();) {
			int it_idx = it->second;

			int max_x = std::max(bboxes[it_idx].x1, bboxes[last].x1);
			int max_y = std::max(bboxes[it_idx].y1, bboxes[last].y1);
			int min_x = std::min(bboxes[it_idx].x2, bboxes[last].x2);
			int min_y = std::min(bboxes[it_idx].y2, bboxes[last].y2);
			// maxX1 and maxY1 reuse
			max_x = std::max(min_x - max_x + 1, 0);
			max_y = std::max(min_y - max_y + 1, 0);
			// IOU reuse for the area of two bbox
			float iou = (float)max_x * (float)max_y;
			if (min_model)
				iou = iou / std::min(bboxes[it_idx].area, bboxes[last].area);
			else
				iou = iou / (bboxes[it_idx].area + bboxes[last].area - iou);

			if (iou > overlap_threshold)
				it = v_scores.erase(it);
			else
				++it;
		}
	}

	v_pick.resize(n_pick);
	std::vector<Bbox> ret(n_pick);
	for (int i = 0; i < n_pick; i++)
		ret[i] = bboxes[v_pick[i]];

	*p_bboxes = std::move(ret);
}

void MTCNN::refine(std::vector<Bbox> *p_bboxes, int height, int width) {
	for (auto &bbox : *p_bboxes) {
		auto bbw = float(bbox.x2 - bbox.x1 + 1);
		auto bbh = float(bbox.y2 - bbox.y1 + 1);
		float x1 = (float)bbox.x1 + bbox.regre_coord[0] * bbw;
		float y1 = (float)bbox.y1 + bbox.regre_coord[1] * bbh;
		float x2 = (float)bbox.x2 + bbox.regre_coord[2] * bbw;
		float y2 = (float)bbox.y2 + bbox.regre_coord[3] * bbh;

		// Square
		float w = x2 - x1 + 1;
		float h = y2 - y1 + 1;
		float max_side = std::max(w, h);
		x1 = x1 + w * 0.5f - max_side * 0.5f;
		y1 = y1 + h * 0.5f - max_side * 0.5f;
		bbox.x2 = (int)std::round(x1 + max_side - 1);
		bbox.y2 = (int)std::round(y1 + max_side - 1);
		bbox.x1 = (int)std::round(x1);
		bbox.y1 = (int)std::round(y1);

		// boundary check
		bbox.x1 = std::max(bbox.x1, 0);
		bbox.y1 = std::max(bbox.y1, 0);
		bbox.x2 = std::min(bbox.x2, width - 1);
		bbox.y2 = std::min(bbox.y2, height - 1);

		bbox.area = float(bbox.x2 - bbox.x1) * float(bbox.y2 - bbox.y1);
	}
}

std::vector<MTCNN::Bbox> MTCNN::run_p_net(const ncnn::Mat &image) const {
	auto minl = (float)std::min(image.w, image.h);
	float m = (float)kMinDetSize / kMinSize;
	minl *= m;
	float factor = kPreFactor;
	std::vector<float> scales;
	while (minl > kMinDetSize) {
		scales.push_back(m);
		minl *= factor;
		m = m * factor;
	}

	std::vector<Bbox> ret;
	for (float &scale : scales) {
		int hs = (int)ceil((float)image.h * scale);
		int ws = (int)ceil((float)image.w * scale);
		ncnn::Mat in;
		ncnn::resize_bilinear(image, in, ws, hs);
		ncnn::Extractor ex = m_p_net.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score, location;
		ex.extract("prob1", score);
		ex.extract("conv4-2", location);

		std::vector<Bbox> bboxes = make_bbox(score, location, scale);
		// printf("P1: %ld\n", bboxes.size());
		nms(&bboxes, kNMSThresholds[0]);
		ret.insert(ret.end(), bboxes.begin(), bboxes.end());

		// printf("P2: %ld\n", bboxes.size());
	}
	return ret;
}

std::vector<MTCNN::Bbox> MTCNN::run_r_net(const ncnn::Mat &image, std::vector<Bbox> &&prev_bboxes) const {
	std::vector<Bbox> ret;
	for (auto &prev_bbox : prev_bboxes) {
		ncnn::Mat temp_img;
		ncnn::copy_cut_border(image, temp_img, prev_bbox.y1, image.h - prev_bbox.y2, prev_bbox.x1,
		                      image.w - prev_bbox.x2);
		ncnn::Mat in;
		ncnn::resize_bilinear(temp_img, in, 24, 24);
		ncnn::Extractor ex = m_r_net.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score, bbox;
		ex.extract("prob1", score);
		ex.extract("conv5-2", bbox);
		if ((float)score[1] > kThresholds[1]) {
			for (int channel = 0; channel < 4; channel++)
				prev_bbox.regre_coord[channel] = (float)bbox[channel]; //*(bbox.data+channel*bbox.cstep);
			prev_bbox.area = float(prev_bbox.x2 - prev_bbox.x1) * float(prev_bbox.y2 - prev_bbox.y1);
			prev_bbox.score = score.channel(1)[0];                     //*(score.data+score.cstep);
			ret.push_back(prev_bbox);
		}
	}
	return ret;
}

std::vector<MTCNN::Bbox> MTCNN::run_o_net(const ncnn::Mat &image, std::vector<Bbox> &&prev_bboxes) const {
	std::vector<Bbox> ret;
	for (auto &prev_bbox : prev_bboxes) {
		ncnn::Mat temp_img;
		copy_cut_border(image, temp_img, prev_bbox.y1, image.h - prev_bbox.y2, prev_bbox.x1, image.w - prev_bbox.x2);
		ncnn::Mat in;
		resize_bilinear(temp_img, in, 48, 48);
		ncnn::Extractor ex = m_o_net.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score, bbox, keyPoint;
		ex.extract("prob1", score);
		ex.extract("conv6-2", bbox);
		ex.extract("conv6-3", keyPoint);
		if ((float)score[1] > kThresholds[2]) {
			for (int channel = 0; channel < 4; channel++)
				prev_bbox.regre_coord[channel] = (float)bbox[channel];
			prev_bbox.area = float(prev_bbox.x2 - prev_bbox.x1) * float(prev_bbox.y2 - prev_bbox.y1);
			prev_bbox.score = score.channel(1)[0];
			ret.push_back(prev_bbox);
		}
	}
	return ret;
}

} // namespace aibum