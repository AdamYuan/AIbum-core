#include <ImageNet.hpp>

#include <efficientnet_b0_fp16.bin.h>
#include <efficientnet_b0_fp16.id.h>

#include "Util.hpp"

namespace aibum {

ImageNet::ImageNet() {
	m_net.load_param(efficientnet_b0_fp16_param_bin);
	m_net.load_model(efficientnet_b0_fp16_bin);
}

std::vector<Tag> ImageNet::GetTags(const Image &image, int count) const {
	ncnn::Mat in = Image2Mat<224, 224, ncnn::Mat::PIXEL_RGB>(image);

	constexpr float kMeanValues[3] = {123.675f, 116.28f, 103.53f};
	constexpr float kNormValues[3] = {0.01712475383, 0.0175070028, 0.0174291939};
	in.substract_mean_normalize(kMeanValues, kNormValues);

	ncnn::Mat out;
	ncnn::Extractor ex = m_net.create_extractor();
	ex.input(efficientnet_b0_fp16_param_id::BLOB_input0, in);
	ex.extract(efficientnet_b0_fp16_param_id::BLOB_output0, out);
	std::vector<float> cls_scores(out.w);
	for (int j = 0; j < out.w; j++)
		cls_scores[j] = out[j];

	std::vector<Tag> ret(out.w);
	for (int i = 0; i < out.w; ++i)
		ret[i] = {i, out[i]};

	const auto tag_score_cmp = [](const Tag &l, const Tag &r) { return l.score > r.score; };

	if (count > 1) {
		std::nth_element(ret.begin(), ret.begin() + count, ret.end(), tag_score_cmp);
		if (ret.size() > count)
			ret.resize(count);
		std::sort(ret.begin(), ret.end(), tag_score_cmp);

		return ret;
	} else if (count == 1)
		return std::vector<Tag>{*std::min_element(ret.begin(), ret.end(), tag_score_cmp)};
	return {};
}

} // namespace aibum