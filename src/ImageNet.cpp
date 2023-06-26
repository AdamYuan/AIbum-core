#include <ImageNet.hpp>

namespace aibum {

ImageNet::ImageNet(const char *model_dir) {
	m_net.load_param((std::string{model_dir} + "mobilenet_v3.param").c_str());
	m_net.load_model((std::string{model_dir} + "mobilenet_v3.bin").c_str());
}

std::vector<Tag> ImageNet::GetTags(const cv::Mat &image, unsigned int max_count) {
	ncnn::Mat in =
	    ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 224, 224);
	constexpr float kMeanValues[3] = {0.485f / 255.f, 0.456f / 255.f, 0.406f / 255.f};
	constexpr float kNormValues[3] = {1.0 / 0.229f / 255.f, 1.0 / 0.224f / 255.f, 1.0 / 0.225f / 255.f};
	in.substract_mean_normalize(kMeanValues, kNormValues);

	ncnn::Mat out;
	ncnn::Extractor ex = m_net.create_extractor();
	ex.input("in0", in);
	ex.extract("out0", out);
	std::vector<float> cls_scores(out.w);
	for (int j = 0; j < out.w; j++)
		cls_scores[j] = out[j];

	std::vector<Tag> ret(out.w);
	for (int i = 0; i < out.w; ++i)
		ret[i] = {i, out[i]};
	std::sort(ret.begin(), ret.end(), [](const Tag &l, const Tag &r) { return l.score > r.score; });

	if (ret.size() > max_count)
		ret.resize(max_count);

	return ret;
}

} // namespace aibum