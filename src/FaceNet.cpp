#include <FaceNet.hpp>

namespace aibum {

FaceNet::FaceNet(const char *param, const char *model) {
	m_net.load_param(param);
	m_net.load_model(model);
}

std::array<float, 128> FaceNet::GetFeature(const cv::Mat &image) {
	ncnn::Mat ncnn_img =
	    ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 112, 112);

	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_light_mode(true);
	ex.input("data", ncnn_img);
	ncnn::Mat out;
	ex.extract("fc1", out);

	std::array<float, 128> ret{};
	for (int i = 0; i < 128; i++)
		ret[i] = out[i];
	return ret;
}

} // namespace aibum