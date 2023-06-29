#include <FaceNet.hpp>

#include <mobilefacenet.id.h>
#include <mobilefacenet.mem.h>

namespace aibum {

FaceNet::FaceNet() {
	m_net.load_param(mobilefacenet_param_bin);
	m_net.load_model(mobilefacenet_bin);
}

std::array<float, 128> FaceNet::GetFeature(const cv::Mat &image) {
	ncnn::Mat ncnn_img =
	    ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 112, 112);

	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_light_mode(true);
	ex.input(mobilefacenet_param_id::BLOB_data, ncnn_img);
	ncnn::Mat out;
	ex.extract(mobilefacenet_param_id::BLOB_fc1, out);

	std::array<float, 128> ret{};
	for (int i = 0; i < 128; i++)
		ret[i] = out[i];

	// normalize
	float l2 = 0.0;
	{ // Kahan Sum
		float c = 0.0f;
		for (float f : ret) {
			f *= f;
			float y = f - c;
			float t = l2 + y;
			c = (t - l2) - y;
			l2 = t;
		}
		l2 = std::sqrt(l2);
	}
	float inv_l2 = 1.0f / l2;

	for (float &f : ret)
		f *= inv_l2;

	return ret;
}

} // namespace aibum