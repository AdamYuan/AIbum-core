#include <FaceNet.hpp>

#include <mobilefacenet.id.h>
#include <mobilefacenet.mem.h>
#include <cmath>

#include "Util.hpp"

namespace aibum {

FaceNet::FaceNet() {
	m_net.load_param(mobilefacenet_param_bin);
	m_net.load_model(mobilefacenet_bin);
}

FaceFeature FaceNet::GetFeature(const Image &image) const {
	ncnn::Mat in = Image2Mat<112, 112, ncnn::Mat::PIXEL_RGB>(image);
	return get_feature_rgb_112_112(in);
}

std::vector<Face> FaceNet::GetFaces(const SCRFD &scrfd, const Image &image) const {

	std::vector<FaceBox> face_boxes = scrfd.Detect(image);
	std::vector<Face> faces;
	faces.reserve(face_boxes.size());

	ncnn::Mat mat = Image2Mat<ncnn::Mat::PIXEL_RGB>(image, image.width, image.height);

	for (auto face_box : face_boxes) {
		ncnn::Mat part, in;
		ncnn::copy_cut_border(mat, part, face_box.y, image.height - face_box.y - face_box.h, face_box.x,
		                      image.width - face_box.x - face_box.w);
		ncnn::resize_bilinear(part, in, 112, 112);

		faces.push_back({face_box.x, face_box.y, face_box.w, face_box.h, get_feature_rgb_112_112(in)});
	}
	return faces;
}

FaceFeature FaceNet::get_feature_rgb_112_112(const ncnn::Mat &image) const {
	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_light_mode(true);
	ex.input(mobilefacenet_param_id::BLOB_data, image);
	ncnn::Mat out;
	ex.extract(mobilefacenet_param_id::BLOB_fc1, out);

	FaceFeature feature{};
	for (int i = 0; i < 128; i++)
		feature[i] = out[i];

	// normalize
	float l2 = 0.0;
	{ // Kahan Sum
		float c = 0.0f;
		for (float f : feature) {
			f *= f;
			float y = f - c;
			float t = l2 + y;
			c = (t - l2) - y;
			l2 = t;
		}
		l2 = std::sqrt(l2);
	}
	float inv_l2 = 1.0f / l2;

	for (float &f : feature)
		f *= inv_l2;

	return feature;
}

} // namespace aibum