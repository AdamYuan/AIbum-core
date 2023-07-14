#include <FaceNet.hpp>

#include <cmath>
#include <mobilefacenet.id.h>
#include <mobilefacenet.mem.h>

#include "Util.hpp"

namespace aibum {

#ifndef AIBUM_CORE_WASM
void FaceNet::LoadFromFile(const char *model_path) {
	m_net.load_param(mobilefacenet_param_bin);
	m_net.load_model(model_path);
}
#endif
void FaceNet::LoadFromMemory(const unsigned char *model_bin) {
	m_net.load_param(mobilefacenet_param_bin);
	m_net.load_model(model_bin);
}

FaceFeature FaceNet::GetFeature(const Image &image) const {
	ncnn::Mat in = Image2Mat<112, 112, ncnn::Mat::PIXEL_RGB>(image);
	return get_feature_rgb_112_112(in);
}

inline static std::tuple<std::array<float, 6>, int> get_alignment_matrix(const aibum::FaceBox &box, float scale) {
	auto [nose_x, nose_y] = box.landmarks[2];
	auto [left_eye_x, left_eye_y] = box.landmarks[0];
	auto [right_eye_x, right_eye_y] = box.landmarks[1];
	float eye_dx = right_eye_x - left_eye_x, eye_dy = right_eye_y - left_eye_y;
	float angle = std::atan2(eye_dy, eye_dx);
	float alpha = std::cos(angle), beta = std::sin(angle);
	float w = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy) * scale;
	std::array<float, 6> mat = {alpha, beta,  -alpha * nose_x - beta * nose_y + w * 0.5f,
	                            -beta, alpha, beta * nose_x - alpha * nose_y + w * 0.5f};
	std::array<float, 6> inv_mat;
	ncnn::invert_affine_transform(mat.data(), inv_mat.data());
	return {inv_mat, (int)w};
}

std::vector<Face> FaceNet::GetFaces(const SCRFD &scrfd, const Image &image) const {

	std::vector<FaceBox> face_boxes = scrfd.Detect(image);
	std::vector<Face> faces;
	faces.reserve(face_boxes.size());

	ncnn::Mat mat = Image2Mat<ncnn::Mat::PIXEL_RGB>(image, image.width, image.height);

	for (auto face_box : face_boxes) {
		auto [matrix, w] = get_alignment_matrix(face_box, 2.5f);
		ncnn::Mat in;

		switch (image.pixel_type) {
		case ncnn::Mat::PIXEL_GRAY: {
			std::vector<unsigned char> face_data(w * w);
			ncnn::warpaffine_bilinear_c1(image.data, image.width, image.height, face_data.data(), w, w, matrix.data());
			in = Image2Mat<ncnn::Mat::PIXEL_RGB>({face_data.data(), w, w, image.pixel_type}, 112, 112);
			break;
		}
		case ncnn::Mat::PIXEL_RGB:
		case ncnn::Mat::PIXEL_BGR: {
			std::vector<unsigned char> face_data(w * w * 3);
			ncnn::warpaffine_bilinear_c3(image.data, image.width, image.height, face_data.data(), w, w, matrix.data());
			in = Image2Mat<ncnn::Mat::PIXEL_RGB>({face_data.data(), w, w, image.pixel_type}, 112, 112);
			break;
		}
		default: {
			std::vector<unsigned char> face_data(w * w * 4);
			ncnn::warpaffine_bilinear_c4(image.data, image.width, image.height, face_data.data(), w, w, matrix.data());
			in = Image2Mat<ncnn::Mat::PIXEL_RGB>({face_data.data(), w, w, image.pixel_type}, 112, 112);
			break;
		}
		}

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