#pragma once

#include <FaceNet.hpp>
#include <SCRFD.hpp>

namespace aibum {

struct Face {
	int x, y, size;
	std::array<float, 128> feature;
};

class SCRFDFaceNet {
private:
	SCRFD m_mtcnn;
	FaceNet m_face_net;

public:
	inline SCRFDFaceNet() : m_mtcnn(), m_face_net() {}
	inline std::vector<Face> GetFaces(const cv::Mat &image) {
		std::vector<FaceBox> face_boxes = m_mtcnn.Detect(image);
		std::vector<Face> faces;
		faces.reserve(face_boxes.size());
		for (auto face_box : face_boxes) {
			cv::Rect rect{face_box.x, face_box.y, face_box.size, face_box.size};
			faces.push_back({face_box.x, face_box.y, face_box.size, m_face_net.GetFeature(image(rect).clone())});
		}
		return faces;
	}
};

} // namespace aibum
