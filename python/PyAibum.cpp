#include <FaceNet.hpp>
#include <ImageNet.hpp>
#include <MTCNN.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Tag = aibum::ImageTag;

struct PyImageNet final : public aibum::ImageNet {
	using ImageNet::ImageNet;
	inline py::array_t<Tag> getTags(py::array_t<uint8_t> &image, unsigned int max_count) {
		py::buffer_info buf = image.request();
		cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);
		auto tags = GetTags(mat, max_count);

		py::array_t<Tag> ret((ssize_t)tags.size());
		std::copy(tags.begin(), tags.end(), ret.mutable_data());
		return ret;
	}
};

struct Face {
	int x, y, size;
	std::array<float, 128> feature;
};

struct PyMTCNNFaceNet final : public aibum::MTCNN, public aibum::FaceNet {
	inline explicit PyMTCNNFaceNet(const char *model_dir) : aibum::MTCNN(model_dir), aibum::FaceNet(model_dir) {}
	inline py::array_t<Face> getFaces(py::array_t<uint8_t> &image) {
		py::buffer_info buf = image.request();
		cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

		std::vector<aibum::FaceBox> face_boxes = Detect(mat);

		std::vector<Face> faces;
		faces.reserve(face_boxes.size());

		for (auto face_box : face_boxes) {
			cv::Rect rect{face_box.x, face_box.y, face_box.size, face_box.size};
			faces.push_back({face_box.x, face_box.y, face_box.size, GetFeature(mat(rect))});
		}

		py::array_t<Face> ret((ssize_t)faces.size());
		std::copy(faces.begin(), faces.end(), ret.mutable_data());
		return ret;
	}
};

PYBIND11_MODULE(pyaibum, m) {
	PYBIND11_NUMPY_DTYPE(Tag, index, score);
	py::class_<PyImageNet>(m, "ImageNet")
	    .def(py::init<const char *>())
	    .def("getTags", &PyImageNet::getTags, py::return_value_policy::move);

	PYBIND11_NUMPY_DTYPE(Face, x, y, size, feature);
	py::class_<PyMTCNNFaceNet>(m, "MTCNNFaceNet")
	    .def(py::init<const char *>())
	    .def("getFaces", &PyMTCNNFaceNet::getFaces, py::return_value_policy::move);
}
