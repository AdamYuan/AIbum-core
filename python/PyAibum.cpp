#include <ImageNet.hpp>
#include <MTCNNFaceNet.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct PyImageNet final : public aibum::ImageNet {
	using aibum::ImageNet::ImageNet;
	inline py::array_t<aibum::Tag> getTags(py::array_t<uint8_t> &image, unsigned int count) {
		py::buffer_info buf = image.request();
		cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);
		auto tags = GetTags(mat, count);

		py::array_t<aibum::Tag> ret((ssize_t)tags.size());
		std::copy(tags.begin(), tags.end(), ret.mutable_data());
		return ret;
	}
};

struct PyMTCNNFaceNet final : public aibum::MTCNNFaceNet {
	using aibum::MTCNNFaceNet::MTCNNFaceNet;
	inline py::array_t<aibum::Face> getFaces(py::array_t<uint8_t> &image) {
		py::buffer_info buf = image.request();
		cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

		std::vector<aibum::Face> faces = GetFaces(mat);

		py::array_t<aibum::Face> ret((ssize_t)faces.size());
		std::copy(faces.begin(), faces.end(), ret.mutable_data());
		return ret;
	}
};

PYBIND11_MODULE(pyaibum, m) {
	PYBIND11_NUMPY_DTYPE(aibum::Tag, index, score);
	py::class_<PyImageNet>(m, "ImageNet")
	    .def(py::init<const char *>())
	    .def("getTags", &PyImageNet::getTags, py::return_value_policy::move);

	PYBIND11_NUMPY_DTYPE(aibum::Face, x, y, size, feature);
	py::class_<PyMTCNNFaceNet>(m, "MTCNNFaceNet")
	    .def(py::init<const char *>())
	    .def("getFaces", &PyMTCNNFaceNet::getFaces, py::return_value_policy::move);
}
