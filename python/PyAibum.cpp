#include <ImageNet.hpp>
#include <SCRFDFaceNet.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyImage {
private:
	cv::Mat m_mat;

public:
	inline explicit PyImage(const char *filename) {
		m_mat = cv::imread(filename, cv::IMREAD_COLOR);
	}

	inline explicit PyImage(const py::array_t<uint8_t> &image) {
		py::buffer_info buf = image.request();
		m_mat = cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *) buf.ptr).clone();
	}

	inline const cv::Mat &GetMat() const { return m_mat; }
};

struct PyImageNet final : public aibum::ImageNet {
	using aibum::ImageNet::ImageNet;

	inline py::array_t<aibum::Tag> getTags(const PyImage &image, unsigned int count) {
		auto tags = GetTags(image.GetMat(), count);

		py::array_t<aibum::Tag> ret((ssize_t) tags.size());
		std::copy(tags.begin(), tags.end(), ret.mutable_data());
		return ret;
	}
};

struct PySCRFDFaceNet final : public aibum::SCRFDFaceNet {
	using aibum::SCRFDFaceNet::SCRFDFaceNet;

	inline py::array_t<aibum::Face> getFaces(const PyImage &image) {
		std::vector<aibum::Face> faces = GetFaces(image.GetMat());

		py::array_t<aibum::Face> ret((ssize_t) faces.size());
		std::copy(faces.begin(), faces.end(), ret.mutable_data());
		return ret;
	}
};

PYBIND11_MODULE(pyaibum_core, m) {
	py::class_<PyImage>(m, "Image")
			.def(py::init<const char *>())
			.def(py::init<const py::array_t<uint8_t> &>());

	PYBIND11_NUMPY_DTYPE(aibum::Tag, index, score);
	py::class_<PyImageNet>(m, "ImageNet")
			.def(py::init<>())
			.def("getTags", &PyImageNet::getTags, py::return_value_policy::move);

	PYBIND11_NUMPY_DTYPE(aibum::Face, x, y, size, feature);
	py::class_<PySCRFDFaceNet>(m, "SCRFDFaceNet")
			.def(py::init<>())
			.def("getFaces", &PySCRFDFaceNet::getFaces, py::return_value_policy::move);
}
