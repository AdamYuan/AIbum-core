#include <emscripten/bind.h>
#include <emscripten/html5.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <string>

#include <FaceNet.hpp>
#include <ImageNet.hpp>

namespace e = emscripten;

class WASMImage {
private:
	unsigned char *m_data{};
	int m_width{}, m_height{};

public:
	inline WASMImage() = default;
	inline explicit WASMImage(const std::string &filename) { loadFromFile(filename); }
	inline WASMImage(uintptr_t buffer_ptr, size_t size) { loadFromMemory(buffer_ptr, size); }
	inline ~WASMImage() {
		if (m_data)
			stbi_image_free(m_data);
	}

	inline void loadFromMemory(uintptr_t buffer_ptr, size_t size) {
		if (m_data) {
			stbi_image_free(m_data);
			m_data = nullptr;
		}
		int comp;
		m_data = stbi_load_from_memory((const unsigned char *)buffer_ptr, (int)size, &m_width, &m_height, &comp, 3);
	}
	inline void loadFromFile(const std::string &filename) {
		if (m_data) {
			stbi_image_free(m_data);
			m_data = nullptr;
		}
		int comp;
		m_data = stbi_load(filename.c_str(), &m_width, &m_height, &comp, 3);
	}
	inline bool valid() const { return m_data; }

	inline aibum::Image GetImage() const { return {m_data, m_width, m_height, ncnn::Mat::PIXEL_RGB}; }
};

class WASMImageNet {
private:
	aibum::ImageNet m_object;

public:
	inline WASMImageNet() : m_object() {}
	inline std::vector<aibum::Tag> getTags(const WASMImage &image, int count) const {
		if (!image.valid())
			return {};
		return m_object.GetTags(image.GetImage(), count);
	}
};

class WASMFaceNet {
private:
	aibum::SCRFD m_detector;
	aibum::FaceNet m_face_net;

public:
	inline WASMFaceNet() : m_detector(), m_face_net() {}
	inline std::vector<aibum::Face> getFaces(const WASMImage &image) const {
		if (!image.valid())
			return {};
		return m_face_net.GetFaces(m_detector, image.GetImage());
	}
};

template <typename T> e::class_<std::vector<T>> my_register_vector(const char *name) {
	typedef std::vector<T> VecType;

	size_t (VecType::*size)() const = &VecType::size;

	const auto toArray = +[](const VecType &v) -> e::val { return e::val::array(v.begin(), v.end()); };
	const auto get = +[](const VecType &v, typename VecType ::size_type index) -> e::val {
		return index < v.size() ? e::val(v[index]) : e::val::undefined();
	};

	return e::class_<VecType>(name)
	    .template constructor<>()
	    .function("size", size)
	    .function("toArray", toArray)
	    .function("toObject", toArray)
	    .function("get", get);
}

template <typename T, std::size_t L> e::class_<std::array<T, L>> my_register_array(const char *name) {
	typedef std::array<T, L> ArrType;

	size_t (ArrType::*size)() const = &ArrType::size;

	const auto toArray = +[](const ArrType &v) -> e::val { return e::val::array(v.begin(), v.end()); };
	const auto get = +[](const ArrType &v, typename ArrType ::size_type index) -> e::val {
		return index < v.size() ? e::val(v[index]) : e::val::undefined();
	};

	return e::class_<ArrType>(name)
	    .template constructor<>()
	    .function("size", size)
	    .function("toArray", toArray)
	    .function("toObject", toArray)
	    .function("get", get);
}

EMSCRIPTEN_BINDINGS(AIbumCore) {
	e::value_object<aibum::Face>("Face")
	    .field("x", &aibum::Face::x)
	    .field("y", &aibum::Face::y)
	    .field("w", &aibum::Face::w)
	    .field("h", &aibum::Face::h)
	    .field("feature", &aibum::Face::feature);
	e::value_object<aibum::Tag>("Tag").field("index", &aibum::Tag::index).field("score", &aibum::Tag::score);

	my_register_array<float, 128>("FaceFeature");

	my_register_vector<aibum::Face>("FaceList");
	my_register_vector<aibum::Tag>("TagList");

	e::class_<WASMImage>("Image")
	    .constructor()
	    .constructor<uintptr_t, size_t>()
	    .constructor<const std::string &>()
	    .function("loadFromFile", &WASMImage::loadFromFile)
	    .function("loadFromMemory", &WASMImage::loadFromMemory, e::allow_raw_pointers())
	    .function("valid", &WASMImage::valid);

	e::class_<WASMImageNet>("ImageNet").constructor().function("getTags", &WASMImageNet::getTags);

	e::class_<WASMFaceNet>("FaceNet").constructor().function("getFaces", &WASMFaceNet::getFaces);
}