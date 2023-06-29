#include <emscripten/bind.h>
#include <emscripten/html5.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <FaceNet.hpp>
#include <ImageNet.hpp>

namespace e = emscripten;

class WASMImage {
private:
	unsigned char *m_data{};
	int m_width{}, m_height{};

public:
	inline WASMImage() = default;
	inline explicit WASMImage(const char *filename) { loadFromFile(filename); }
	inline WASMImage(unsigned char *buffer, size_t size) { loadFromMemory(buffer, size); }
	inline void loadFromMemory(const unsigned char *buffer, size_t size) {
		int comp;
		m_data = stbi_load_from_memory(buffer, (int)size, &m_width, &m_height, &comp, 3);
	}
	inline void loadFromFile(const char *filename) {
		int comp;
		m_data = stbi_load(filename, &m_width, &m_height, &comp, 3);
	}
	inline void free() { stbi_image_free(m_data); }
	inline bool valid() const { return m_data; }

	inline aibum::Image GetImage() const { return {m_data, m_width, m_height, ncnn::Mat::PIXEL_RGB}; }
};

class WASMImageNet {
private:
	aibum::ImageNet m_object;

public:
	inline WASMImageNet() : m_object() {}
	inline std::vector<aibum::Tag> getTags(const WASMImage &image, int count) const {
		return m_object.GetTags(image.GetImage(), count);
	}
	inline void free() { m_object.~ImageNet(); }
};

class WASMFaceNet {
private:
	aibum::SCRFD m_detector;
	aibum::FaceNet m_face_net;

public:
	inline WASMFaceNet() : m_detector(), m_face_net() {}
	inline std::vector<aibum::Face> getFaces(const WASMImage &image) const {
		return m_face_net.GetFaces(m_detector, image.GetImage());
	}
	inline void free() {
		m_detector.~SCRFD();
		m_face_net.~FaceNet();
	}
};

EMSCRIPTEN_BINDINGS(AIbumCore) {
	e::value_object<aibum::Face>("Face")
	    .field("x", &aibum::Face::x)
	    .field("y", &aibum::Face::y)
	    .field("w", &aibum::Face::w)
	    .field("h", &aibum::Face::h)
	    .field("feature", &aibum::Face::feature);
	e::value_object<aibum::Tag>("Tag").field("index", &aibum::Tag::index).field("score", &aibum::Tag::score);

	e::register_vector<aibum::Face>("FaceList");
	e::register_vector<aibum::Tag>("TagList");

	e::class_<WASMImage>("Image")
	    .constructor()
	    .function("loadFromFile", &WASMImage::loadFromFile, e::allow_raw_pointers())
	    .function("loadFromMemory", &WASMImage::loadFromMemory, e::allow_raw_pointers())
	    .function("valid", &WASMImage::valid)
	    .function("free", &WASMImage::free);

	e::class_<WASMImageNet>("ImageNet")
	    .constructor()
	    .function("getTags", &WASMImageNet::getTags)
	    .function("free", &WASMImageNet::free);

	e::class_<WASMFaceNet>("FaceNet")
		.constructor()
		.function("getFaces", &WASMFaceNet::getFaces)
		.function("free", &WASMFaceNet::free);
}