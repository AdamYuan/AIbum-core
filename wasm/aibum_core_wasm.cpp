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
	inline void loadFromMemory(const unsigned char *buffer, size_t size) {}
	inline void loadFromFile(const char *filename) {}
	inline void free() { stbi_image_free(m_data); }

	inline Image GetImage() const { return {m_data, m_width, m_height, ncnn::Mat::PIXEL_RGB}; }
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
	    .function("free", &WASMImage::free);

	e::class_<WASMImageNet>("ImageNet")
	    .constructor()
	    .function("getTags", &WASMImageNet::getTags)
	    .function("free", &WASMImageNet::free);
}