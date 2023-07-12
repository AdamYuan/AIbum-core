#include <emscripten/bind.h>
#include <emscripten/fetch.h>
#include <emscripten/threading.h>
#include <emscripten/wasm_worker.h>

#include "aligned_allocator.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_STDIO
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#ifdef AIBUM_WASM_SIMD
#define STBI__X86_TARGET
#endif
#include <stb_image.h>

#include <FaceNet.hpp>
#include <ImageNet.hpp>
#include <StyleTransfer.hpp>

namespace e = emscripten;

template <typename T, typename Allocator = std::allocator<T>>
inline void js_array_to_vec(const e::val &v, std::vector<T, Allocator> *p_vec) {
	const auto l = v["length"].as<unsigned>();
	p_vec->resize(l);
	e::val mem_view{e::typed_memory_view(l, p_vec->data())};
	mem_view.call<void>("set", v);
}

struct WASMFace {
	int x, y, w, h;
	e::val feature;

	inline WASMFace() : x{}, y{}, w{}, h{}, feature(e::val::null()) {}
	inline explicit WASMFace(const aibum::Face &face)
	    : x{face.x}, y{face.y}, w{face.w}, h{face.h}, feature{e::val::array(face.feature.begin(), face.feature.end())} {
	}
};

class WASMImage {
private:
	std::vector<unsigned char> m_data;
	int m_width{}, m_height{};

public:
	inline WASMImage() = default;
	inline explicit WASMImage(const ncnn::Mat &ncnn_rgb_image) : m_width{ncnn_rgb_image.w}, m_height{ncnn_rgb_image.h} {
		m_data.resize(m_width * m_height * 4);
		ncnn_rgb_image.to_pixels(m_data.data(), ncnn::Mat::PIXEL_RGB2RGBA);
	}
	inline explicit WASMImage(const e::val &u8_array) { load(u8_array); }

	inline void load(const e::val &u8_array) {
		std::vector<uint8_t> u8_vec;
		js_array_to_vec<uint8_t>(u8_array, &u8_vec);
		int comp;
		auto data = stbi_load_from_memory(u8_vec.data(), (int)u8_vec.size(), &m_width, &m_height, &comp, 4);
		if (data == nullptr) {
			m_data.clear();
			return;
		}
		m_data.resize(m_width * m_height * 4);
		std::copy(data, data + m_data.size(), m_data.data());
	}
	inline bool valid() const { return !m_data.empty(); }

	inline int getWidth() const { return m_width; }
	inline int getHeight() const { return m_height; }
	inline e::val getData() const { return e::val(e::typed_memory_view(m_width * m_height * 4, m_data.data())); }

	inline aibum::Image GetImage() const { return {m_data.data(), m_width, m_height, ncnn::Mat::PIXEL_RGBA}; }
};

class WASM4AFetcher {
private:
	std::string m_uri;
	std::vector<uint8_t, AlignedAllocator<uint8_t, 4>> m_bin;

public:
	template <typename Model> inline bool Fetch(Model *p_model, const e::val &fetcher, const std::string &uri) {
		if (m_uri == uri)
			return true;

		const e::val u8_array = fetcher(uri).await();
		if (u8_array.isUndefined() || u8_array.isNull())
			return false;

		js_array_to_vec<uint8_t, AlignedAllocator<uint8_t, 4>>(u8_array, &m_bin);
		m_uri = uri;
		p_model->Clear();
		p_model->LoadFromMemory(m_bin.data());
		return true;
	}
};

class WASMImageNet {
private:
	aibum::ImageNet m_image_net;
	WASM4AFetcher m_fetcher;

public:
	inline WASMImageNet() : m_image_net() {}
	inline bool load(const e::val &fetcher, const std::string &uri) {
		return m_fetcher.Fetch(&m_image_net, fetcher, uri);
	}
	inline e::val getTags(const WASMImage &image, int count) const {
		if (!image.valid())
			return e::val::array();
		auto tags = m_image_net.GetTags(image.GetImage(), count);
		return e::val::array(tags.begin(), tags.end());
	}
};

class WASMFaceNet {
private:
	aibum::SCRFD m_detector;
	aibum::FaceNet m_face_net;
	WASM4AFetcher m_scrfd_fetcher, m_facenet_fetcher;

public:
	inline WASMFaceNet() : m_detector(), m_face_net() {}
	inline bool load(const e::val &fetcher, const std::string &scrfd_uri, const std::string &facenet_uri) {
		bool scrfd_success = m_scrfd_fetcher.Fetch(&m_detector, fetcher, scrfd_uri);
		bool facenet_success = m_facenet_fetcher.Fetch(&m_face_net, fetcher, facenet_uri);
		return scrfd_success && facenet_success;
	}
	inline e::val getFaces(const WASMImage &image) const {
		if (!image.valid())
			return e::val::array();
		auto faces = m_face_net.GetFaces(m_detector, image.GetImage());
		std::vector<WASMFace> wasm_faces;
		wasm_faces.reserve(faces.size());
		for (const auto &face : faces)
			wasm_faces.emplace_back(face);
		return e::val::array(wasm_faces.begin(), wasm_faces.end());
	}
};

class WASMStyleTransfer {
private:
	aibum::StyleTransfer m_style_transfer;
	WASM4AFetcher m_fetcher;

public:
	inline WASMStyleTransfer() = default;
	inline bool load(const e::val &fetcher, const std::string &uri) {
		return m_fetcher.Fetch(&m_style_transfer, fetcher, uri);
	}
	inline WASMImage transfer(const WASMImage &image, int target_size) const {
		int target_w, target_h;
		if (image.getWidth() < image.getHeight()) {
			target_h = target_size;
			target_w = target_size * image.getWidth() / image.getHeight();
		} else {
			target_w = target_size;
			target_h = target_size * image.getHeight() / image.getWidth();
		}
		auto result = m_style_transfer.Transfer(image.GetImage(), target_w, target_h);
		return WASMImage{result};
	}
};

static WASMImageNet image_net;
static WASMFaceNet face_net;
static WASMStyleTransfer style_transfer;

EMSCRIPTEN_BINDINGS(AIbumCore) {
	e::value_object<WASMFace>("Face")
	    .field("x", &WASMFace::x)
	    .field("y", &WASMFace::y)
	    .field("w", &WASMFace::w)
	    .field("h", &WASMFace::h)
	    .field("feature", &WASMFace::feature);
	e::value_object<aibum::Tag>("Tag").field("index", &aibum::Tag::index).field("score", &aibum::Tag::score);

	e::class_<WASMImage>("Image")
	    .constructor()
	    .constructor<const e::val &>()
	    .function("load", &WASMImage::load)
	    .property("width", &WASMImage::getWidth)
	    .property("height", &WASMImage::getHeight)
	    .property("data", &WASMImage::getData)
	    .property("valid", &WASMImage::valid);

	e::function(
	    "getTags", +[](const WASMImage &image, int count) { return image_net.getTags(image, count); });
	e::function(
	    "getFaces", +[](const WASMImage &image) { return face_net.getFaces(image); });
	e::function(
	    "transferStyle",
	    +[](const WASMImage &image, int target_size) { return style_transfer.transfer(image, target_size); });
	e::function(
	    "loadImageNet", +[](const e::val &fetcher, const std::string &uri) { image_net.load(fetcher, uri); });
	e::function(
	    "loadFaceNet", +[](const e::val &fetcher, const std::string &scrfd_uri, const std::string &facenet_uri) {
		    face_net.load(fetcher, scrfd_uri, facenet_uri);
	    });
	e::function(
	    "loadStyleTransfer", +[](const e::val &fetcher, const std::string &uri) { style_transfer.load(fetcher, uri); });
}
