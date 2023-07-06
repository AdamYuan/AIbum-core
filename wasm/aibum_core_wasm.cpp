#include <emscripten/bind.h>

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

#ifdef AIBUM_WASM_THREADS
#include "worker.hpp"
static ThreadPool worker(1);
#endif

namespace e = emscripten;

template <typename Func> inline std::invoke_result_t<Func> run_func(Func &&func) {
#ifdef AIBUM_WASM_THREADS
	auto future = worker.enqueue(std::forward<Func>(func));
	return future.get();
#else
	return func();
#endif
}

template <typename T, typename Allocator = std::allocator<T>>
inline void js_array_to_vec(const e::val &v, std::vector<T, Allocator> *p_vec) {
	const auto l = v["length"].as<unsigned>();
	p_vec->resize(l);
	e::val mem_view{e::typed_memory_view(l, p_vec->data())};
	mem_view.call<void>("set", v);
}

class WASMImage {
private:
	unsigned char *m_data{};
	int m_width{}, m_height{};

public:
	inline WASMImage() = default;
	inline explicit WASMImage(const ncnn::Mat &ncnn_rgb_image) : m_width{ncnn_rgb_image.w}, m_height{ncnn_rgb_image.h} {
		m_data = (unsigned char *)stbi__malloc_mad3(m_width, m_height, 4, 0);
		ncnn_rgb_image.to_pixels(m_data, ncnn::Mat::PIXEL_RGB2RGBA);
	}
	inline explicit WASMImage(const e::val &u8_array) { load(u8_array); }
	inline ~WASMImage() { free(); }
	inline void free() {
		if (m_data) {
			stbi_image_free(m_data);
			m_data = nullptr;
		}
	}

	inline void load(const e::val &u8_array) {
		if (m_data) {
			stbi_image_free(m_data);
			m_data = nullptr;
		}
		std::vector<uint8_t> u8_vec;
		js_array_to_vec<uint8_t>(u8_array, &u8_vec);
		int comp;
		m_data = stbi_load_from_memory(u8_vec.data(), (int)u8_vec.size(), &m_width, &m_height, &comp, 4);
	}
	inline bool valid() const { return m_data; }

	inline int getWidth() const { return m_width; }
	inline int getHeight() const { return m_height; }
	inline e::val getData() const { return e::val(e::typed_memory_view(m_width * m_height * 4, m_data)); }

	inline aibum::Image GetImage() const { return {m_data, m_width, m_height, ncnn::Mat::PIXEL_RGBA}; }
};

class WASMImageNet {
private:
	aibum::ImageNet m_object;

public:
	inline WASMImageNet() : m_object() {}
	inline e::val getTags(const WASMImage &image, int count) const {
		if (!image.valid())
			return e::val::array();
		auto tags = run_func([this, &image, count]() { return m_object.GetTags(image.GetImage(), count); });
		return e::val::array(tags.begin(), tags.end());
	}
	inline void free() { m_object.Clear(); }
};

struct WASMFace {
	int x, y, w, h;
	e::val feature;

	inline WASMFace() : x{}, y{}, w{}, h{}, feature(e::val::null()) {}
	inline WASMFace(const aibum::Face &face)
	    : x{face.x}, y{face.y}, w{face.w}, h{face.h}, feature{e::val::array(face.feature.begin(), face.feature.end())} {
	}
};

class WASMFaceNet {
private:
	aibum::SCRFD m_detector;
	aibum::FaceNet m_face_net;

public:
	inline WASMFaceNet() : m_detector(), m_face_net() {}
	inline e::val getFaces(const WASMImage &image) const {
		if (!image.valid())
			return e::val::array();
		auto faces = run_func([this, &image]() { return m_face_net.GetFaces(m_detector, image.GetImage()); });
		std::vector<WASMFace> wasm_faces;
		wasm_faces.reserve(faces.size());
		for (const auto &face : faces)
			wasm_faces.emplace_back(face);
		return e::val::array(wasm_faces.begin(), wasm_faces.end());
	}
	inline void free() {
		m_face_net.Clear();
		m_detector.Clear();
	}
};

class WASMStyleTransfer {
private:
	std::vector<uint8_t, AlignedAllocator<uint8_t, 4>> m_model_bin;
	aibum::StyleTransfer m_style_transfer;

public:
	inline WASMStyleTransfer() = default;
	inline explicit WASMStyleTransfer(const e::val &val) { load(val); }
	inline void load(const e::val &val) {
		m_style_transfer.Clear();
		js_array_to_vec<uint8_t, AlignedAllocator<uint8_t, 4>>(val, &m_model_bin);
		m_style_transfer.LoadFromMemory(m_model_bin.data());
	}
	inline bool valid() const { return !m_model_bin.empty(); }
	inline WASMImage transfer(const WASMImage &image, int target_size) const {
		int target_w, target_h;
		if (image.getWidth() < image.getHeight()) {
			target_h = target_size;
			target_w = target_size * image.getWidth() / image.getHeight();
		} else {
			target_w = target_size;
			target_h = target_size * image.getHeight() / image.getWidth();
		}
		auto result = run_func([this, &image, target_w, target_h]() {
			return m_style_transfer.Transfer(image.GetImage(), target_w, target_h);
		});
		return WASMImage{result};
	}
	inline void free() {
		m_style_transfer.Clear();
		m_model_bin.clear();
		m_model_bin.shrink_to_fit();
	}
};

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
		.function("free", &WASMImage::free)
	    .property("width", &WASMImage::getWidth)
	    .property("height", &WASMImage::getHeight)
	    .property("data", &WASMImage::getData)
	    .property("valid", &WASMImage::valid);

	e::class_<WASMImageNet>("ImageNet")
	    .constructor()
	    .function("getTags", &WASMImageNet::getTags)
	    .function("free", &WASMImageNet::free);

	e::class_<WASMFaceNet>("FaceNet")
	    .constructor()
	    .function("getFaces", &WASMFaceNet::getFaces)
	    .function("free", &WASMFaceNet::free);

	e::class_<WASMStyleTransfer>("StyleTransfer")
	    .constructor()
	    .constructor<const e::val &>()
	    .property("valid", &WASMStyleTransfer::valid)
	    .function("load", &WASMStyleTransfer::load)
	    .function("transfer", &WASMStyleTransfer::transfer)
	    .function("free", &WASMStyleTransfer::free);
}
