#include <StyleTransfer.hpp>

#include "Util.hpp"
#include <styletransfer.id.h>
#include <styletransfer.param.bin.h>

namespace aibum {

StyleTransfer::StyleTransfer(const char *model_path) {
	m_net.opt.lightmode = true;
	m_net.load_param(styletransfer_param_bin);
	m_net.load_model(model_path);
}

StyleTransfer::StyleTransfer(const unsigned char *model_bin) {
	m_net.opt.lightmode = true;
	m_net.load_param(styletransfer_param_bin);
	m_net.load_model(model_bin);
}

ncnn::Mat StyleTransfer::Transfer(const Image &image) {
	const int target_size = 256;
	int target_w, target_h;
	if (image.width < image.height) {
		target_h = target_size;
		target_w = target_size * image.width / image.height;
	} else {
		target_w = target_size;
		target_h = target_size * image.height / image.width;
	}
	ncnn::Mat in = Image2Mat<ncnn::Mat::PIXEL_RGB>(image, target_w, target_h);
	ncnn::Mat out;
	{
		ncnn::Extractor ex = m_net.create_extractor();
		ex.input(styletransfer_param_id::BLOB_input1, in);
		ex.extract(styletransfer_param_id::BLOB_output1, out);
	}
	return out;
}

} // namespace aibum
