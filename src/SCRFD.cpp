#include <SCRFD.hpp>
#include <algorithm>
#include <cmath>

#include <scrfd.bin.h>
#include <scrfd.id.h>

#include "Util.hpp"

namespace aibum {

#ifndef AIBUM_CORE_WASM
void SCRFD::LoadFromFile(const char *model_path) {
	m_net.load_param(scrfd_param_bin);
	m_net.load_model(model_path);
}
#endif

void SCRFD::LoadFromMemory(const unsigned char *model_bin) {
	m_net.load_param(scrfd_param_bin);
	m_net.load_model(model_bin);
}

std::vector<FaceBox> SCRFD::Detect(const Image &image) const {
	const int target_size = 640;
	const float kProbThreshold = 0.45f;
	const float kNMSThreshold = 0.45f;

	int w = image.width;
	int h = image.height;
	float scale = 1.f;
	if (w > h) {
		scale = (float)target_size / (float)w;
		w = target_size;
		h = int((float)h * scale);
	} else {
		scale = (float)target_size / (float)h;
		h = target_size;
		w = int((float)w * scale);
	}

	ncnn::Mat in = Image2Mat<ncnn::Mat::PIXEL_RGB>(image, w, h);

	// pad to target_size rectangle
	int wpad = (w + 31) / 32 * 32 - w;
	int hpad = (h + 31) / 32 * 32 - h;
	ncnn::Mat in_pad;
	ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
	                       0.f);

	constexpr float kMeanValues[3] = {127.5f, 127.5f, 127.5f};
	constexpr float kNormValues[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
	in_pad.substract_mean_normalize(kMeanValues, kNormValues);

	ncnn::Extractor ex = m_net.create_extractor();

	ex.input(scrfd_param_id::BLOB_input_1, in_pad);

	std::vector<BBox> face_proposals;

	// stride 8
	{
		ncnn::Mat score_blob, bbox_blob, kps_blob;
		ex.extract(scrfd_param_id::BLOB_score_8, score_blob);
		ex.extract(scrfd_param_id::BLOB_bbox_8, bbox_blob);
		ex.extract(scrfd_param_id::BLOB_kps_8, kps_blob);

		const int base_size = 16;
		const int feat_stride = 8;
		ncnn::Mat ratios(1);
		ratios[0] = 1.f;
		ncnn::Mat scales(2);
		scales[0] = 1.f;
		scales[1] = 2.f;
		ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

		std::vector<BBox> faceobjects32 =
		    generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, kProbThreshold);
		face_proposals.insert(face_proposals.end(), faceobjects32.begin(), faceobjects32.end());
	}

	// stride 16
	{
		ncnn::Mat score_blob, bbox_blob, kps_blob;
		ex.extract(scrfd_param_id::BLOB_score_16, score_blob);
		ex.extract(scrfd_param_id::BLOB_bbox_16, bbox_blob);
		ex.extract(scrfd_param_id::BLOB_kps_16, kps_blob);

		const int base_size = 64;
		const int feat_stride = 16;
		ncnn::Mat ratios(1);
		ratios[0] = 1.f;
		ncnn::Mat scales(2);
		scales[0] = 1.f;
		scales[1] = 2.f;
		ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

		std::vector<BBox> faceobjects16 =
		    generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, kProbThreshold);
		face_proposals.insert(face_proposals.end(), faceobjects16.begin(), faceobjects16.end());
	}

	// stride 32
	{
		ncnn::Mat score_blob, bbox_blob, kps_blob;
		ex.extract(scrfd_param_id::BLOB_score_32, score_blob);
		ex.extract(scrfd_param_id::BLOB_bbox_32, bbox_blob);
		ex.extract(scrfd_param_id::BLOB_kps_32, kps_blob);

		const int base_size = 256;
		const int feat_stride = 32;
		ncnn::Mat ratios(1);
		ratios[0] = 1.f;
		ncnn::Mat scales(2);
		scales[0] = 1.f;
		scales[1] = 2.f;
		ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

		std::vector<BBox> faceobjects8 =
		    generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, kProbThreshold);
		face_proposals.insert(face_proposals.end(), faceobjects8.begin(), faceobjects8.end());
	}

	// sort all proposals by score from highest to lowest
	std::sort(face_proposals.begin(), face_proposals.end(),
	          [](const BBox &l, const BBox &r) { return l.prob > r.prob; });

	// apply nms with nms_threshold
	std::vector<int> picked = nms_sorted_bboxes(face_proposals, kNMSThreshold);

	const int face_count = (int)picked.size();
	std::vector<FaceBox> face_boxes(face_count);

	for (int i = 0; i < face_count; i++) {
		const BBox &rect = face_proposals[picked[i]];

		{
			// adjust offset to original unpadded
			float x0 = (rect.x - float(wpad) * 0.5f) / scale;
			float y0 = (rect.y - float(hpad) * 0.5f) / scale;
			float x1 = (rect.x + rect.width - float(wpad) * 0.5f) / scale;
			float y1 = (rect.y + rect.height - float(hpad) * 0.5f) / scale;
			// Bound
			int ix0 = std::max((int)x0, 0);
			int ix1 = std::min((int)x1, image.width);
			int iy0 = std::max((int)y0, 0);
			int iy1 = std::min((int)y1, image.height);

			face_boxes[i].x = ix0;
			face_boxes[i].y = iy0;
			face_boxes[i].w = ix1 - ix0;
			face_boxes[i].h = iy1 - iy0;
		}

		{
			float x0 = (rect.landmarks[0].x - float(wpad * 0.5)) / scale;
			float y0 = (rect.landmarks[0].y - float(hpad * 0.5)) / scale;
			float x1 = (rect.landmarks[1].x - float(wpad * 0.5)) / scale;
			float y1 = (rect.landmarks[1].y - float(hpad * 0.5)) / scale;
			float x2 = (rect.landmarks[2].x - float(wpad * 0.5)) / scale;
			float y2 = (rect.landmarks[2].y - float(hpad * 0.5)) / scale;
			float x3 = (rect.landmarks[3].x - float(wpad * 0.5)) / scale;
			float y3 = (rect.landmarks[3].y - float(hpad * 0.5)) / scale;
			float x4 = (rect.landmarks[4].x - float(wpad * 0.5)) / scale;
			float y4 = (rect.landmarks[4].y - float(hpad * 0.5)) / scale;

			face_boxes[i].landmarks[0].x = std::max(std::min(x0, (float)image.width), 0.f);
			face_boxes[i].landmarks[0].y = std::max(std::min(y0, (float)image.height), 0.f);
			face_boxes[i].landmarks[1].x = std::max(std::min(x1, (float)image.width), 0.f);
			face_boxes[i].landmarks[1].y = std::max(std::min(y1, (float)image.height), 0.f);
			face_boxes[i].landmarks[2].x = std::max(std::min(x2, (float)image.width), 0.f);
			face_boxes[i].landmarks[2].y = std::max(std::min(y2, (float)image.height), 0.f);
			face_boxes[i].landmarks[3].x = std::max(std::min(x3, (float)image.width), 0.f);
			face_boxes[i].landmarks[3].y = std::max(std::min(y3, (float)image.height), 0.f);
			face_boxes[i].landmarks[4].x = std::max(std::min(x4, (float)image.width), 0.f);
			face_boxes[i].landmarks[4].y = std::max(std::min(y4, (float)image.height), 0.f);
		}
	}

	return face_boxes;
}

std::vector<int> SCRFD::nms_sorted_bboxes(const std::vector<BBox> &faceobjects, float nms_threshold) {
	std::vector<int> picked;

	const int n = (int)faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
		areas[i] = faceobjects[i].width * faceobjects[i].height;

	for (int i = 0; i < n; i++) {
		const BBox &a = faceobjects[i];

		bool keep = true;
		for (int j : picked) {
			const BBox &b = faceobjects[j];

			// intersection over union
			float ow = std::max(std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x), 0.0f),
			      oh = std::max(std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y), 0.0f);
			float inter_area = ow * oh;
			float union_area = areas[i] + areas[j] - inter_area;

			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold) {
				keep = false;
				break;
			}
		}

		if (keep)
			picked.push_back(i);
	}
	return picked;
}

ncnn::Mat SCRFD::generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
	int num_ratio = ratios.w;
	int num_scale = scales.w;

	ncnn::Mat anchors;
	anchors.create(4, num_ratio * num_scale);

	const float cx = 0;
	const float cy = 0;

	for (int i = 0; i < num_ratio; i++) {
		float ar = ratios[i];

		int r_w = (int)std::roundl((float)base_size / std::sqrt(ar));
		int r_h = (int)std::roundl((float)r_w * ar); // round(base_size * sqrt(ar));

		for (int j = 0; j < num_scale; j++) {
			float scale = scales[j];

			float rs_w = (float)r_w * scale;
			float rs_h = (float)r_h * scale;

			float *anchor = anchors.row(i * num_scale + j);

			anchor[0] = cx - rs_w * 0.5f;
			anchor[1] = cy - rs_h * 0.5f;
			anchor[2] = cx + rs_w * 0.5f;
			anchor[3] = cy + rs_h * 0.5f;
		}
	}

	return anchors;
}

std::vector<SCRFD::BBox> SCRFD::generate_proposals(const ncnn::Mat &anchors, int feat_stride,
                                                   const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob,
                                                   const ncnn::Mat &kps_blob, float prob_threshold) {
	int w = score_blob.w;
	int h = score_blob.h;

	// generate face proposal from bbox deltas and shifted anchors
	const int num_anchors = anchors.h;

	std::vector<BBox> proposals;

	for (int q = 0; q < num_anchors; q++) {
		const float *anchor = anchors.row(q);

		const ncnn::Mat score = score_blob.channel(q);
		const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
		const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);

		// shifted anchor
		float anchor_y = anchor[1];

		float anchor_w = anchor[2] - anchor[0];
		float anchor_h = anchor[3] - anchor[1];

		for (int i = 0; i < h; i++) {
			float anchor_x = anchor[0];

			for (int j = 0; j < w; j++) {
				int index = i * w + j;

				float prob = score[index];

				if (prob >= prob_threshold) {
					float dx = bbox.channel(0)[index] * (float)feat_stride;
					float dy = bbox.channel(1)[index] * (float)feat_stride;
					float dw = bbox.channel(2)[index] * (float)feat_stride;
					float dh = bbox.channel(3)[index] * (float)feat_stride;

					float cx = anchor_x + anchor_w * 0.5f;
					float cy = anchor_y + anchor_h * 0.5f;

					float x0 = cx - dx;
					float y0 = cy - dy;
					float x1 = cx + dw;
					float y1 = cy + dh;

					proposals.push_back({x0, y0, x1 - x0 + 1, y1 - y0 + 1, prob});

					auto &obj = proposals.back();

					obj.landmarks[0].x = cx + kps.channel(0)[index] * (float)feat_stride;
					obj.landmarks[0].y = cy + kps.channel(1)[index] * (float)feat_stride;
					obj.landmarks[1].x = cx + kps.channel(2)[index] * (float)feat_stride;
					obj.landmarks[1].y = cy + kps.channel(3)[index] * (float)feat_stride;
					obj.landmarks[2].x = cx + kps.channel(4)[index] * (float)feat_stride;
					obj.landmarks[2].y = cy + kps.channel(5)[index] * (float)feat_stride;
					obj.landmarks[3].x = cx + kps.channel(6)[index] * (float)feat_stride;
					obj.landmarks[3].y = cy + kps.channel(7)[index] * (float)feat_stride;
					obj.landmarks[4].x = cx + kps.channel(8)[index] * (float)feat_stride;
					obj.landmarks[4].y = cy + kps.channel(9)[index] * (float)feat_stride;
				}

				anchor_x += (float)feat_stride;
			}

			anchor_y += (float)feat_stride;
		}
	}
	return proposals;
}

} // namespace aibum