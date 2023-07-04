#include <StyleTransfer.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 2)
		return -1;

	cv::Mat image = cv::imread(argv[1]);

	aibum::StyleTransfer style_transfer{argv[0]};
	ncnn::Mat out = style_transfer.Transfer({image.data, image.cols, image.rows, ncnn::Mat::PIXEL_BGR});

	cv::Mat out_bgr;
	out_bgr.create(out.h, out.w, CV_8UC3);
	out.to_pixels(out_bgr.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imshow("out", out_bgr);
	cv::waitKey(0);
}