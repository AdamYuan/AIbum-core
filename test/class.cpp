#include "ImageNetTags.hpp"
#include <ImageNet.hpp>

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::ImageNet image_net{};

	cv::Mat image = cv::imread(argv[0]);
	auto tags = image_net.GetTags({image.data, image.cols, image.rows, ncnn::Mat::PIXEL_BGR}, 5);

	for (const auto &tag : tags)
		printf("%s [%d] = %f\n", kImageNetTags[tag.index], tag.index, tag.score);

	return 0;
}
