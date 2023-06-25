#include <ImageNet.hpp>
#include <ImageNetTags.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::ImageNet image_net{"models/mobilenet_v3.param", "models/mobilenet_v3.bin"};

	cv::Mat image = cv::imread(argv[0]);
	auto tags = image_net.GetTags(image, 5);

	for (const auto &tag : tags)
		printf("%s [%d] = %f\n", kImageNetTags[tag.index], tag.index, tag.score);

	return 0;
}
