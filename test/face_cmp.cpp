#include <FaceNet.hpp>
#include <opencv2/opencv.hpp>

inline float cosine(const aibum::FaceFeature &l, const aibum::FaceFeature &r) {
	float ret = .0f;
	for (std::size_t i = 0; i < 128; ++i)
		ret += l[i] * r[i];
	return ret;
}

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 2)
		return -1;

	aibum::SCRFD scrfd{};
	aibum::FaceNet face_net{};

	scrfd.LoadFromFile("./models/scrfd_2.5g_kps-opt2.bin");
	face_net.LoadFromFile("./models/mobilefacenet.bin");

	cv::Mat img1 = cv::imread(argv[0]), img2 = cv::imread(argv[1]);
	std::vector<aibum::Face> faces1 = face_net.GetFaces(scrfd, {img1.data, img1.cols, img1.rows, ncnn::Mat::PIXEL_BGR}),
	                         faces2 = face_net.GetFaces(scrfd, {img2.data, img2.cols, img2.rows, ncnn::Mat::PIXEL_BGR});

	printf("faces1: %ld\nfaces2: %ld\n", faces1.size(), faces2.size());

	for (const auto &f1 : faces1) {
		for (const auto &f2 : faces2) {
			float sim = (cosine(f1.feature, f2.feature) + 1.0f) * 0.5f;
			printf("Similarity %f ", sim);
			if (sim > 0.75f)
				printf("SIMILAR");
			printf("\n");
		}
	}
}