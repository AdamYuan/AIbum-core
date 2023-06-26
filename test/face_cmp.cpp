#include <MTCNNFaceNet.hpp>

inline float cosine(const std::array<float, 128> &l, const std::array<float, 128> &r) {
	float ret = .0f;
	for (std::size_t i = 0; i < 128; ++i)
		ret += l[i] * r[i];
	return ret;
}

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 2)
		return -1;

	aibum::MTCNNFaceNet mtcnn_face_net{"models/"};

	cv::Mat img1 = cv::imread(argv[0]), img2 = cv::imread(argv[1]);
	std::vector<aibum::Face> faces1 = mtcnn_face_net.GetFaces(img1), faces2 = mtcnn_face_net.GetFaces(img2);

	printf("faces1: %ld\nfaces2: %ld\n", faces1.size(), faces2.size());

	for (const auto &f1 : faces1) {
		for (const auto &f2 : faces2) {
			float sim = (cosine(f1.feature, f2.feature) + 1.0f) * 0.5f;
			printf("Similarity %f ", sim);
			if (sim > 0.65f)
				printf("SIMILAR");
			printf("\n");
		}
	}
}