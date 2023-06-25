#include <FaceNet.hpp>
#include <MTCNN.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::MTCNN mtcnn{"models/det1.param", "models/det1.bin",   "models/det2.param",
	                   "models/det2.bin",   "models/det3.param", "models/det3.bin"};

	cv::Mat image = cv::imread(argv[0]);
	std::vector<aibum::FaceBox> face_boxes = mtcnn.Detect(image);

	aibum::FaceNet face_net{"models/mobilefacenet.param", "models/mobilefacenet.bin"};

	for (auto face : face_boxes) {
		cv::Rect rect{face.x, face.y, face.size, face.size};
		cv::Mat cropped_image = image(rect);

		cv::imshow("MTCNN", cropped_image);

		auto features = face_net.GetFeature(cropped_image);
		for (float f : features)
			printf("%f ", f);
		printf("\n");

		cv::waitKey();
	}

	return 0;
}