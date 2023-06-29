#include <SCRFDFaceNet.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::SCRFD scrfd{};

	cv::Mat image = cv::imread(argv[0]);
	std::vector<aibum::FaceBox> face_boxes = scrfd.Detect(image);

	for (const auto &face_box : face_boxes) {
		cv::Rect rect{face_box.x, face_box.y, face_box.size, face_box.size};
		cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("SCRFD", image);
	cv::waitKey();

	return 0;
}