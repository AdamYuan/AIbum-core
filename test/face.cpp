#include <MTCNNFaceNet.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::MTCNN mtcnn{"models/"};

	cv::Mat image = cv::imread(argv[0]);
	std::vector<aibum::FaceBox> face_boxes = mtcnn.Detect(image);

	cv::Mat tagged_image = image.clone();
	for (const auto &face_box : face_boxes) {
		cv::Rect rect{face_box.x, face_box.y, face_box.size, face_box.size};
		cv::rectangle(tagged_image, rect, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("MTCNN", tagged_image);
	cv::waitKey();

	return 0;
}