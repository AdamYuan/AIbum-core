#include <SCRFD.hpp>
#include <opencv2/opencv.hpp>

std::tuple<std::array<float, 6>, int> get_alignment_matrix(const aibum::FaceBox &box, float scale) {
	auto [nose_x, nose_y] = box.landmarks[2];
	auto [left_eye_x, left_eye_y] = box.landmarks[0];
	auto [right_eye_x, right_eye_y] = box.landmarks[1];
	float eye_dx = right_eye_x - left_eye_x, eye_dy = right_eye_y - left_eye_y;
	float angle = std::atan2(eye_dy, eye_dx);
	float alpha = std::cos(angle), beta = std::sin(angle);
	float w = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy) * scale;
	std::array<float, 6> mat = {alpha, beta,  -alpha * nose_x - beta * nose_y + w * 0.5f,
	                            -beta, alpha, beta * nose_x - alpha * nose_y + w * 0.5f};
	std::array<float, 6> inv_mat;
	ncnn::invert_affine_transform(mat.data(), inv_mat.data());
	return {inv_mat, (int)w};
}

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1)
		return -1;

	aibum::SCRFD scrfd{};
	scrfd.LoadFromFile("./models/scrfd_2.5g_kps-opt2.bin");

	cv::Mat image = cv::imread(argv[0]);
	std::vector<aibum::FaceBox> face_boxes = scrfd.Detect({image.data, image.cols, image.rows, ncnn::Mat::PIXEL_BGR});

	for (const auto &face_box : face_boxes) {
		// cv::Rect rect{face_box.x, face_box.y, face_box.w, face_box.h};
		// cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
		int color = 0;
		for (auto [x, y] : face_box.landmarks) {
			cv::circle(image, cv::Point{(int)x, (int)y}, 1, cv::Scalar(0, 255, color), 2);
			color += 50;
		}
	}
	// cv::imshow("SCRFD", image);
	// cv::waitKey();

	for (const auto &face_box : face_boxes) {
		auto [matrix, w] = get_alignment_matrix(face_box, 2.5f);
		cv::Mat face(w, w, image.type());
		ncnn::warpaffine_bilinear_c3((const unsigned char *)image.data, image.cols, image.rows,
		                             (unsigned char *)face.data, w, w, matrix.data());
		cv::imshow("SCRFD", face);
		cv::waitKey();
	}

	return 0;
}