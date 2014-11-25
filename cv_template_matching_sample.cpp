//
//  cv_template_matcing_sample.cpp - テンプレートマッチングのサンプル
//
//  参考
//      http://docs.opencv.org/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
//
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#ifdef _WIN32
#ifdef _DEBUG
#pragma comment(lib, "opencv_core2410d.lib")
#pragma comment(lib, "opencv_imgproc2410d.lib")
#pragma comment(lib, "opencv_highgui2410d.lib")
#else
#pragma comment(lib, "opencv_core2410.lib")
#pragma comment(lib, "opencv_imgproc2410.lib")
#pragma comment(lib, "opencv_highgui2410.lib")
#endif
#endif

std::string window_name = "cv_template_matching_sample";

int template_size = 100;
double match_threshold = 0.7;

cv::Mat capture_img;
cv::Mat template_img;
cv::Point match_pos;
bool has_match_pos = false;

void clear()
{
	template_img.release();
	match_pos = cv::Point(-1, -1);
	has_match_pos = false;
}

cv::Rect create_roi_center(const int &x, const int &y, const int &size) 
{
	cv::Rect roi;
	roi.x      = x - size / 2;
	roi.y      = y - size / 2;
	roi.width  = size;
	roi.height = size;
	return roi;
}

cv::Rect create_roi_lt(const cv::Point &pos, const int &size) 
{
	cv::Rect roi;
	roi.x      = pos.x;
	roi.y      = pos.y;
	roi.width  = size;
	roi.height = size;
	return roi;
}

cv::Rect correct_range(const cv::Rect &src, const cv::Size &size)
{
	cv::Rect r = src;

	if (r.x < 0) r.x = 0;
	if (r.y < 0) r.y = 0;
	if (size.width  < r.x + r.width ) r.x = size.width  - r.width;
	if (size.height < r.y + r.height) r.y = size.height - r.height;

	if (r.x < 0) {
		r.x = 0;
		r.width = size.width;
	}
	if (r.y < 0) {
		r.y = 0;
		r.height = size.height;
	}

	return r;
}

void pickup_template_image(int x, int y)
{
	cv::Mat gray_img;
	cv::Rect template_roi;

	cv::cvtColor(capture_img, gray_img, CV_BGR2GRAY);

	template_roi = create_roi_center(x, y, template_size);
	template_roi = correct_range(template_roi, cv::Size(640, 480));

	if (!capture_img.empty()) {
		gray_img(template_roi).copyTo(template_img);
	}
}

void mouse_callback_function(int event, int x, int y, int, void* userdata)
{
	if (event & CV_EVENT_LBUTTONDOWN) {
		pickup_template_image(x, y);
	}
}

int main(int argc, char* argv[])
{
	cv::VideoCapture capture;
	capture.open(0);

	cv::namedWindow(window_name);
	cv::setMouseCallback(window_name, mouse_callback_function, NULL);

	clear();

	while(true) {
		capture >> capture_img;

		has_match_pos = false;

		if (!template_img.empty()) {
			// グレースケール画像に変換
			cv::Mat gray_img;
			cv::cvtColor(capture_img, gray_img, CV_BGR2GRAY);

			// 結果格納用cv::Matの準備
			cv::Size result_size(gray_img.cols - template_img.cols + 1, gray_img.rows - template_img.rows + 1);
			cv::Mat result(result_size, CV_32FC1);

			// テンプレートマッチングの実行
			cv::matchTemplate(gray_img, template_img, result, CV_TM_CCOEFF_NORMED);

			// CV_TM_CCOEFF_NORMEDの場合は最大値の座標がマッチ位置
			double min_val, max_val, match_val;
			cv::Point min_pos, max_pos;
			cv::minMaxLoc(result, &min_val, &max_val, &min_pos, &max_pos, cv::Mat());

			match_pos = max_pos;
			match_val = max_val;

			// 設定した閾値以上の場合はマッチしていると判断する
			if (match_val > match_threshold) {
				has_match_pos = true;
			}
			
			// 設定した閾値以上の場合はマッチしていると判断する
			cv::Mat result_img(result.size(), CV_8UC1);
			cv::convertScaleAbs(result, result_img, 255.0, 0.0);
			cv::imshow("result", result_img);
		}

		// 結果の描画
		cv::Mat canvas;
		capture_img.copyTo(canvas);

		if (has_match_pos) {
			cv::Rect match_roi = create_roi_lt(match_pos, template_size);
			cv::rectangle(canvas, match_roi, CV_RGB(0, 255, 0));
		}

		if (!template_img.empty()) {
			cv::Mat tmp_img;
			cv::cvtColor(template_img, tmp_img, CV_GRAY2BGR);
			cv::Rect tmp_roi(canvas.cols - tmp_img.cols, canvas.rows - tmp_img.rows, tmp_img.cols, tmp_img.rows);
			tmp_img.copyTo(canvas(tmp_roi));
			cv::rectangle(canvas, tmp_roi, CV_RGB(0,255,0));
		}
		cv::imshow(window_name, canvas);

		int c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
		else if (c == 'c') {
			clear();
		}
	}

	capture.release();
	cv::destroyAllWindows();

	return 0;
}

