#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram을 쌓습니다. 
					histogram[inputMat.at<uchar>(y, x)]++;

					// hint 1 : for loop 를 이용해서 cv::Mat 순회 시 (1채널의 경우) 
					// inputMat.at<uchar>(y, x)와 같이 데이터에 접근할 수 있습니다. 
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			
			dst.create(srcMat.size(), CV_8UC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// Todo : hs 2차원 히스토그램을 계산하는 함수를 작성합니다. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					int quantized_h_value = UTIL::quantize(srcMat.at<cv::Vec3b>(y, x)[0]);
					int quantized_s_value = UTIL::quantize(srcMat.at<cv::Vec3b>(y, x)[1]);

					outputProb.at<uchar>(y, x) = (uchar)UTIL::h_r(model_hist, input_hist, quantized_h_value, quantized_s_value)*255;
					// hint 1 : UTIL::quantize()를 이용해서 srtMat의 값을 양자화합니다. 
					// hint 2 : UTIL::h_r() 함수를 이용해서 outputPorb 값을 계산합니다. 
				}
			}
		}

		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int & threshold)
		{
			cv::Mat inputMat = src.getMat();
			dst.create(inputMat.size(), CV_8UC1);
			cv::Mat outputMat = dst.getMat();

			for (int i = 0; i < inputMat.rows; i++) {
				for (int j = 0; j < inputMat.cols; j++) {
					if (inputMat.at<uchar>(i, j) <= threshold)
						outputMat.at<uchar>(i, j) = 0;
					else
						outputMat.at<uchar>(i, j) = 255;

				}
			}
		}

		void thresh_otsu(cv::InputArray src, cv::OutputArray dst)
		{
			cv::Mat inputMat = src.getMat();

			int inputHistogram[256] = { 0, };
			IPCVL::IMG_PROC::calcHist(inputMat, inputHistogram);

			double max_v_between = 0;
			int max_index_v_between = 0;
			for (int t = 1; t < 256; t++) {
				int a = 0;
				int b = 0;
				double mean_0 = 0;
				double mean_1 = 0;
				for (int i = 0; i < t; i++) {
					a += inputHistogram[i];
					mean_0 += inputHistogram[i] * i;
				}
				
				for (int j = t; j < 256; j++) {
					b += inputHistogram[j];
					mean_1 += inputHistogram[j] * j;
				}
				mean_0 /= a;
				mean_1 /= b;
				double w_0 = (double)a / (a + b);
				double v_between = w_0*(1 - w_0)*((mean_0 - mean_1)*(mean_0 - mean_1));
				if (max_v_between < v_between) {
					max_v_between = v_between;
					max_index_v_between = t;
				}
			}
			thresh_binary(src,dst, max_index_v_between);
		}

		void flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;
				IPCVL::IMG_PROC::flood_fill4(l, j, i + 1, label);
				IPCVL::IMG_PROC::flood_fill4(l, j - 1, i, label);
				IPCVL::IMG_PROC::flood_fill4(l, j, i - 1, label);
				IPCVL::IMG_PROC::flood_fill4(l, j + 1, i, label);
			}
		}

		void flood_fill8(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;
				IPCVL::IMG_PROC::flood_fill8(l, j, i + 1, label);
				IPCVL::IMG_PROC::flood_fill8(l, j+1, i + 1, label);
				IPCVL::IMG_PROC::flood_fill8(l, j + 1, i, label);
				IPCVL::IMG_PROC::flood_fill8(l, j + 1, i - 1, label);
				IPCVL::IMG_PROC::flood_fill8(l, j, i - 1, label);
				IPCVL::IMG_PROC::flood_fill8(l, j - 1, i-1, label);
				IPCVL::IMG_PROC::flood_fill8(l, j - 1, i, label);
				IPCVL::IMG_PROC::flood_fill8(l, j - 1, i+1, label);
			}
		}

		void efficient_flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			std::queue<int> j_que;
			std::queue<int> i_que;
			j_que.push(j);
			i_que.push(i);
			while (!j_que.empty()) {
				int y = j_que.front();
				int x = i_que.front();
				j_que.pop();
				i_que.pop();
				if (l.at<int>(y, x) == -1) {
					int left = x;
					int right = x;
					while (l.at<int>(y, left - 1) == -1) left--;
					while (l.at<int>(y, right + 1) == -1) right++;
					for (int c = left; c < right + 1; c++) {
						l.at<int>(y, c) = label;
						if (l.at<int>(y - 1, c) == -1 && (c == left || l.at<int>(y - 1, c - 1) != -1)) {
							j_que.push(y - 1);
							i_que.push(c);
						}
						if (l.at<int>(y + 1, c) == -1 && (c == left || l.at<int>(y + 1, c - 1) != -1)) {
							j_que.push(y + 1);
							i_que.push(c);
						}
					}
				}
			}

		}

		void flood_fill(cv::InputArray src, cv::OutputArray dst, const UTIL::CONNECTIVITIES & direction)
		{
			cv::Mat inputMat = src.getMat();
			dst.create(inputMat.size(), CV_32SC1);
			cv::Mat outputMat = dst.getMat();


			for (int j = 0; j < inputMat.rows; j++) {
				for (int i = 0; i < inputMat.cols; i++) {
					if (j == 0 || i == 0 || j == inputMat.rows - 1 || i == inputMat.cols - 1) 
						outputMat.at<int>(j, i) = 0;
					else if (inputMat.at<uchar>(j, i) == 255)
						outputMat.at<int>(j, i) = -1;
					else
						outputMat.at<int>(j, i) = 0;
				}
			}

			if (direction == IPCVL::UTIL::CONNECTIVITIES::NAIVE_FOURWAY) {
				int label = 1;
				for (int j = 1; j < outputMat.rows - 1; j++) {
					for (int i = 1; i < outputMat.cols - 1; i++) {
						if (outputMat.at<int>(j, i) == -1) {
							IPCVL::IMG_PROC::flood_fill4(outputMat, j, i, label);
							label++;
						}
					}
				}
			}
			else if (direction == IPCVL::UTIL::CONNECTIVITIES::NAIVE_EIGHT_WAY) {
				int label = 1;
				for (int j = 1; j < outputMat.rows - 1; j++) {
					for (int i = 1; i < outputMat.cols - 1; i++) {
						if (outputMat.at<int>(j, i) == -1) {
							IPCVL::IMG_PROC::flood_fill8(outputMat, j, i, label);
							label++;
						}
					}
				}
			}
			else if (direction == IPCVL::UTIL::CONNECTIVITIES::EFFICIENT_FOURWAY) {
				int label = 1;
				for (int j = 1; j < outputMat.rows - 1; j++) {
					for (int i = 1; i < outputMat.cols - 1; i++) {
						if (outputMat.at<int>(j, i) == -1) {
							IPCVL::IMG_PROC::efficient_flood_fill4(outputMat, j, i, label);
							label++;
						}
					}
				}
			}
			else {

			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					int quantized_h_value = UTIL::quantize(mat_h.at<uchar>(y, x));
					int quantized_s_value = UTIL::quantize(mat_s.at<uchar>(y, x));
					histogram[quantized_h_value][quantized_s_value]++;
					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram에 있는 값들을 순회하며 (hsv.rows * hsv.cols)으로 정규화합니다. 
					histogram[j][i] /= (hsv.rows * hsv.cols);
				}
			}
		}
	}  // namespace IMG_PROC
}

