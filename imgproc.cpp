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

