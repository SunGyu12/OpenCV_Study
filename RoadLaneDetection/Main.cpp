#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>
#include <string>
#include <vector>


//winspool.lib
//comdlg32.lib
//advapi32.lib
//shell32.lib
//ole32.lib
//oleaut32.lib
//uuid.lib
//odbc32.lib
//odbccp32.lib





#include "RoadLaneDetector.h"

using namespace cv;
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
	low_H = min(high_H - 1, low_H);
	setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
	high_H = max(high_H, low_H + 1);
	setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
	low_S = min(high_S - 1, low_S);
	setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
	high_S = max(high_S, low_S + 1);
	setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
	low_V = min(high_V - 1, low_V);
	setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
	high_V = max(high_V, low_V + 1);
	setTrackbarPos("High V", window_detection_name, high_V);
}
int main1()
{
	VideoCapture cap("input_2nd_lane.mp4");  //���� �ҷ�����
	namedWindow(window_capture_name);
	namedWindow(window_detection_name);
	// Trackbars to set thresholds for HSV values
	createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
	createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
	createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
	createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
	createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
	createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
	Mat frame, frame_HSV, frame_threshold;
	while (true) {
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		// Convert from BGR to HSV colorspace
		cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
		// Detect the object based on HSV Range Values
		inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
		// Show the frames
		imshow(window_capture_name, frame);
		imshow(window_detection_name, frame_threshold);
		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	return 0;
}

int main()
{
	cv::ocl::setUseOpenCL(true);
	cout << cv::ocl::haveOpenCL() << endl;
	if (!cv::ocl::haveOpenCL())
	{
		cout << "OpenCL IS not avaiable ..." << endl;
		return 0;
	}

	cv::namedWindow("result", WindowFlags::WINDOW_NORMAL);
	const Size wndSize = Size(854, 480);
	cv::resizeWindow("result", wndSize);
	RoadLaneDetector roadLaneDetector;
	Mat img_frame, img_filter, img_edges, img_mask, img_lines, img_result;
	vector<Vec4i> lines;
	vector<vector<Vec4i> > separated_lines;
	vector<Point> lane;
	string dir;
	
	//VideoCapture video("input_2nd_lane.mp4");  //���� �ҷ�����
	VideoCapture video("input_1st_lane.mp4");  //���� �ҷ�����

	if (!video.isOpened())
	{
		cout << "������ ������ �� �� �����ϴ�. \n" << endl;
		return -1;
	}

	video.read(img_frame);
	if (img_frame.empty()) return -1;

	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //���ϴ� �ڵ� ����

	//double fps_in = video.get(CAP_PROP_FPS);/25 fps
	double fps = video.get(CAP_PROP_FPS);	//������
	string filename = "./result.avi";		//��� ����

	writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
	if (!writer.isOpened()) {
		cout << "����� ���� ���� ������ �� �� �����ϴ�. \n";
		return -1;
	}

	video.read(img_frame);
	int cnt = 0;

	double fps_in = video.get(CAP_PROP_FPS);

	int frames = video.get(CAP_PROP_POS_FRAMES);//current frame number
	int frames_count = video.get(CAP_PROP_FRAME_COUNT);//totally frame count
	
	const int moving_avg_count = 3;
	Mat accumulated_img[moving_avg_count];
	int running_count = 0;

	Mat avg_img(img_frame.rows, img_frame.cols, CV_64FC3);

	while (1) {
		//1. ���� ������ �о�´�.
		if (!video.read(img_frame)) break;
		
		{
			
			accumulated_img[running_count++] = img_frame.clone();

			if (running_count < moving_avg_count) continue;
			else
			{
				Mat temp;
				avg_img.convertTo(avg_img, CV_64FC3);
				avg_img = 0;
				for (int i = 0; i < moving_avg_count; ++i)
				{
					accumulated_img[i].convertTo(temp, CV_64FC3);
					avg_img += (temp*1.2);
				}
				avg_img.convertTo(avg_img, CV_8UC1, 1. / moving_avg_count);

				for (int i = 0; i < moving_avg_count; ++i)
					accumulated_img[i].convertTo(temp, CV_8UC1);

				for (int i = 0; i < moving_avg_count-1;i++)
					accumulated_img[i+1].copyTo(accumulated_img[i]);
				
				avg_img.convertTo(avg_img, CV_8UC3);
				running_count--;
			}
			
		}
		img_frame = avg_img;

		Mat enhanceImg;
		bilateralFilter(img_frame, enhanceImg, 10, 50, 50);

		//2. ���, ����� ���� ���� �ִ� �͸� ���͸��Ͽ� ���� �ĺ��� �����Ѵ�.
		img_filter = roadLaneDetector.extract_colors(enhanceImg);

		//3. ������ GrayScale ���� ��ȯ�Ѵ�.
		cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

		//enhancement image level
		if(0)
		{
			Mat binaryAdaptive;
			int blockSize = 21; // �̿� ũ��
			int threshold = 10; //ȭ�Ҹ� (���-��� ��)�� ��
			adaptiveThreshold(img_filter, // �Է¿���
				binaryAdaptive, // ����ȭ ��� ����
				255, // �ִ� ȭ�� �� 
				ADAPTIVE_THRESH_MEAN_C, // Adaptive �Լ�
				THRESH_BINARY_INV, // ����ȭ Ÿ��
				blockSize,  // �̿�ũ��
				threshold); // threshold used
			Mat im_floodfill = img_filter.clone();
			floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));

			// Invert floodfilled image
			Mat im_floodfill_inv;
			bitwise_not(im_floodfill, im_floodfill_inv);

			// Combine the two images to get the foreground.
			Mat binaryAdaptive_filled = (img_filter | im_floodfill_inv);
		}

		//4. Canny Edge Detection���� ������ ����. (���� ���Ÿ� ���� Gaussian ���͸��� ����)
		double threshold1=20, threshold2 = 80;
		Canny(img_filter, img_edges, threshold1, threshold2);
		
		//5. �ڵ����� ������� �ٴڿ� �����ϴ� �������� �����ϱ� ���� ���� ������ ����
		img_mask = roadLaneDetector.limit_region(img_edges);

		//6. Hough ��ȯ���� ���������� ���� ������ ����
		lines = roadLaneDetector.houghLines(img_mask);

		if (lines.size() > 0) {
			//7. ������ ������������ �¿� ������ ���� ���ɼ��� �ִ� �����鸸 ���� �̾Ƽ� �¿� ���� ������ ����Ѵ�. 
			// ���� ȸ�͸� �Ͽ� ���� ������ ���� ã�´�.
			separated_lines = roadLaneDetector.separateLine(img_mask, lines);
			lane = roadLaneDetector.regression(separated_lines, img_frame);
			
			//8. ���� ���� ����
			dir = roadLaneDetector.predictDir();

			//9. ���� ���� ������ ������ �׸��� ���� �ٰ����� ������ ä���. ���� ���� ���� �ؽ�Ʈ�� ���� ���
			img_result = roadLaneDetector.drawLine(img_frame, lane, dir);
		}
		
		if (!img_result.empty())
		{
			//10. ����� ������ ���Ϸ� ����. ĸ���Ͽ� ���� ����
			writer << img_result;
			if (cnt++ == 15)
				imwrite("img_result.jpg", img_result);  //ĸ���Ͽ� ���� ����

			//11. ��� ���� ���
			cv::line(img_result, Point(0, 5), Point(img_result.cols - 1, 5), Scalar(10, 10, 10), 20);
			//cv::line(img_result, Point(0, 5), Point(frame, 5), Scalar(100, 50, 200), 30);
			frames = video.get(CAP_PROP_POS_FRAMES);
			drawMarker(img_result, Point((frames * 1.0 / frames_count) * wndSize.width, 5), Scalar(100, 50, 200), MARKER_TRIANGLE_DOWN, 10);
			imshow("result", img_result);
		}
		

		//esc Ű ����
		if (waitKey(1) == 27) 
			break;
	}
	return 0;
}
