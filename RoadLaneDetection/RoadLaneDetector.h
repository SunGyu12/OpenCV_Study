#pragma once

#define OPENCV_470
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef OPENCV_470
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#endif // OPENCV_470

#ifdef _DEBUG
#pragma comment(lib,"opencv_world470d.lib")
#else	//RELEASE
#pragma comment(lib,"opencv_world470.lib")
#endif
using namespace std;
using namespace cv;

#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace cv;

class RoadLaneDetector
{
public:
	RoadLaneDetector();
	~RoadLaneDetector();
private:
	double img_size, img_center;
	double left_m, right_m;
	Point left_b, right_b;
	bool left_detect = false, right_detect = false;

	//관심 영역 범위 계산시 사용 
	double poly_bottom_width = 0.85;  //사다리꼴 아래쪽 가장자리 너비 계산을 위한 백분율
	double poly_top_width = 0.07;     //사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
	double poly_height = 0.4;         //사다리꼴 높이 계산을 위한 백분율

	//detection predict Direction
	double predict_x;

	Mat drawDebug;
public:
	Mat extract_colors(Mat img_frame);
	Mat limit_region(Mat img_edges); 
	vector<Vec4i> houghLines(Mat img_mask);
	vector<vector<Vec4i> > separateLine(Mat img_edges, vector<Vec4i> lines); 
	vector<Point> regression(vector<vector<Vec4i> > separated_lines, Mat img_input);
	string predictDir();
	Mat drawLine(Mat img_input, vector<Point> lane, string dir);
	void DrawDashedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, cv::Scalar color= cv::Scalar(0, 0, 255), int thickness=1, std::string style= "dotted", int gap=10);
	bool IntersectPoint(const Point& AP1, const Point& AP2, const Point& BP1, const Point& BP2, Point* IP);
};

