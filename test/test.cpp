#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include "YoloV5.h"

int main()
{
	// Full User manual in Yolov5.h
	// YoloV5 yolo(torch::cuda::is_available() ? "./yolov5s.cuda.pt" : "./yolov5s.cpu.pt", torch::cuda::is_available());
	YoloV5 yolo("/home/onion/dev/YoloV5-LibTorch/test/yolov5s.cpu.pt", false);
	yolo.prediction(torch::rand({1, 3, 640, 640}));
	// Read labels
	// Not a necessary step, just write something on bbox
	std::ifstream f("./coco.txt");
	std::string name = "";
	int i = 0;
	std::map<int, std::string> labels;
	while (std::getline(f, name))
	{
		labels.insert(std::pair<int, std::string>(i, name));
		i++;
	}
	// Obtain cv::Mat
	cv::VideoCapture cap = cv::VideoCapture("sampleVid.mp4");

	// Setup width and height, not necessary
	// default value is 640 * 640
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1000);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 800);
	cv::Mat frame;
	while (cap.isOpened())
	{
		// Read a frame
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "Read frame failed!" << std::endl;
			break;
		}
		// Predict
		clock_t start = clock();
		std::vector<torch::Tensor> r = yolo.prediction(frame);
		clock_t ends = clock();
		std::cout <<"Running Time : "<<(double)(ends - start) / CLOCKS_PER_SEC << std::endl;
		// draw rectangle
		frame = yolo.drawRectangle(frame, r[0], labels);
		// show image
		cv::imshow("", frame);
		if (cv::waitKey(1) == 27) break;
	}
	return 0;
}