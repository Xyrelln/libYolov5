#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;


/**
 * ImageResizeData 图片处理过后保存图片的数据结构
 */
class ImageResizeData
{
public:
    // 添加处理过后的图片
	void setImg(cv::Mat img);
    // 获取处理过后的图片
	cv::Mat getImg();
    // 当原始图片宽高比大于处理过后图片宽高比时此函数返回 true
	bool isW();
    // 当原始图片高宽比大于处理过后图片高宽比时此函数返回 true
	bool isH();
    // 添加处理之后图片的宽undefined reference to `cv::imread(std::string const&, int)'
	void setWidth(int width);
    // 获取处理之后图片的宽
	int getWidth();
    // 添加处理之后图片的高
	void setHeight(int height);
    // 获取处理之后图片的高
	int getHeight();
    // 添加原始图片的宽
	void setW(int w);
    // 获取原始图片的宽
	int getW();
    // 添加原始图片的高
	void setH(int h);
    // 获取原始图片的高
	int getH();
    // 添加从原始图片到处理过后图片所添加黑边大小
	void setBorder(int border);
    // 获取从原始图片到处理过后图片所添加黑边大小
	int getBorder();
private:
    // 处理过后图片高
	int height;
	// 处理过后图片宽
    int width;
	// 原始图片宽
    int w;
	// 原始图片高
    int h;
	// 从原始图片到处理图片所添加的黑边大小
    int border;
	// 处理过后的图片
    cv::Mat img;
};

/**
 * YoloV5 的实现类
 */
class YoloV5
{
public:
    /**
     * 构造函数
     * @param ptFile yoloV5 pt文件路径
	 * @param isCuda 是否使用 cuda 默认不起用
	 * @param height yoloV5 训练时图片的高
	 * @param width yoloV5 训练时图片的宽
	 * @param confThres 非极大值抑制中的 scoreThresh
	 * @param iouThres 非极大值抑制中的 iouThresh
     */
	YoloV5(std::string ptFile, bool isCuda = false, bool isHalf = false, int height = 640, int width = 640,  float confThres = 0.25, float iouThres = 0.45);
	/**
	 * 预测函数
	 * @param data 语言预测的数据格式 (batch, rgb, height, width)
	 */
	std::vector<torch::Tensor> prediction(torch::Tensor data);
	/**
	 * 预测函数
	 * @param filePath 需要预测的图片路径
	 */
	std::vector<torch::Tensor> prediction(std::string filePath);
	/**
	 * 预测函数
	 * @param img 需要预测的图片
	 */
	std::vector<torch::Tensor> prediction(cv::Mat img);
	/**
	 * 预测函数 
	 * @param imgs 需要预测的图片集合
	 */
	std::vector<torch::Tensor> prediction(std::vector <cv::Mat> imgs);
	/**
	 * 改变图片大小的函数
	 * @param img 原始图片
	 * @param height 要处理成的图片的高
	 * @param width 要处理成的图片的宽
	 * @return 封装好的处理过后图片数据结构
	 */
	static ImageResizeData resize(cv::Mat img, int height, int width);
	/**
	 * 改变图片大小的函数
	 * @param img 原始图片
	 * @return 封装好的处理过后图片数据结构
	 */
	ImageResizeData resize(cv::Mat img);
	/**
	 * 改变图片大小的函数
	 * @param imgs 原始图片集合
	 * @param height 要处理成的图片的高
	 * @param width 要处理成的图片的宽
	 * @return 封装好的处理过后图片数据结构
	 */
	static std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs, int height, int width);
	/**
	 * 改变图片大小的函数
	 * @param imgs 原始图片集合
	 * @return 封装好的处理过后图片数据结构
	 */
	std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param colors 每种类型对应颜色
	 * @param labels 类别标签 
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param colos 每种类型对应颜色
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 用于判断给定数据是否存在预测
	 * @param clazz 通过预测函数处理好的结果
	 * @return 如果图片中存在给定某一种分类返回 true
	 */
	bool existencePrediction(torch::Tensor clazz);
	/**
	 * 用于判断给定数据是否存在预测
	 * @param classs 通过预测函数处理好的结果
	 * @return 如果图片集合中存在给定某一种分类返回 true
	 */
	bool existencePrediction(std::vector<torch::Tensor> classs);


private:
	bool isCuda;
	bool isHalf;
	// confidence threshold
	float confThres;
	// IoU threshold
	float iouThres;
	float height;
	float width;
	// color mapping
	std::map<int, cv::Scalar> mainColors;
	// model
	torch::jit::script::Module model;
	// get random ccolor
	cv::Scalar getRandScalar();
	// image channel ensure rgb
	cv::Mat img2RGB(cv::Mat img);
	// image to Tensor
	torch::Tensor img2Tensor(cv::Mat img);
	// (center_x center_y w h) to (left, top, right, bottom)
	torch::Tensor xywh2xyxy(torch::Tensor x);
	// (left, top, right, bottom) to (center_x center_y w h)
	torch::Tensor xyxy2xywh(torch::Tensor x);
	// NMS
	torch::Tensor nms(torch::Tensor bboxes, torch::Tensor scores, float thresh);
	// prediction size back to original
	std::vector<torch::Tensor> sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs);
	// NMS for predictions
	std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float confThres = 0.25, float iouThres = 0.45);
};

// 終わらない
class LoadImages {
	public:
	// variables and assets
	std::vector<std::string> files;
	std::vector<std::string> images, videos;
	unsigned int ni = 0, nv = 0, nf = 0;  // number of images, video and total files
	unsigned int count = 0;  // a flag that stores the next file index, denotes from 0

	// Constructor
	LoadImages(std::vector<std::string> paths, int img_size=640, int stride=32) {
	
		// paths contains a sequence of .jpg strings
		// now ensure paths are absolute and push them respectively into vector "files"
		for (auto path : paths) {
			std::string p = fs::absolute(p);
			if (p.std::string::find('*') != std::string::npos) { // if '*' in p
				//TODO: wildcard 
			} else {
				files.push_back(path);
			}
		}

		// catagorize files into images or videos, push into corresponding vector
		for (auto file : files) {
			if (std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), _file_extension(file)) != IMG_FORMATS.end()) {
				images.push_back(file);
			} else if (std::find(VID_FORMATS.begin(), VID_FORMATS.end(), _file_extension(file)) != VID_FORMATS.end()) {
				videos.push_back(file);
			} else {
				std::cout << file << " is not detected as an image or a video" << std::endl;
			}
		}
		
		ni = images.size();
		nv = videos.size();
		nf = ni + nv;

		// TODO: if any videos, neoVideo()
	}
	
	private:
	// Const references
	const std::vector<std::string> IMG_FORMATS{ "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm" };  // include image suffixes
	const std::vector<std::string> VID_FORMATS{ "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"};  // include video suffixes
	
	// Functions
	cv::Mat read() {		
		cv::Mat im0;
		if (count >= nf) {			// Remember to check if returned matrix is empty.						
			return im0;				// It indicates file traverse is over.	
		}
		std::string path = files[count];
		count += 1;
		im0 = cv::imread(path);
		if (im0.data == NULL) {
			std::cout << "Invalid Image:" << path << std::endl;
			std::abort();
		} 
		return im0;
	}


	void _new_video(std::string path);

	// utils
	std::string _file_extension(std::string filename) {
		std::size_t found = filename.find_last_of('.');
		return filename.substr(found);
	}

	std::string _file_name(std::string path) const {  // filename with extension
		std::size_t found = path.find_last_of("/\\");
		return path.substr(found);
	}

};