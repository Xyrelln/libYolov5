#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <filesystem>
#include "YoloV5.h"
#include "AverageHash.h"
using namespace std;
namespace fs = std::filesystem;

#define BRIGHTNESS_THRESHOLD 45


int main() {

    // setup model
    YoloV5 yolo("/home/onion/dev/YoloV5-LibTorch/test/yolov5s.cpu.pt", false);
    // YoloV5 yolo(torch::cuda::is_available() ? "/home/onion/dev/YoloV5-LibTorch/test/yolov5s.cuda.pt" : "/home/onion/dev/YoloV5-LibTorch/test/yolov5s.cpu.pt", torch::cuda::is_available());
    string folder = "/home/onion/dev/mmd/yolov5/imgOutput";
    vector<string> images;

    for (const auto & image : fs::directory_iterator(folder)) {
        images.push_back(image.path());
    }

    sort(images.begin(), images.end());  // directory_iterator won't promise sequence

    cv::Mat cropped_frame;  // this for loop finds the first detection of a sequence of images and crop it
    for (const auto & image : images) {
        cv::Mat frame = cv::imread(image);
        std::vector<torch::Tensor> r = yolo.prediction(frame);  // rows of predictions. Predictions is {xyxy, conf, class} with length of 6
        torch::Tensor empty_dummy = torch::empty({0, 6});

        // if there are no detections, load the next frame
        if (r[0].sizes() == empty_dummy.sizes()) {
            cout << "Prediction for " << image << " is empty" << endl;
        }
        // if there is, predict and crop that image to prediction
        else {
            frame = yolo.drawRectangle(frame, r[0], 1);
            float x1 = *r[0][0][0].data_ptr<float>();
            float y1 = *r[0][0][1].data_ptr<float>();
            float x2 = *r[0][0][2].data_ptr<float>();
            float y2 = *r[0][0][3].data_ptr<float>();

            float x = (x1 + x2) / 2;
            float y = (y1 + y2) / 2;
            float w = (x2 - x1) / 2;
            float h = (y2 - y1) / 2;

            cv::Rect crop_region(x, y, w, h);
            cropped_frame = frame(crop_region);
            // cv::imshow("", cropped_frame);
            // cv::waitKey(0);
            cout << "image " << image << " processed: " << endl;
            break;
        }
    }

    vector<int> this_hash, last_hash;
    for (int i = 0; i < images.size(); i++) {
        cv::Mat image = cv::imread(images[i]);
        this_hash = AverageHash(image);
        
        if (i > 0) {
            if (Difference(this_hash, last_hash) > BRIGHTNESS_THRESHOLD) {
                cout << "Frame " << i << " changed" << endl;
                return 0;
            }
        }
    }

    cout << "Nothing changed detected" << endl;


    return 0;
}

