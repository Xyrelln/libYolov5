#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <filesystem>
#include "YoloV5.h"
#include "AverageHash.h"
using namespace std;
namespace fs = std::filesystem;


int main() {

    // setup model
    YoloV5 yolo("/home/onion/dev/YoloV5-LibTorch/test/model.pt", false);
    string folder = "/home/onion/dev/YoloV5-LibTorch/imgOutput";
    vector<string> images;

    for (const auto & image : fs::directory_iterator(folder)) {
        images.push_back(image.path());
    }

    sort(images.begin(), images.end());  // directory_iterator won't promise sequence

    // this for loop finds the first detection of a sequence of images and crop it
    cv::Rect crop_region;  
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
            cout << "Detected character in " << image << endl;
            // frame = yolo.drawRectangle(frame, r[0], 1);
            float x1 = *r[0][0][0].data_ptr<float>();
            float y1 = *r[0][0][1].data_ptr<float>();
            float x2 = *r[0][0][2].data_ptr<float>();
            float y2 = *r[0][0][3].data_ptr<float>();

            crop_region.x = (int)x1;  // top left
            crop_region.y = (int)y1;  // top left
            crop_region.width = (int)(x2 - x1);
            crop_region.height = (int)(y2 - y1) / 2;  // get top half of the character model
        
            // // test 0
            // cv::Mat cropped_frame = frame(crop_region);
            // cv::imshow("", cropped_frame);
            // cv::waitKey(0);

            break;
        }
    }

    vector<int> this_hash, last_hash;
    for (int i = 0; i < images.size(); i++) {  // read images and each crop with crop_region. Then check diff of AverageHash of cropped reigon.
        cv::Mat image = cv::imread(images[i]);
        cv::Mat cropped_image = image(crop_region);

        // // test 0
        // cv::Mat temp;
        // cv::resize(cropped_image, temp, cv::Size(640, 640), 0, 0, cv::INTER_NEAREST);
        // cv::imshow("0", temp);
        // cv::waitKey(0);

        this_hash = AverageHash(cropped_image);
        
        if (i > 0) {
            int diff = Difference(this_hash, last_hash);
            if (diff > DIFF_THRESHOLD) {
                cout << "Frame: " << images[i] << " changed" << endl;
                cout << "Diff: " << diff << endl;
                return 0;
            }
            // cout << "Diff for frame " << i << ": " << diff << endl;  // test 0 
        }
        last_hash = this_hash;
    }

    cout << "Nothing changed detected" << endl;


    return 0;
}

