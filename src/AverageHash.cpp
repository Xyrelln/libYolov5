#include "AverageHash.h"
using namespace std;



vector<int> AverageHash(cv::Mat image, int width, int height, int threshold) {

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));

    cv::Mat gray_image;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);
;
    cv::Mat binarized_image;
    cv::threshold(gray_image, binarized_image, threshold, 255, cv::THRESH_BINARY);

    vector<int> gray_vector;
    for (int i = 0; i < binarized_image.rows; i++)
    {
        for (int j = 0; j < binarized_image.cols; j++)
        {
            uchar value = binarized_image.at<uchar>(i, j);
            if (value == 0)
            {
                gray_vector.push_back(0);
            }
            else
            {
                gray_vector.push_back(1);
            }
        }
    }

    // cv::Mat temp;  // test 0
    // cv::resize(binarized_image, temp, cv::Size(640, 640), 0, 0, cv::INTER_NEAREST);
    // cv::imshow("", temp);
    // cv::waitKey(0);
    
    return gray_vector;
} 

vector<int> AverageHash(string path, int width, int height, int threshold) {
    cv::Mat image = cv::imread(path);
    return AverageHash(image);
}

int Difference(vector<int> hash1, vector<int> hash2) {
    int diff = 0;
    int size1 = hash1.size();
    int size2 = hash2.size();
    int max_size = size1 > size2? size1 : size2;
    int min_size = size1 < size2? size1 : size2;

    for (int i = 0; i < min_size; i++) {
        if (hash1[i] != hash2[i]) {
            diff++;
        }
    }

    diff += max_size - min_size;

    return diff;
}