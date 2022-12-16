#include "AverageHash.h"
using namespace std;



vector<int> AverageHash(cv::Mat image, int width, int height, int threshold) {

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));

    cv::Mat gray_image;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat binarized_image;
    cv::threshold(gray_image, binarized_image, threshold, 255, cv::THRESH_BINARY);
    
    cv::Mat flat = image.reshape(1, binarized_image.total()*binarized_image.channels());
    vector<int> gray_array = image.isContinuous()? flat : flat.clone();
    
    
    return gray_array;
} 

vector<int> AverageHash(string path, int width, int height, int threshold) {
    cv::Mat image = cv::imread(path);
    AverageHash(image);
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