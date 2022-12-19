#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

#define BRIGHTNESS_THRESHOLD 70
#define DIFF_THRESHOLD 60


vector<int> AverageHash(cv::Mat image, int width=16, int height=16, int threshold=BRIGHTNESS_THRESHOLD);

vector<int> AverageHash(string path, int width=16, int height=16, int threshold=BRIGHTNESS_THRESHOLD);

int Difference(vector<int> hash1, vector<int> hash2);