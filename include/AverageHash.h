#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;


vector<int> AverageHash(cv::Mat image, int width=16, int height=16, int threshold=45);

vector<int> AverageHash(string path, int width=16, int height=16, int threshold=45);

int Difference(vector<int> hash1, vector<int> hash2);