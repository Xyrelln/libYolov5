# Project LibYOLOv5
This project is a C++ library of YOLOv5, providing basic functions in production.

## Prerequisites
CMake, OpenCV and libTorch installed 

## Usage (tested under archlinux version 2022.12.01)
1. Install Opencv and libTorch
2. Add to path
Example:
```
$ export Torch_DIR=/usr/local/libtorch
$ export OpenCV_DIR=/usr/local/lib
```
3. Compile test file
```
$ mkdir build && cd build
$ cmake ..
$ make
```
