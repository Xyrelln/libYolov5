find_package(OpenEXR)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_library(YoloV5 YoloV5.cpp)
add_library(AverageHash AverageHash.cpp)

target_link_directories(YoloV5 PUBLIC /usr/local/lib)

target_link_libraries(YoloV5 ${OpenCV_LIBS})
target_link_libraries(YoloV5 ${TORCH_LIBRARIES})

target_link_libraries(AverageHash ${OpenCV_LIBS})
