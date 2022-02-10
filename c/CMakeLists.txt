cmake_minimum_required(VERSION 3.5)

project(step001_blurandsave)

find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(step001_blurandsave step001_blurandsave.cpp)

target_link_libraries(step001_blurandsave vpi opencv_core opencv_imgproc)

if(OpenCV_VERSION VERSION_LESS 3)
    target_link_libraries(step001_blurandsave opencv_highgui)
else()
    target_link_libraries(step001_blurandsave opencv_imgcodecs)
endif()
