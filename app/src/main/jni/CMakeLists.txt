cmake_minimum_required(VERSION 3.4.1)

add_library(colorspace_conversion SHARED
    imageutils_jni.cpp
	rgb2yuv.cpp
	rgb2yuv.h
	yuv2rgb.cpp
	yuv2rgb.h)

target_link_libraries(colorspace_conversion
                      android
                      log)