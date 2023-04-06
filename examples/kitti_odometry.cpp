#include <vector>
#include <iostream>
#include <cmath>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"
#include "visual_odometry/Triangulation.h"
#include "visual_odometry/MotionEstimation.h"

int main()
{
    cv::Mat imageLeft, imageRight;

    std::string imageLeftPath = "../resources/00/image_0/";
    std::string imageRightPath = "../resources/00/image_1/";
    std::string calibPath = "../resources/00/calib.txt";

    std::string frameCountStr;
    std::string fileName;
    uint frameCount = 0;
    frameCountStr = std::to_string(frameCount);
    fileName = std::string(6 - frameCountStr.length(), '0') + frameCountStr + ".png";

    imageLeft = cv::imread(imageLeftPath + fileName, cv::IMREAD_GRAYSCALE);
    imageRight = cv::imread(imageRightPath + fileName, cv::IMREAD_GRAYSCALE);
    VO::Triangulation triangulation = VO::Triangulation(calibPath);
    VO::MotionEstimation motionEstimation = VO::MotionEstimation(calibPath);

    VO::Frame framePrev, frameCurrent;
    framePrev = triangulation.triangulate(imageLeft, imageRight);

    std::cout << framePrev.pose << std::endl;
    while (true)
    {
        frameCountStr.clear();
        fileName.clear();
        frameCountStr = std::to_string(frameCount);
        ++frameCount;
        fileName = std::string(6 - frameCountStr.length(), '0') + frameCountStr + ".png";

        imageLeft = cv::imread(imageLeftPath + fileName, cv::IMREAD_GRAYSCALE);
        imageRight = cv::imread(imageRightPath + fileName, cv::IMREAD_GRAYSCALE);


        frameCurrent = triangulation.triangulate(imageLeft, imageRight);

        int ret = motionEstimation.motionEstimate(framePrev, frameCurrent);
        std::cout << frameCurrent.pose << std::endl;
        framePrev = frameCurrent;
    }

    return 0;
}