#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"
#include "visual_odometry/Triangulation.h"
#include "visual_odometry/MotionEstimation.h"

int main()
{
    cv::Mat imageLeft, imageRight;
    cv::Mat resultImage(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(resultImage, "Ground Truth", cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::putText(resultImage, "Result", cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    std::string scene = "07";
    std::string imageLeftPath = "../resources/sequences/" + scene + "/image_0/";
    std::string imageRightPath = "../resources/sequences/" + scene + "/image_1/";
    std::string calibPath = "../resources/sequences/" + scene + "/calib.txt";
    std::string posePath = "../resources/poses/" + scene + ".txt";
    double value;

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

    std::ifstream ifs(posePath);
    std::string line;
    if (!ifs.is_open())
    {
        std::cout << "cannot open posePath : " << posePath << std::endl;
    }
    std::getline(ifs, line);

    while (std::getline(ifs, line))
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
        int x = static_cast<int>(frameCurrent.pose.at<double>(0, 3) / 0.3 + 750.0);
        int z = static_cast<int>(-frameCurrent.pose.at<double>(2, 3) / 0.3 + 600.0);

        cv::circle(resultImage, cv::Point(x, z), 1, cv::Scalar(0, 255, 0), cv::FILLED);

        double xPose, zPose;
        std::stringstream ss(line);
        ss >> value >> value >> value >> xPose >> value >> value >> value >> value >> value >> value >> value >> zPose;

        int xPoseInt = static_cast<int>(xPose / 0.3 + 750.0);
        int zPoseInt = static_cast<int>(-zPose / 0.3 + 600.0);
        cv::circle(resultImage, cv::Point(xPoseInt, zPoseInt), 1, cv::Scalar(0, 0, 255), cv::FILLED);
        framePrev = frameCurrent;

        cv::imshow("result", resultImage);
        cv::waitKey(1);
    }

    cv::imwrite("../result/scene" + scene + "_result.jpg", resultImage);

    return 0;
}