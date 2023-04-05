#include <vector>
#include <iostream>
#include <cmath>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"
#include "visual_odometry/Triangulation.h"

int main()
{
    cv::Mat imageLeft1 = cv::imread("../resources/00/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRight1 = cv::imread("../resources/00/image_1/000000.png", cv::IMREAD_GRAYSCALE);

    cv::Mat imageLeft2 = cv::imread("../resources/00/image_0/000001.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRight2 = cv::imread("../resources/00/image_1/000001.png", cv::IMREAD_GRAYSCALE);

    std::string calibPath = "../resources/00/calib.txt";

    VO::Triangulation triangulation = VO::Triangulation(calibPath);
    VO::Frame frame1 = triangulation.triangulate(imageLeft1, imageRight1);
    VO::Frame frame2 = triangulation.triangulate(imageLeft2, imageRight2);

    std::cout << frame1.points3D.size() << std::endl;
    std::cout << frame2.points3D.size() << std::endl;
    return 0;
}