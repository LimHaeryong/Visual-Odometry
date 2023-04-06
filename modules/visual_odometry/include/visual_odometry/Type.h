#ifndef TYPE_H_
#define TYPE_H_


#include <vector>

#include <opencv2/opencv.hpp>

namespace VO
{
    struct Frame
    {
        std::vector<cv::Point2d> keyPoints;
        cv::Mat descriptors;
        std::vector<cv::Point3d> points3D;
        cv::Mat pose;
        cv::Mat relativePose;
    };
};

#endif // TYPE_H_