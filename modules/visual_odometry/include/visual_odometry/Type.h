#ifndef TYPE_H_
#define TYPE_H_


#include <vector>

#include <opencv2/opencv.hpp>

namespace VO
{
    struct Frame
    {
        std::vector<cv::Point2f> keyPoints;
        cv::Mat descriptors;
        std::vector<cv::Point3f> points3D;
        cv::Vec6f pose;
    };
};

#endif // TYPE_H_