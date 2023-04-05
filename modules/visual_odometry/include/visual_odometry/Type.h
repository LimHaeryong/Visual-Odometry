#include <vector>

#include <opencv2/opencv.hpp>

namespace VO
{

    struct Frame
    {
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;
        std::vector<cv::Point3f> points3D;
        cv::Vec6f pose;
    };
};
