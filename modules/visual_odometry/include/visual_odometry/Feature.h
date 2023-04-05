#ifndef IMAGE_PREPROCESS_H_
#define IMAGE_PREPROCESS_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "visual_odometry/Type.h"

namespace VO{
    class Feature
    {
    public:
        Feature();
        virtual ~Feature();

        void detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors);
        void match(const cv::Mat& descriptor1, const cv::Mat& descriptor2, std::vector<cv::DMatch>& matches);
        
    private:
        cv::Ptr<cv::Feature2D> featureExtractor;
        cv::Ptr<cv::DescriptorMatcher> matcher;
    };
    

};

#endif // IMAGE_PREPROCESS_H_