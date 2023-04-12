#ifndef IMAGE_PREPROCESS_H_
#define IMAGE_PREPROCESS_H_

#include <cmath>
#include <vector>
#include <omp.h>

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
        void match(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);
        void stereoMatch(const std::vector<cv::KeyPoint> &keyPointsLeft, const cv::Mat &descriptorsLeft, const std::vector<cv::KeyPoint> &keyPointsRight, const cv::Mat &descriptorsRight, std::vector<cv::DMatch> &matches);

    private:
        std::vector<std::vector<uint>> getRightKeypointIndicesInRow(double margin, const std::vector<cv::KeyPoint> &keyPointsRight);
        std::pair<uint, uint> findClosestKeyPoint(const cv::KeyPoint& keyPointLeft, const cv::Mat &descriptorLeft, const std::vector<uint>& indicesRight, const std::vector<cv::KeyPoint> &keyPointsRight, const cv::Mat& descriptorsRight, const float xRightMin, const float xRightMax);
        uint getHammingDist(const cv::Mat& descriptor1, const cv::Mat& descriptor2);

        cv::Ptr<cv::Feature2D> featureExtractor;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        cv::Size imageSize;
        std::vector<float> scaleFactors;
        float focalLength;
        uint hammingDistThresh;
    };
    

};

#endif // IMAGE_PREPROCESS_H_