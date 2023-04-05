#include "visual_odometry/Feature.h"

VO::Feature::Feature()
{
    featureExtractor = cv::ORB::create();
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

}

VO::Feature::~Feature()
{
}

void VO::Feature::detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors)
{
    featureExtractor->detectAndCompute(image, cv::Mat(), keyPoints, descriptors);
}

void VO::Feature::match(const cv::Mat& descriptor1, const cv::Mat& descriptor2, std::vector<cv::DMatch>& matches)
{
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptor1, descriptor2, knnMatches, 2);
    
    matches.reserve(knnMatches.size());
    for(auto& match : knnMatches)
    {
        if(match[0].distance < match[1].distance * 0.7)
        {
            matches.push_back(match[0]);
        }
    }
}   