#include <vector>
#include <iostream>
#include <cmath>

#include "visual_odometry/Feature.h"

int main()
{
    cv::Mat imageLeft = cv::imread("../resources/00/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRight = cv::imread("../resources/00/image_1/000000.png", cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

    std::vector<cv::KeyPoint> keyPointLeft, keyPointLeft_, keyPointRight, keyPointRight_;
    cv::Mat descriptorLeft, descriptorRight;

    cv::Mat PLeft = (cv::Mat_<float>(3, 4) << 718.856f, 0.0f, 607.192f, 0.0f,
                     0.0f, 718.856f, 185.215f, 0.0f,
                     0.0f, 0.0f, 1.0f, 0.0f);

    cv::Mat PRight = (cv::Mat_<float>(3, 4) << 718.856f, 0.0f, 607.192f, -386.144f,
                      0.0f, 718.856f, 185.215f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f);
    

    feature->detect(imageLeft, keyPointLeft_);
    feature->detect(imageRight, keyPointRight_);
    std::cout << "size : " << keyPointLeft_.size() << " " << keyPointRight_.size() << std::endl;
    for (const auto &kp : keyPointLeft_)
    {
        if (kp.response > 0.005)
        {
            keyPointLeft.push_back(kp);
        }
    }

    for (const auto &kp : keyPointRight_)
    {
        if (kp.response > 0.005)
        {
            keyPointRight.push_back(kp);
        }
    }
    std::cout << "size : " << keyPointLeft.size() << " " << keyPointRight.size() << std::endl;
    feature->compute(imageLeft, keyPointLeft, descriptorLeft);
    feature->compute(imageRight, keyPointRight, descriptorRight);

    // // feature->detectAndCompute(imageLeft, cv::Mat(), keyPointLeft, descriptorLeft);
    // // feature->detectAndCompute(imageRight, cv::Mat(), keyPointRight, descriptorRight);

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(descriptorLeft, descriptorRight, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    goodMatches.reserve(matches.size());

    for (const auto &match : matches)
    {
        if (match[0].distance / match[1].distance < 0.7)
        {
            goodMatches.push_back(match[0]);
        }
    }

    std::vector<cv::Point2f> matchedPointsLeft, matchedPointsRight;
    matchedPointsLeft.reserve(goodMatches.size());
    matchedPointsRight.reserve(goodMatches.size());

    for (const auto &match : goodMatches)
    {
        matchedPointsLeft.push_back(keyPointLeft[match.queryIdx].pt);
        matchedPointsRight.push_back(keyPointLeft[match.trainIdx].pt);
    }

    cv::Mat drawMatches;
    cv::drawMatches(imageLeft, keyPointLeft, imageRight, keyPointRight, goodMatches, drawMatches);

    cv::imshow("match", drawMatches);
    cv::Mat mat3D;
    cv::triangulatePoints(PLeft, PRight, matchedPointsLeft, matchedPointsRight, mat3D);

    std::vector<cv::Point3f> points3D;
    points3D.reserve(mat3D.cols);
    for (int i = 0; i < mat3D.cols; ++i)
    {
        cv::Point3f point;
        float w = mat3D.at<float>(3, i);
        point.x = mat3D.at<float>(0, i) / w;
        point.y = mat3D.at<float>(1, i) / w;
        point.z = mat3D.at<float>(2, i) / w;
        points3D.push_back(point);

        mat3D.at<float>(0, i) /= w;
        mat3D.at<float>(1, i) /= w;
        mat3D.at<float>(2, i) /= w;
        mat3D.at<float>(3, i) = 1.0f;
    }

    cv::Mat projectedPoints = PLeft * mat3D;
    float rep_err = 0.0f;
    for (int i = 0; i < projectedPoints.cols; ++i)
    {
        cv::Point point;
        float z = projectedPoints.at<float>(2, i);
        point.x = projectedPoints.at<float>(0, i) / z;
        point.y = projectedPoints.at<float>(1, i) / z;
        cv::circle(imageLeft, point, 10, cv::Scalar(255, 0, 0));
        std::cout << "point :  " << matchedPointsLeft[i] << std::endl;
        std::cout << "reproj : " << point << std::endl;
        rep_err += (matchedPointsLeft[i].x - point.x) * (matchedPointsLeft[i].x - point.x) + (matchedPointsLeft[i].y - point.y) * (matchedPointsLeft[i].y - point.y);
    }

    std::cout << "rep_err : " << std::pow(rep_err, 0.5) / static_cast<float>(matchedPointsLeft.size()) << std::endl;

    for (const auto &point : matchedPointsLeft)
    {
        cv::drawMarker(imageLeft, point, cv::Scalar(255, 0, 0));
    }
    cv::imshow("left", imageLeft);

    cv::waitKey(0);

    for (const auto &kp : keyPointLeft)
    {
        std::cout << kp.response << std::endl;
    }

    return 0;
}