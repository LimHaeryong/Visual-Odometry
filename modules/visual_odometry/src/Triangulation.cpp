#include "visual_odometry/Triangulation.h"

namespace VO
{
    Triangulation::Triangulation(const std::string &calibPath)
    {
        feature = Feature();
        std::ifstream ifs(calibPath);
        std::string line;
        double value;
        projectionLeft = cv::Mat::zeros(3, 4, CV_64F);
        projectionRight = cv::Mat::zeros(3, 4, CV_64F);
        if (!ifs.is_open())
        {
            std::cout << "cannot open calibPath : " << calibPath << std::endl;
        }
        std::getline(ifs, line);
        std::stringstream ss(line.substr(4));
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                ss >> value;
                projectionLeft.at<double>(row, col) = value;
            }
        }
        std::getline(ifs, line);
        ss.clear();
        ss.str(line.substr(4));
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                ss >> value;
                projectionRight.at<double>(row, col) = value;
            }
        }
        ifs.close();

        cv::Mat K1, R1, t, K2, R2;
        cv::decomposeProjectionMatrix(projectionLeft, K1, R1, t);
        cv::decomposeProjectionMatrix(projectionRight, K2, R2, t);

        cv::Mat translation;
        translation = K1.inv() * projectionLeft(cv::Rect(3, 0, 1, 3)) - K2.inv() * projectionRight(cv::Rect(3, 0, 1, 3));

        baseline = cv::norm(cv::Point3d(translation));
    }

    Triangulation::~Triangulation()
    {
    }

    Frame Triangulation::triangulate(const cv::Mat &imageLeft, const cv::Mat &imageRight)
    {
        Frame frame;
        std::vector<cv::KeyPoint> keyPointLeft, keyPointRight;
        cv::Mat descriptorLeft, descriptorRight;
        feature.detectAndCompute(imageLeft, keyPointLeft, descriptorLeft);
        feature.detectAndCompute(imageRight, keyPointRight, descriptorRight);

        std::vector<cv::DMatch> matches;
        feature.match(descriptorLeft, descriptorRight, matches);
        int matchSize = matches.size();
        std::vector<cv::Point2d> matchedKeyPointsLeft, matchedKeyPointsRight;
        cv::Mat matchedDescriptorsLeft;
        matchedKeyPointsLeft.reserve(matchSize);
        matchedKeyPointsRight.reserve(matchSize);
        matchedDescriptorsLeft.reserve(matchSize);


        for (int i = 0; i < matchSize; ++i)
        {
            int queryIdx = matches[i].queryIdx;
            cv::Point2d leftPoint = keyPointLeft[queryIdx].pt;
            cv::Point2d rightPoint = keyPointRight[matches[i].trainIdx].pt;
            if(std::abs(leftPoint.y - rightPoint.y) > 10.0)
            {
                continue;
            }
            matchedDescriptorsLeft.push_back(descriptorLeft.row(queryIdx));
            matchedKeyPointsLeft.push_back(leftPoint);
            matchedKeyPointsRight.push_back(rightPoint);
        }

        cv::Mat triangulatedPoints;
        cv::triangulatePoints(projectionLeft, projectionRight, matchedKeyPointsLeft, matchedKeyPointsRight, triangulatedPoints);

        frame.points3D.reserve(matchSize);
        frame.keyPoints.reserve(matchSize);
        frame.descriptors.reserve(matchSize);
        for(int i = 0; i < triangulatedPoints.cols; ++i)
        {
            cv::Point3d point3D;
            double w = triangulatedPoints.at<double>(3, i);
            point3D.x = triangulatedPoints.at<double>(0, i) / w;
            point3D.y = triangulatedPoints.at<double>(1, i) / w;
            point3D.z = triangulatedPoints.at<double>(2, i) / w;
            if(cv::norm(point3D) > baseline * 150.0)
            {
                continue;
            }
            frame.keyPoints.push_back(matchedKeyPointsLeft[i]);
            frame.descriptors.push_back(matchedDescriptorsLeft.row(i));
            frame.points3D.push_back(point3D);
        }
        
        frame.pose = cv::Mat::eye(4, 4, CV_64F);
        frame.relativePose = cv::Mat::eye(4, 4, CV_64F);

        return frame;
    }
};
