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
        std::vector<cv::Point2d> matchedKeyPointRight;
        matchedKeyPointRight.reserve(matchSize);
        frame.keyPoints.reserve(matchSize);
        frame.descriptors.reserve(matchSize);
        int max_diff = 0;
        for (int i = 0; i < matchSize; ++i)
        {
            int queryIdx = matches[i].queryIdx;
            cv::Point2d left = keyPointLeft[queryIdx].pt;
            cv::Point2d right = keyPointRight[matches[i].trainIdx].pt;
            if(std::abs(left.y - right.y) > 5)
            {
                continue;
            }
            frame.keyPoints.push_back(left);
            frame.descriptors.push_back(descriptorLeft.row(queryIdx));
            matchedKeyPointRight.push_back(right);
        }

        cv::Mat triangulatedPoints;
        cv::triangulatePoints(projectionLeft, projectionRight, frame.keyPoints, matchedKeyPointRight, triangulatedPoints);

        frame.points3D.reserve(matchSize);
        for(int i = 0; i < matchSize; ++i)
        {
            cv::Point3d point3D;
            double w = triangulatedPoints.at<double>(3, i);
            point3D.x = triangulatedPoints.at<double>(0, i) / w;
            point3D.y = triangulatedPoints.at<double>(1, i) / w;
            point3D.z = triangulatedPoints.at<double>(2, i) / w;
            frame.points3D.push_back(point3D);
        }

        frame.pose = cv::Mat::eye(4, 4, CV_64F);
        frame.relativePose = cv::Mat::eye(4, 4, CV_64F);
        return frame;
    }
};
