#include "visual_odometry/Triangulation.h"

namespace VO
{
    Triangulation::Triangulation(const std::string &calibPath)
    {
        feature = Feature();
        std::ifstream ifs(calibPath);
        std::string line;
        float value;
        projectionLeft = cv::Mat::zeros(3, 4, CV_32F);
        projectionRight = cv::Mat::zeros(3, 4, CV_32F);
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
                projectionLeft.at<float>(row, col) = value;
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
                projectionRight.at<float>(row, col) = value;
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

        std::vector<cv::Point2f> matchedKeyPointRight;
        matchedKeyPointRight.reserve(matchSize);
        frame.keyPoints.reserve(matchSize);
        frame.descriptors.reserve(matchSize);
        for (int i = 0; i < matchSize; ++i)
        {
            int queryIdx = matches[i].queryIdx;
            frame.keyPoints.push_back(keyPointLeft[queryIdx].pt);
            frame.descriptors.push_back(descriptorLeft.row(queryIdx));
            matchedKeyPointRight.push_back(keyPointRight[matches[i].trainIdx].pt);
        }

        cv::Mat triangulatedPoints;
        cv::triangulatePoints(projectionLeft, projectionRight, frame.keyPoints, matchedKeyPointRight, triangulatedPoints);

        frame.points3D.reserve(matchSize);
        for(int i = 0; i < matchSize; ++i)
        {
            cv::Point3f point3D;
            float w = triangulatedPoints.at<float>(3, i);
            point3D.x = triangulatedPoints.at<float>(0, i) / w;
            point3D.y = triangulatedPoints.at<float>(1, i) / w;
            point3D.z = triangulatedPoints.at<float>(2, i) / w;
            frame.points3D.push_back(point3D);
        }

        return frame;
    }
};
