#include "visual_odometry/MotionEstimation.h"

namespace VO
{
    MotionEstimation::MotionEstimation(const std::string &calibPath)
    {
        feature = Feature();
        std::ifstream ifs(calibPath);
        std::string line;
        double value;
        cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

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
                if (col == 3)
                {
                    continue;
                }
                cameraMatrix.at<double>(row, col) = value;
            }
        }
    }

    MotionEstimation::~MotionEstimation()
    {
    }

    int MotionEstimation::motionEstimate(const Frame &frame1, Frame &frame2)
    {

        std::vector<cv::DMatch> matches;
        feature.match(frame1.descriptors, frame2.descriptors, matches);
        std::size_t matchSize = matches.size();

        std::vector<cv::Point3d> matchedPoints3D_1;
        std::vector<cv::Point2d> matchedPoints2D_2;
        matchedPoints3D_1.reserve(matchSize);
        matchedPoints2D_2.reserve(matchSize);

        for (int i = 0; i < matches.size(); i++)
        {
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx;
            matchedPoints3D_1.push_back(frame1.points3D[queryIdx]);
            matchedPoints2D_2.push_back(frame2.keyPoints[trainIdx]);
        }

        cv::Mat rvec, tvec, inliers;
        try
        {
            cv::solvePnPRansac(matchedPoints3D_1, matchedPoints2D_2, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 4.0f, 0.99, inliers);
        }
        catch (cv::Exception &e)
        {
            std::cerr << " solvePnP Error : " << e.what() << std::endl;
            frame2.pose = frame1.pose.clone() * frame2.relativePose;
            return -2;
        }

        size_t numInliers = inliers.rows;

        std::vector<cv::Point3d> inlierPoints3D(numInliers);
        std::vector<cv::Point2d> inlierPoints2D(numInliers);
        std::vector<cv::DMatch> inlierMatches(numInliers);
        
        for (int i = 0; i < numInliers; ++i)
        {
            int idx = inliers.at<int>(i);
            inlierPoints3D[i] = matchedPoints3D_1[idx];
            inlierPoints2D[i] = matchedPoints2D_2[idx];
            inlierMatches[i] = matches[idx];
        }
        cv::solvePnPRefineLM(inlierPoints3D, inlierPoints2D, cameraMatrix, cv::Mat(), rvec, tvec);

        double translationSum = 0.0;
        for (int i = 0; i < 3; i++)
        {
            translationSum += tvec.at<double>(i) * tvec.at<double>(i);
        }
        if (translationSum > 100.0)
        {
            frame2.pose = frame1.pose.clone() * frame2.relativePose;
            return -1;
        }

        cv::Mat rotationMatrix;
        cv::Rodrigues(rvec, rotationMatrix);
        double trace = cv::trace(rotationMatrix.t())[0];
        if (trace < 0.0)
        {
            frame2.pose = frame1.pose.clone() * frame2.relativePose;
            return -1;
        }
        cv::Mat relativePose = cv::Mat::eye(4, 4, CV_64F);
        rotationMatrix = rotationMatrix.t();
        tvec = -rotationMatrix * tvec;
        rotationMatrix.copyTo(relativePose(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(relativePose(cv::Rect(3, 0, 1, 3)));
        relativePose.copyTo(frame2.relativePose);
        frame2.pose = frame1.pose.clone() * frame2.relativePose;
        frame2.matchesWithKeyFrames.push_back(std::move(inlierMatches));
        if (translationSum < 0.01 && trace > 2.96)
        {
            return -1;
        }

        return 0;
    }

};