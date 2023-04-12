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
        std::vector<cv::Point3d> matchedPoints3D_1;
        std::vector<cv::Point2d> matchedPoints2D_2;

        for (int i = 0; i < matches.size(); i++)
        {
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx;
            double distance3D = cv::norm(frame1.points3D[queryIdx] - frame2.points3D[trainIdx]);
            if (distance3D > 100.0)
            {
                continue;
            }
            matchedPoints3D_1.push_back(frame1.points3D[queryIdx]);
            matchedPoints2D_2.push_back(frame2.keyPoints[trainIdx]);
            frame2.unmatchedIndices.erase(trainIdx);
        }

        cv::Mat rvec, tvec, inliers;
        try
        {
            cv::solvePnPRansac(matchedPoints3D_1, matchedPoints2D_2, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 4.0f, 0.99, inliers);
            std::vector<cv::Point3d> inlierPoints3D(inliers.rows);
            std::vector<cv::Point2d> inlierPoints2D(inliers.rows);

            for(int i = 0; i < inliers.rows; ++i)
            {
                int idx = inliers.at<int>(i);
                inlierPoints3D[i] = matchedPoints3D_1[idx];
                inlierPoints2D[i] = matchedPoints2D_2[idx];
            }
            cv::solvePnPRefineLM(inlierPoints3D, inlierPoints2D, cameraMatrix, cv::Mat(), rvec, tvec);
        }
        catch (cv::Exception &e)
        {
            std::cerr << " solvePnP Error : " << e.what() << std::endl;
            //frame2.relativePose = frame1.relativePose.clone();
            frame2.pose = frame1.pose.clone() * frame2.relativePose;
            return -2;
        }

        double translationSum = 0.0;
        for (int i = 0; i < 3; i++)
        {
            translationSum += tvec.at<double>(i) * tvec.at<double>(i);
        }
        if (translationSum > 100.0)
        {
            //frame2.relativePose = frame1.relativePose.clone();
            frame2.pose = frame1.pose.clone() * frame2.relativePose;
            return -1;
        }
        
        cv::Mat rotationMatrix;
        cv::Rodrigues(rvec, rotationMatrix);
        double trace = cv::trace(rotationMatrix.t())[0];
        if (trace < 0.0)
        {
            //frame2.relativePose = frame1.relativePose.clone();
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
        return 0;
    }

};