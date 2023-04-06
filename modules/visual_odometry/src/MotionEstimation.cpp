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
            matchedPoints3D_1.push_back(frame1.points3D[queryIdx]);
            matchedPoints2D_2.push_back(frame2.keyPoints[trainIdx]);
        }

        //std::cout << "matched point size : " << matchedPoints2D_2.size() << std::endl;
        cv::Mat rvec, tvec;
        try
        {
            cv::solvePnP(matchedPoints3D_1, matchedPoints2D_2, cameraMatrix, cv::Mat(), rvec, tvec);
        }
        catch (cv::Exception &e)
        {
            std::cerr << " solvePnP Error : " << e.what() << std::endl;
            frame2.relativePose = frame1.relativePose.clone();
            frame2.pose = frame2.relativePose*frame1.pose.clone();
            return -1;
        }

        double translationSum = 0.0;
        for(int i = 0; i < 3; i++)
        {
            translationSum += tvec.at<double>(i) * tvec.at<double>(i);
        }
        if(translationSum > 10.0)
        {
            frame2.relativePose = frame1.relativePose.clone();
            frame2.pose = frame2.relativePose*frame1.pose.clone();
            return -1;
        }
        // std::vector<cv::Vec6d> cameras(2);
        // cameras[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // cameras[1] = (rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2), tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        cv::Mat rotationMatrix;
        cv::Rodrigues(rvec, rotationMatrix);
        cv::Mat relativePose = cv::Mat::eye(4, 4, CV_64F);
        rotationMatrix.copyTo(relativePose(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(relativePose(cv::Rect(3, 0, 1, 3)));
        //std::cout << "relative pose : " << relativePose << std::endl;
        frame2.pose = frame1.pose.clone() * relativePose.clone();
        // ceres::Problem ba;

        // for (int i = 0; i < matches.size(); ++i)
        // {
        //     ceres::CostFunction *costFunc = ReprojectionError::create(matchedPoints2D_2[i], cameraMatrix.at<double>(0, 0), cv::Point2d(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2)));
        //     double* camera = (double*)(&(cameras[1]));
        //     double* X = (double*)(&(matchedPoints3D_1[i]));
        //     ba.AddResidualBlock(costFunc, NULL, camera, X);
        // }
        return 0;
    }

};