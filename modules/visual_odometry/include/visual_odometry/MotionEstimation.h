#ifndef MOTION_ESTIMATION_H_
#define MOTION_ESTIMATION_H_

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"

namespace VO
{
    class MotionEstimation
    {
    public:
        MotionEstimation(const std::string& calibPath);
        virtual ~MotionEstimation();

        int motionEstimate(const Frame& frame1, Frame& frame2);
    private:
        Feature feature;
        cv::Mat cameraMatrix;
    };


};

#endif // MOTION_ESTIMATION_H_