#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"

namespace VO
{
    class Triangulation
    {
    public:
        Triangulation(const std::string& calibPath);
        virtual ~Triangulation();

        Frame triangulate(const cv::Mat& imageLeft, const cv::Mat& imageRight);

    private:
        Feature feature;
        cv::Mat projectionLeft;
        cv::Mat projectionRight;

    };
};

#endif // TRIANGULATION_H_