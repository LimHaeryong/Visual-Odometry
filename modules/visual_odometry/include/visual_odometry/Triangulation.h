#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_

#include <fstream>
#include <sstream>
#include <iostream>

#include "visual_odometry/Type.h"
#include "visual_odometry/Feature.h"

namespace VO
{
    class Triangulation
    {
    public:
        Triangulation(const std::string& calibPath);
        virtual ~Triangulation();

    private:
        Feature feature;
        cv::Mat projectionLeft;
        cv::Mat projectionRight;

    };
};

#endif // TRIANGULATION_H_