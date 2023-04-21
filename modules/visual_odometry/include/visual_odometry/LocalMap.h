#ifndef LOCALMAP_H_
#define LOCALMAP_H_

#include <map>

#include "ceres/ceres.h"
#include "opencv2/opencv.hpp"

#include "visual_odometry/Type.h"

namespace VO
{
    class LocalMap
    {
    public:
        LocalMap();
        virtual ~LocalMap();
        void addKeyFrame(const Frame& frame);
        const Frame& getCurrentKeyFrame();
        void optimize();
    private:
        
        std::map<uint, Frame> keyFrames;
        const uint maxNumKeyFrames;
        uint lastKeyFrameId;
    };
};

#endif // LOCALMAP_H_