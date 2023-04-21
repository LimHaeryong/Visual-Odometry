#include "visual_odometry/LocalMap.h"

namespace VO
{
    LocalMap::LocalMap()
        : maxNumKeyFrames(10), lastKeyFrameId(0)
    {
    }

    LocalMap::~LocalMap()
    {
    }

    void LocalMap::addKeyFrame(const Frame &frame)
    {
        keyFrames[lastKeyFrameId++] = frame;
        if (keyFrames.size() > maxNumKeyFrames)
        {
            keyFrames.erase(keyFrames.begin());
        }
    }

    void LocalMap::optimize()
    {
    }

};