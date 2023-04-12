#include "visual_odometry/Tracking.h"

namespace VO
{
    Tracker::Tracker(uint windowSize_)
        : windowSize(windowSize_)
    {
    }

    Tracker::~Tracker()
    {
    }

    void Tracker::feedFrame(Frame &frame)
    {
        refinePose(frame);




        frames.push_front(frame);
        if(frames.size() > windowSize)
        {
            frames.pop_back();
        }

    }


    void Tracker::refinePose(Frame &frame)
    {
        
    }

};