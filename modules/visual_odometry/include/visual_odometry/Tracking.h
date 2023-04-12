#ifndef TRACKING_H_
#define TRACKING_H_

#include <deque>

#include "visual_odometry/Type.h"

namespace VO
{
    class Tracker
    {
    public:
        Tracker(uint windowSize_);
        virtual ~Tracker();

        void feedFrame(Frame& frame);

    private:
        void refinePose(Frame& frame);

        std::deque<Frame> frames;
        uint windowSize;
    };

};
#endif // TRACKING_H_