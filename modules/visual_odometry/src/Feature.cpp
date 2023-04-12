#include "visual_odometry/Feature.h"

namespace VO
{

    Feature::Feature()
    {
        const int nFeatures = 2000;
        const float scaleFactor = 1.2f;
        const int nLevels = 8;
        featureExtractor = cv::ORB::create(nFeatures, scaleFactor, nLevels);
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        imageSize = cv::Size(1241, 376);
        scaleFactors = std::vector<float>(nLevels, 1.0f);
        for(uint i = 1; i < nLevels; ++i)
        {
            scaleFactors[i] = scaleFactors[i-1] * scaleFactor;
        }
        focalLength = 718.85;
        hammingDistThresh = 75;
    }
    Feature::~Feature()
    {
    }

    void Feature::detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors)
    {
        featureExtractor->detectAndCompute(image, cv::Mat(), keyPoints, descriptors);
    }

    void Feature::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
    {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
        matches.reserve(knnMatches.size());
        for (auto &match : knnMatches)
        {
            if (match[0].distance < match[1].distance * 0.7)
            {
                matches.push_back(match[0]);
            }
        }
    }


    void Feature::stereoMatch(const std::vector<cv::KeyPoint> &keyPointsLeft, const cv::Mat &descriptorsLeft, const std::vector<cv::KeyPoint> &keyPointsRight, const cv::Mat &descriptorsRight, std::vector<cv::DMatch> &matches)
    {
        std::vector<std::vector<uint>> indicesRightInRow = getRightKeypointIndicesInRow(2.0, keyPointsRight);
        matches.reserve(keyPointsLeft.size());
        
        #pragma omp parallel for 
        for(uint idxLeft = 0; idxLeft < keyPointsLeft.size(); ++idxLeft)
        {
            const auto& keyPointLeft = keyPointsLeft[idxLeft];
            const float yLeft = keyPointLeft.pt.y;
            const float xLeft = keyPointLeft.pt.x;

            const std::vector<uint> indicesRight = indicesRightInRow[static_cast<int>(yLeft)];
            if(indicesRight.empty())
            {
                continue;
            }

            const float xRightMin = xLeft - focalLength;
            const float xRightMax = xLeft;

            uint bestIdxRight = 0;
            uint bestHammingDist = hammingDistThresh;
            cv::Mat descriptorLeft = descriptorsLeft.row(idxLeft);
            std::tie(bestIdxRight, bestHammingDist) = findClosestKeyPoint(keyPointLeft, descriptorLeft, indicesRight, keyPointsRight, descriptorsRight, xRightMin, xRightMax);

            if(bestHammingDist >= hammingDistThresh)
            {
                continue;
            }

            cv::DMatch match;
            match.distance = static_cast<float>(bestHammingDist);
            match.queryIdx = idxLeft;
            match.trainIdx = bestIdxRight;

            #pragma omp critical
            {
                matches.push_back(std::move(match));
            }
            
        }
    }

    std::vector<std::vector<uint>> Feature::getRightKeypointIndicesInRow(double margin, const std::vector<cv::KeyPoint> &keyPointsRight)
    {
        std::vector<std::vector<uint>> indicesRightInRow(imageSize.height, std::vector<uint>());

        for(uint idx = 0; idx < keyPointsRight.size(); ++idx)
        {
            const auto& keyPoint = keyPointsRight[idx];
            float y = keyPoint.pt.y;
            float r = margin * scaleFactors[keyPoint.octave];
            int yMax = cvCeil(y + r);
            int yMin = cvFloor(y - r);

            for(int row = yMin; row <= yMax; ++row)
            {
                indicesRightInRow[row].push_back(idx);
            }
        }
        return indicesRightInRow;
    }

    std::pair<uint, uint> Feature::findClosestKeyPoint(const cv::KeyPoint& keyPointLeft, const cv::Mat &descriptorLeft, const std::vector<uint>& indicesRight, const std::vector<cv::KeyPoint> &keyPointsRight, const cv::Mat& descriptorsRight, const float xRightMin, const float xRightMax)
    {
        uint bestIdxRight = 0;
        uint bestHammingDist = hammingDistThresh;

        const int scaleLeft = keyPointLeft.octave;

        for(const uint idxRight : indicesRight)
        {
            const cv::KeyPoint& keyPointRight = keyPointsRight[idxRight];
            const int scaleRight = keyPointRight.octave;
            if(std::abs(scaleRight - scaleLeft) > 1)
            {
                continue;
            }

            const float xRight = keyPointRight.pt.x;
            if(xRight < xRightMin || xRight > xRightMax)
            {
                continue;
            }

            const cv::Mat &descriptorRight = descriptorsRight.row(idxRight);
            const uint hammingDist = getHammingDist(descriptorLeft, descriptorRight);

            if(hammingDist < bestHammingDist)
            {
                bestIdxRight = idxRight;
                bestHammingDist = hammingDist;
            }

            
        }

        return std::make_pair(bestIdxRight, bestHammingDist);
    }

    uint Feature::getHammingDist(const cv::Mat& descriptor1, const cv::Mat& descriptor2)
    {
        std::vector<uchar> bits1, bits2;
        descriptor1.convertTo(bits1, CV_8U);
        descriptor2.convertTo(bits2, CV_8U);

        uint hammingDist = 0;
        for(uint i = 0; i < 32; ++i)
        {
            hammingDist += cv::countNonZero(bits1[i]^bits2[i]);
        }
        return hammingDist;
    }
};
