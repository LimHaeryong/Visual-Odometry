#include "visual_odometry/Triangulation.h"

VO::Triangulation::Triangulation(const std::string &calibPath)
{
    feature = Feature();

    std::ifstream ifs(calibPath);
    std::string line;
    float value;
    projectionLeft = cv::Mat::zeros(3, 4, CV_32F);
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
            projectionLeft.at<float>(row, col) = value;
        }
    }

    std::getline(ifs, line);
    ss.clear();
    ss.str(line.substr(4));
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            ss >> value;
            projectionRight.at<float>(row, col) = value;
        }
    }
    ifs.close();
        


}

VO::Triangulation::~Triangulation()
{
}
