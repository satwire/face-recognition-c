// Face Recognition C++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "tensorflow/c/c_api.h"

using namespace std;
using namespace cv;

int main()
{
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    Mat image;
    image = imread("opencv-logo-small.png", IMREAD_COLOR);

    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
