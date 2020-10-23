// Face Recognition C++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "tensorflow/c/c_api.h"

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    Net net = readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel");
    Mat image = imread("404902_2745393247130_736455659_n.jpg", IMREAD_COLOR);

    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int imageHeight = image.rows;
    int imageWidth = image.cols;

    Mat imageResized;
    resize(image, imageResized, Size(300, 300));

    Mat blob = blobFromImage(imageResized, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), true);

    net.setInput(blob, "data");
    Mat detections = net.forward("detection_out");

    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.5)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * imageWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * imageHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * imageWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * imageHeight);

            rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2, 4);
        }
    }

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    return 0;
}
