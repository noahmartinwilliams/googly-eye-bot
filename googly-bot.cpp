#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

const double pi = 3.14159265358979;
int main(int argc, char **argv)
{
	srandom(atoi(argv[1]));
	String cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye.xml";

	CascadeClassifier cascade;
	cascade.load(cascade_name);

	Mat image, gray_image;
	image = imread("image.jpeg", CV_LOAD_IMAGE_COLOR);

	vector<Rect> eyes;

	cvtColor(image, gray_image, CV_BGR2GRAY);
	equalizeHist(gray_image, gray_image);

	cascade.detectMultiScale(gray_image, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < eyes.size(); i++) {
		Point center(eyes[i].x + eyes[i].width * 0.5, eyes[i].y + eyes[i].height * 0.5);
		double radius=sqrt((0.25*eyes[i].width*eyes[i].width)+(0.25*eyes[i].height*eyes[i].height));

		double theta = (double) random() ;
		double dist = (((double) (random() % 100)) / 100.0) * radius;
		Point pupil_center((eyes[i].x + eyes[i].width * 0.5) + dist * cos(theta), (eyes[i].y + eyes[i].height * 0.5) + dist * sin(theta));

		circle(image, center, radius, Scalar(255, 255, 255), -1);
		circle(image, pupil_center, radius * 0.3, Scalar(0, 0, 0), -1);
	}
	imwrite("output.png", image);

	return 0;
}
