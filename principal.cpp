#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <sstream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <chrono>
#include <iterator>
#include <algorithm>
using namespace cv;
using namespace std;
using namespace std::chrono;
namespace fs = std::__fs::filesystem;

double momentosHu[7];

double distanciaEuclidea(double mHu[7])
{
  double baseDatos[] = {0.159161, 1.33867e-08, 6.25236e-09, 9.46803e-14, -2.45198e-25, 1.069e-17, -2.29054e-24};
  double d = 0.0;
  for (int i = 0; i < 7; i++)
  {
    d += pow(mHu[i] - baseDatos[i], 2);
  }
  return sqrt(d);
}

std::string substring(std::string const &s)
{
  std::string::size_type pos = s.find('_');
  if (pos != std::string::npos)
  {
    return s.substr(0, pos);
  }
  else
  {
    return s;
  }
}

int main(int argc, char *argv[])
{
  namedWindow("Imagen", WINDOW_AUTOSIZE);
  namedWindow("Gray", WINDOW_AUTOSIZE);
  namedWindow("Binaria", WINDOW_AUTOSIZE);

  vector<string> listaMomentosHU = {};
  string path = "./ShapeDataset";

  // indice de imagenes aleatorias
  int numberOfImages = 34533;
  int todasImagenes[numberOfImages];
  for (int i = 0; i < numberOfImages; i++)
  {
    todasImagenes[i] = i;
  }

  int numeroImagenesEntrenamientoYTest = 1700;

  int counter = 0;
  for (const auto &entry : fs::directory_iterator(path))
  {
    if (counter == numeroImagenesEntrenamientoYTest)
      break;
    counter += 1;

    Mat colorImage = imread(entry.path());
    Mat binaryImage;
    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);
    int backgroudColor = (int)grayImage.at<uchar>(0, 0);

    // Crear imagen binaria
    threshold(grayImage, binaryImage, backgroudColor, backgroudColor, THRESH_BINARY);
    for (int i = 0; i < colorImage.rows; i++)
    {
      for (int j = 0; j < colorImage.cols; j++)
      {
        int currentPixel = grayImage.at<uchar>(i, j);
        if (currentPixel == backgroudColor)
        {
          binaryImage.at<uchar>(i, j) = 0;
        }
        else
        {
          binaryImage.at<uchar>(i, j) = 255;
        }
      }
    }

    Moments momentos = moments(binaryImage, true);
    HuMoments(momentos, momentosHu);

    string momentosHUString = "";
    for (int i = 0; i < 7; i++)
    {
      momentosHUString = momentosHUString + std::to_string(momentosHu[i]) + ";";
    }
    string nombreClase = substring(entry.path().filename());
    char delimeter('_');
    momentosHUString = momentosHUString + nombreClase;
    listaMomentosHU.push_back(momentosHUString);
  }

  // double distancia = distanciaEuclidea(momentosHu);
  // if (distancia < 0.001)
  // {
  //   cout << "Es círculo " << endl;
  // }

  ofstream myFile("entrenamiento.csv");
  for (int i = 0; i < numeroImagenesEntrenamientoYTest; i++)
  {
    myFile << listaMomentosHU.at(i) << "\n";
  }
  myFile.close();

  return 0;
}