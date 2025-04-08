#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <limits>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <vector>

// const double PI = 3.14159265358979323846;
using namespace std;

struct KDintegrationParams {\
	int IntegrationArraySize = 10; double KDthreshold = 1.0e-7; int iMAX = 100; int kQAWO = 100; double minB = 1.0e-4; double maxK = 1.0e+6; int stepM = 1000; int num_outer_threads = 1; int num_inner_threads = 1; bool contourQ = false;
};

double phifunc(double s, double A, double thirdBcube);
void newtonfunc(double s, double A, double Bcube, double thirdBcube, double &phi, double &derphi);
void Gets1s2(double phi, double s0, double A, double Bcube, double thirdBcube, double &s1, double &s2);//MIT
void s12(double x, double A, double Bcube, double thirdBcube,double s0p, double &s1, double &s2, double s1ref, double s2ref);
double sDiff(double x, void * p);
double gx(double x, void * p);
double phi0(double A, double B, double &s0p);
double phipp(double s, double B);
double KDapr(int D, double A, double B);
double KDasymp(int D, double A, double B);
double KD(int D, double A, double B, double &abserr, double reltolx, KDintegrationParams &KDip);

double dvdu(double u, double v, double A, double B, double gamma);
bool is_close(double a, double b, double atol = 1.e-8);
double fcontour(double u, void *params);
double fcontour_gaussian(double u, void *params);
double KD_contour(int D, double A, double B, KDintegrationParams &KDip);
std::tuple<double, double, double, double> run_contour(int D, double A, double B, std::vector<double> &fc);
void writeVectorsToFile(const std::vector<double>& vec1, const std::vector<double>& vec2, const std::vector<double>& vec3, const std::vector<double>& vec4, const std::vector<double>& vec5, const std::vector<std::vector<double>> mat, const std::string& filename);
double log_uniform(double min, double max, std::mt19937& gen);
int run_range(int fnum, double scale, double amin, double amax, double bmin, double bmax);
int run_full_range(void);
