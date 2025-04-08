#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <exception>
#include <algorithm>
#include <functional>
#include <numeric>
#include <omp.h>
#include "stdafx.h"
#include "alglibmisc.h"

// const double PI = 3.14159265358979323846;
using namespace std;
using namespace alglib;

// Type Definitions
typedef vector<double> Point2D;
typedef vector<double> Point3D;
typedef vector<vector<double>> Triangle;

struct ComputeKD{\
  Point3D ABres; int goodTriangleQ; double abserr = -1.;
};

// A U X I L L I A R Y    F U N C T I O N S
inline double RelativeDifference(const double a, const double b){ double res = 0.; if(a!=0.) res = abs(b/a-1.); else if(b!=0.) res = abs(a/b-1.); return res; }
inline double dot(const vector<double>& vec1, const vector<double>& vec2){ return inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0); }
inline double TriangleArea(const Triangle T){ return 0.5*abs( (T[0][0]-T[2][0])*(T[1][1]-T[0][1]) - (T[0][0]-T[1][0])*(T[2][1]-T[0][1]) ); }
inline void convertToReal2DArray(vector<Point3D> &points, real_2d_array &array) {
    array.setlength(points.size(), 3);
	#pragma omp parallel for schedule(static)
    for(int i=0;i<points.size();i++){
        array[i][0] = points[i][0];
        array[i][1] = points[i][1];
        array[i][2] = points[i][2];
    }
}
template <typename T> inline std::string Triangulation_to_string_with_precision(T a_value, int prec){ std::ostringstream out; out << std::setprecision(prec) << a_value; return out.str(); }
template <typename T> inline std::string Triangulation_vec_to_str(vector<T> vec){ std::ostringstream oss; for(int i=0;i<vec.size();i++) oss << Triangulation_to_string_with_precision(vec[i],16) << " "; return oss.str(); }
void StartLocalTimer(string descriptor, int coutQ = 1);
void EndLocalTimer(string descriptor, int coutQ = 1);
void ADDTIMING(int TimerTag);
Point3D triangle_centroid(Triangle triangle);
// Check if any of the vertices of one triangle are inside the other triangle
bool trianglesOverlap(const Triangle& t1, const Triangle& t2);
bool pointInsideTriangle(const std::vector<double>& p, const std::vector<std::vector<double>>& triangle);
bool triangleContained(const std::vector<std::vector<double>>& tri, const std::vector<std::vector<double>>& container);
void removeContainedTriangles(std::vector<std::vector<std::vector<double>>>& triangles, const std::vector<std::vector<std::vector<double>>>& containers);

// M A I N    R O U T I N E S
void GetVertextree(vector<Point3D> &Vertices, integer_1d_array &tags, real_2d_array &pointsArray, kdtree &Vertextree);
vector<Triangle> FormTriangles(double threshold, vector<Point3D> &newpoints, kdtree &Vertextree);
int formTriangle(Triangle &t, const double threshold, const Point3D &newpoint, const kdtree &Vertextree);
Triangle FindTriangle(Point2D seed, vector<Triangle> &triangles, kdtree &Vertextree);
Point3D plane_patch(Point2D point, Triangle triangle);
bool GoodTriangleQ(int dim, Triangle triangle, double reltol, double abstol, const int NumChecks);
bool InTriangleQ(Point2D pnt, Triangle triangle);
struct ComputeKD GetTriangulatedFuncVal(int dim, Point2D seed, vector<Triangle>& triangles, double reltol, double abstol, kdtree &Vertextree);
void UpdateTriangulation(int dim, vector<Point3D> &newpoints, vector<Point3D> &Vertices, integer_1d_array &tags, vector<Triangle> &GoodTriangles, double sMinArea, double reltol, double abstol, bool InitPhase, const int NumChecks, real_2d_array &pointsArray, string &TriangulationReport, kdtree &Vertextree);
bool GoodMergerQ(const Triangle &T1, const Triangle &T2, Triangle &mergedTriangle, const int dim, const double reltol, const double abstol, const int NumChecks);
