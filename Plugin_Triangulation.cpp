/*
    Plugin_Triangulation.cpp for mpDPFT
    @create a table for the special function KD and use plane patching as an approximation
    @authors Alexander Hue (PhD Project Code, version 1.0 2nd Apr 2018), Martin-Isbjoern Trappe
*/

#include "Plugin_Triangulation.h"
#include <vector>
#include <iostream>
#include <exception>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <set>
#include <cstdlib>
#include <ctime>
#include <random>
#include <omp.h>
#include "Plugin_KD.h"
#include "stdafx.h"
#include "alglibmisc.h"
#include "dataanalysis.h"


using namespace std;
using namespace alglib;

typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> fsec;
typedef vector<double> Point2D;
typedef vector<double> Point3D;
typedef vector<vector<double>> Triangle;

struct KDintegrationParams KDip;

Triangle ZeroTriangle {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};

const double PI = 3.14159265358979323846;
double ABSERR2 = 1.23456789;

auto localt0 = Time::now(), localt1 = Time::now();
//bool TimingQ = true;
bool TimingQ = true;
bool LocalTimingQ = true;
//bool LocalTimingQ = false;
int COUTQ = 0;
vector<int> COUNTER(100,0);
vector<double> ELAPSED(100,0.);

vector<kdtreerequestbuffer> VertexBUFFER(omp_get_max_threads());



// A U X I L L I A R Y    F U N C T I O N S

// initialize random number generators
void initialize_seeds(vector<mt19937>& engines){ random_device rd; for(auto& engine : engines) engine.seed(rd()); }
vector<mt19937> engines(omp_get_max_threads());
struct SeedInitializer { SeedInitializer() { initialize_seeds(engines); } } seedInitializer;

void StartLocalTimer(string descriptor, int coutQ){
	localt0 = Time::now();
	if(coutQ) cout << "StartTimer for " << descriptor << endl;
}

void EndLocalTimer(string descriptor, int coutQ){
	localt1 = Time::now();
	fsec duration = localt1 - localt0;
	if(coutQ) cout << "EndTimer for " << descriptor << ": " << to_string(duration.count()) << " seconds" << endl;
}

void ADDTIMING(int TimerTag){
	localt1 = Time::now();
	fsec duration = localt1 - localt0;
	ELAPSED[TimerTag] += static_cast<double>(duration.count());
}

Point3D triangle_centroid(Triangle triangle){
	// Find centroid
	Point3D centroid(2);
	centroid[0] = 0.3333333333333333*(triangle[0][0]+triangle[1][0]+triangle[2][0]);
	centroid[1] = 0.3333333333333333*(triangle[0][1]+triangle[1][1]+triangle[2][1]);
	// Patch centroid with plane equation of triangle
	return plane_patch(centroid, triangle);
}

bool trianglesOverlap(const Triangle& t1, const Triangle& t2) {
    return pointInsideTriangle(t1[0], t2) || pointInsideTriangle(t1[1], t2) || pointInsideTriangle(t1[2], t2) ||
           pointInsideTriangle(t2[0], t1) || pointInsideTriangle(t2[1], t1) || pointInsideTriangle(t2[2], t1);
}



// Function to check if a point is inside a triangle
bool pointInsideTriangle(const vector<double>& point, const vector<vector<double>>& triangle) {
    double x0 = triangle[0][0], y0 = triangle[0][1];
    double x1 = triangle[1][0], y1 = triangle[1][1];
    double x2 = triangle[2][0], y2 = triangle[2][1];
    double x = point[0], y = point[1];

    double A = 0.5 * (-y1 * x2 + y0 * (-x1 + x2) + x0 * (y1 - y2) + x1 * y2);
    double sign = A < 0.0 ? -1.0 : 1.0;
    double s = (y0 * x2 - x0 * y2 + (y2 - y0) * x + (x0 - x2) * y) * sign;
    double t = (x0 * y1 - y0 * x1 + (y0 - y1) * x + (x1 - x0) * y) * sign;

    return s > 0.0 && t > 0.0 && (s + t) < 2.0 * A * sign;
}

// Function to check if a triangle is contained within another triangle
bool triangleContained(const vector<vector<double>>& triangle, const vector<vector<double>>& other) {
    for(const auto& point : triangle) if(!pointInsideTriangle(point, other)) return false;
    return true;
}

// Function to remove triangles contained within any triangle from another vector
void removeContainedTriangles(vector<vector<vector<double>>>& TrianglesListToClean, const vector<vector<vector<double>>>& ReferenceTriangles) {
    TrianglesListToClean.erase(remove_if(TrianglesListToClean.begin(), TrianglesListToClean.end(), [&](const auto& triangle) {
        return any_of(ReferenceTriangles.begin(), ReferenceTriangles.end(), [&](const auto& other) {
            return triangleContained(triangle, other);
        });
    }), TrianglesListToClean.end());
}


// M A I N    R O U T I N E S

void GetVertextree(vector<Point3D> &Vertices, integer_1d_array &tags, real_2d_array &pointsArray, kdtree &Vertextree){
	if(TimingQ) StartLocalTimer("GetVertextree",COUTQ);
	int TimerTag = 5;
	convertToReal2DArray(Vertices,pointsArray);
	kdtreebuildtagged(pointsArray, tags, Vertices.size(), 2/*NX*/, 1/*NY*/, 2/*normtype*/, Vertextree);
	for(int b=0;b<VertexBUFFER.size();b++) kdtreecreaterequestbuffer(Vertextree, VertexBUFFER[b]);
	if(TimingQ){ EndLocalTimer("GetVertextree",COUTQ); COUNTER[TimerTag]++; ADDTIMING(TimerTag); }
}

vector<Triangle> FormTriangles(double threshold, vector<Point3D> &newpoints, kdtree &Vertextree){
	
	if(TimingQ) StartLocalTimer("FormTriangles",COUTQ);
	int TimerTag = 2;
	
	vector<Triangle> triangles(0);//A
	
	//for each new point, form triangle with nearest neighbors
	#pragma omp parallel for schedule(static)
	for(int i=0;i<newpoints.size();i++){
		Triangle t = {}; // Container for triangle t with vertices {a,b,c}
		int success = formTriangle(t, threshold, newpoints[i],Vertextree);
		#pragma omp critical
		{
			if(success==2) triangles.push_back(t);
			else if(success==0) cout << "FormTriangles: Warning !!! Did not find two nearest neighbours" << endl;
		}
    }
	
	if(TimingQ){ EndLocalTimer("FormTriangles",COUTQ); COUNTER[TimerTag] = newpoints.size(); ADDTIMING(TimerTag); }
	
	//unsorted list of triangles, will contain duplicates and redundancies
    return triangles;
}

int formTriangle(Triangle &t, const double threshold, const Point3D &newpoint, const kdtree &Vertextree){
	real_1d_array seed;
	seed.setcontent(2, &newpoint[0]);
	real_2d_array XY = "[[]]"; // Container for the points
	bool selfmatch = false; // false means the point cannot be the point itself
	int k = kdtreetsqueryknn(Vertextree, VertexBUFFER[omp_get_thread_num()], seed, 2, selfmatch); // Consistency check: should give 2 points, i.e. k = 2

	// If we found indeed two nearest neighbours
	if(k==2){// Put the results into XY
		kdtreetsqueryresultsxy(Vertextree, VertexBUFFER[omp_get_thread_num()], XY);
		t.push_back({XY[0][0],XY[0][1],XY[0][2]});
		t.push_back({XY[1][0],XY[1][1],XY[1][2]});
		t.push_back({seed[0],seed[1],newpoint[2]});
		sort(t.begin(), t.end());//sort vertices with ascending first component (which is seed[0]==A), i.e. t[0][0]<=t[1][0]<=t[2][0]
		Point2D v = {t[1][0]-t[0][0],t[1][1]-t[0][1]};//b-a
		Point2D w = {t[2][0]-t[0][0],t[2][1]-t[0][1]};//c-a
		double abscrossproduct = abs(v[0]*w[1]-v[1]*w[0]);
		if(abscrossproduct>threshold) return 2;//if w and v are not colinear
		else return 1;
	}
    else return 0;
}

Triangle FindTriangle(Point2D seed, vector<Triangle> &triangles, kdtree &Vertextree){//find a triangle that contains seed={A,B}
	
	int TimerTag = 0;
	if(LocalTimingQ && omp_get_thread_num()==0) StartLocalTimer("find triangle",COUTQ);
	
    if(!triangles.empty()){
		//First filter out good candidates for triangles to save time
		vector<Triangle> candidates(0);
		
		//Option1: brute force search
// 		for(int t=0;t<triangles.size();++t){
// 			if( (triangles[t][0][0] <= seed[0]) && (triangles[t][2][0] >= seed[0]) && (min({triangles[t][0][1], triangles[t][1][1], triangles[t][2][1]}) <= seed[1]) && (max({triangles[t][0][1], triangles[t][1][1], triangles[t][2][1]}) >= seed[1]) ){
// 				candidates.push_back(triangles[t]);
// 			}
// 		}
		
		//Option2: from Vertextree
		real_1d_array Seed;
		Seed.setcontent(2, &seed[0]);
		integer_1d_array tags = "[]"; // Container for the tags
		bool selfmatch = true;//since Seed may be one of the vertices
		int TargetNearestNeighbors = 10;//find at most TargetNearestNeighbors
		int k = kdtreetsqueryknn(Vertextree, VertexBUFFER[omp_get_thread_num()], Seed, TargetNearestNeighbors, selfmatch);
		if(k>0){
			//get tags of vertices
			kdtreetsqueryresultstags(Vertextree, VertexBUFFER[omp_get_thread_num()], tags);
			//get triangle indices
			set<int> I;
			for(int t=0;t<tags.length();++t){//tags[t]=i*3+r, with tags[t] the index of the vertex, i the index of the triangle, and r={0,1,2}
				int tag = (int)tags[t];
				int r = tag%3;
				int i = (tag-r)/3;
				I.insert(i);
				//if(omp_get_thread_num()==0) cout << "FindTriangle: tag = " << tag << " r=" << r << " i=" << i << endl << Triangulation_vec_to_str(seed) << endl << Triangulation_vec_to_str(triangles[i][0]) << " " << Triangulation_vec_to_str(triangles[i][1]) << " " << Triangulation_vec_to_str(triangles[i][2]) << endl; 
			}
			//get triangles
			for(auto i: I) candidates.push_back(triangles[i]);
		}
		
		//for both options: Verify if the seed point is in any of the good candidates (if any)
		for(int t=0;t<candidates.size();++t){
			if(InTriangleQ(seed,candidates[t])){
				if(LocalTimingQ && omp_get_thread_num()==0){ EndLocalTimer("triangle found!",COUTQ); COUNTER[TimerTag]++; ADDTIMING(TimerTag); }
				return candidates[t];
			}
		}		
	}

	if(LocalTimingQ && omp_get_thread_num()==0){ EndLocalTimer("no triangle found...",COUTQ); COUNTER[TimerTag]++; ADDTIMING(TimerTag); }
    return ZeroTriangle;
}

/*
Patch the z-value of 2D point using plane equation on the triangle

@param point: The point of interest
       triangle: The triangle to be patched over using plane equation
@return The 3D version of point of interest with patched value as the z-value
*/
Point3D plane_patch(Point2D point, Triangle triangle)
{
    // Calculate the normal vector to the plane formed by the triangle
    Point3D normal;
    Point3D v1;
    transform(triangle[0].begin(), triangle[0].end(), triangle[2].begin(), back_inserter(v1), minus<double>());
    Point3D v2;
    transform(triangle[1].begin(), triangle[1].end(), triangle[2].begin(), back_inserter(v2), minus<double>());
    normal.push_back(v1[1]*v2[2]-v1[2]*v2[1]);
    normal.push_back(v1[2]*v2[0]-v1[0]*v2[2]);
    normal.push_back(v1[0]*v2[1]-v1[1]*v2[0]);

    // constant of plane equation
    double d = dot(normal, triangle[0]);

    // Compute z value
    double point_z = (d - normal[0]*point[0] - normal[1]*point[1])/normal[2];
    point.push_back(point_z);
	
    return point;
}

bool GoodTriangleQ(int dim, Triangle triangle, double reltol, double abstol, const int NumChecks){// Check if a given triangle is a good triangle
	
	double LaxRelTol = reltol;//10.*reltol;
	
	Point3D testPoint;
	for(int i=0;i<NumChecks;i++){
		if(i==0) testPoint = triangle_centroid(triangle);
		else if(i<4){//edge centres
			int v = i; if(v==3) v = 0;
			Point3D a = triangle[i-1];
			Point3D b = triangle[v];
			//Point3D testPoint = {0.5*(b[0]-a[0]),0.5*(b[1]-a[1]),0.5*(a[2]+b[2])};
			Point3D testPoint = {0.5*(a[0]+b[0]),0.5*(a[1]+b[1]),0.5*(a[2]+b[2])};
		}
		else{//compute a random point within the triangle using barycentric coordinates
			Point3D a = triangle[0];
			Point3D b = triangle[1];
			Point3D c = triangle[2];
			uniform_real_distribution<double> dist(0.0, 1.0);
			double r1 = dist(engines[omp_get_thread_num()]);
			double r2 = dist(engines[omp_get_thread_num()]) * (1.-r1);
			testPoint = {\
				(1.-r1-r2)*a[0] + r1*b[0] + r2*c[0],\
				(1.-r1-r2)*a[1] + r1*b[1] + r2*c[1],\
				(1.-r1-r2)*a[2] + r1*b[2] + r2*c[2]};
		}
		double TrueFunctionValue = KD(dim,testPoint[0],testPoint[1],ABSERR2,reltol,KDip);
		if(RelativeDifference(testPoint[2],TrueFunctionValue)>LaxRelTol) return false;
	}
	
	return true;
}

/*
Check if a point is in a triangle

@param pnt: The point of interest
       triangle: The triangle of interest

@return The boolean value whether the point is in the triangle or not
*/
bool InTriangleQ(Point2D pnt, Triangle triangle)
{
    // Compute the vectors
    vector<double> v0 {{triangle[2][0]-triangle[0][0], triangle[2][1]-triangle[0][1]}};
    vector<double> v1 {{triangle[1][0]-triangle[0][0], triangle[1][1]-triangle[0][1]}};
    vector<double> v2 {{pnt[0]-triangle[0][0], pnt[1]-triangle[0][1]}};

    // Compute dot products
    double dot00 {dot(v0,v0)};
    double dot01 {dot(v0,v1)};
    double dot02 {dot(v0,v2)};
    double dot11 {dot(v1,v1)};
    double dot12 {dot(v1,v2)};

    // Compute barycentric coordinates
    double invDenom = 1 / (dot00*dot11 - dot01*dot01);
    double u = (dot11*dot02 - dot01*dot12) * invDenom;
    double v = (dot00*dot12 - dot01*dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

/*
Compute the value of the special function KD given a 2D point.
- Use the plane-equation approximated value if a good triangle exists.
- Otherwise, compute the exact value

@param seed: The point of interest
       points: The table of values
       GoodTriangles: The list of good GoodTriangles
       new_points: A box that contains newly added points
       new_triangles: A box that contains newly added triangles
@return The z-value of the point of interest. points and GoodTriangles are updated
        accordingly.
*/

//MIT:
struct ComputeKD GetTriangulatedFuncVal(int dim, Point2D seed, vector<Triangle> &GoodTriangles, double reltol, double abstol, kdtree &Vertextree){
	
	struct ComputeKD computekd;
	
	double FunctionValue;
	double LaxValidationFraction = 0.01;
	double LaxRelTol = reltol;//10.*reltol;//
	
	Triangle candidate = FindTriangle(seed, GoodTriangles, Vertextree);
	if(candidate == ZeroTriangle){//...seed is not within any good triangle
		computekd.goodTriangleQ = 0;
		if(LocalTimingQ && omp_get_thread_num()==0) StartLocalTimer("KD",COUTQ);
		int TimerTag = 1;
		FunctionValue = KD(dim,seed[0],seed[1],computekd.abserr,reltol,KDip);
		if(LocalTimingQ && omp_get_thread_num()==0){
			EndLocalTimer("KD",COUTQ);
			COUNTER[TimerTag]++;
			ADDTIMING(TimerTag);
		}
	}
	else{//...seed is within a good triangle
		computekd.goodTriangleQ = 1;
		double TriangulatedValue = plane_patch(seed, candidate)[2];
		uniform_real_distribution<double> dist(0.0, 1.0);
		if( dist(engines[omp_get_thread_num()]) < LaxValidationFraction ){
			FunctionValue = KD(dim,seed[0],seed[1],computekd.abserr,reltol,KDip);
			if(RelativeDifference(FunctionValue,TriangulatedValue)>LaxRelTol) computekd.goodTriangleQ = -1;
			else computekd.goodTriangleQ = 2;
		}
		else FunctionValue = TriangulatedValue;
		computekd.abserr = 1.;
	}
	computekd.ABres = {{seed[0],seed[1],FunctionValue}};
	
    return computekd;
}

void UpdateTriangulation(int dim, vector<Point3D> &newpoints, vector<Point3D> &Vertices, integer_1d_array &tags, vector<Triangle> &GoodTriangles, double MinArea, double reltol, double abstol, bool InitPhase, const int NumChecks, real_2d_array &pointsArray, string &TriangulationReport, kdtree &Vertextree){
	
	TriangulationReport = "#############   UpdateTriangulation: REPORT    #############\n";
	bool UpdateQ = false;
	int TC, NGT, TimerTag;
	double threshold = 1.0e-2;
	
	if(InitPhase && Vertices.size()>0){//build first kd-tree of (loaded!) Vertices and distribute to all workers
		GetVertextree(Vertices, tags, pointsArray, Vertextree);
	}
	else if(newpoints.size()>0 && Vertices.size()==0){//build first kd-tree of newpoints (no Vertices loaded) and distribute to all workers
		tags.setlength(newpoints.size());
		for(int i=0;i<newpoints.size();i++) tags[i] = i;
		GetVertextree(newpoints, tags, pointsArray, Vertextree);
		TriangulationReport += "KD.Vertextree built\n";
	}
	//Note: Removed search among old points, since it is almost never successful and creates huge data file
	
	if(newpoints.size()>0){//update triangulation and create new triangles only with new points (if available)
		vector<Triangle> triangle_candidates = FormTriangles(threshold, newpoints, Vertextree);
		//add to existing GoodTriangles and Vertices if applicable
		TC = triangle_candidates.size();
		vector<int> GoodLabels(TC,0);
		if(TimingQ) StartLocalTimer("UpdateTriangulation: LabelGoodTriangles");
		TimerTag = 3;
		//check triangle_candidates
		#pragma omp parallel for schedule(static)
		for(int t=0;t<TC;++t){
			if(TriangleArea(triangle_candidates[t])>MinArea){
				if(GoodTriangleQ(dim,triangle_candidates[t],reltol,abstol,NumChecks)) GoodLabels[t] = 1;
			}
		}
		if(TimingQ){ EndLocalTimer("UpdateTriangulation: LabelGoodTriangles"); COUNTER[TimerTag] = TC; ADDTIMING(TimerTag); }
		if(TimingQ) StartLocalTimer("UpdateTriangulation: CollectGoodTriangles");
		TimerTag = 4;
		NGT = accumulate(GoodLabels.begin(),GoodLabels.end(),0);
		for(int t=0;t<TC;++t){
			if(GoodLabels[t]==1){
				GoodTriangles.push_back(triangle_candidates[t]);
				for(int p=0;p<3;p++) Vertices.push_back(triangle_candidates[t][p]);
			}
		}
		//Note: It is futile to check for and remove candidates that are already contained entirely within any GoodTriangle, because newpoints are only those that are not within any GoodTriangle. Hence, a triangle formed with a newpoint cannot lie entirely within any GoodTriangle
		tags.setlength(Vertices.size());
		#pragma omp parallel for schedule(static)
		for(int t=0;t<Vertices.size();++t) tags[t] = t;
		if(TimingQ){ EndLocalTimer("UpdateTriangulation: CollectGoodTriangles"); COUNTER[TimerTag] = NGT; ADDTIMING(TimerTag); }
		//build new kd-tree of Vertices and distribute to all workers
		GetVertextree(Vertices, tags, pointsArray, Vertextree);
		UpdateQ = true;
	}
		
	
	if(newpoints.size()>0) TriangulationReport += "UpdateGoodTriangles: " + to_string(NGT) + " new good triangles found (out of " + to_string(TC) + " candidates)\n";
	vector<string> TimerTags = {"FindTriangle [@thread_id=0] ",\
		                        "KD [@thread_id=0]           ",\
		                        "FormTriangles               ",\
		                        "LabelGoodTriangles          ",\
		                        "CollectGoodTriangles        ",\
		                        "GetVertextree               "\
	};
	for(int t=0;t<TimerTags.size();++t){
		if(COUNTER[t]>0) TriangulationReport += "    ELAPSED time for TimerTag = " + to_string(t) + ": " + TimerTags[t] + " = " + to_string(ELAPSED[t]) + "	(" + to_string(ELAPSED[t]/((double)COUNTER[t])) + ") 		seconds in total (on average)		Counts = " + to_string(COUNTER[t]) + "\n";
	}
	
	COUNTER = vector<int>(100,0);
	ELAPSED = vector<double>(100,0.);
	
	if(UpdateQ) TriangulationReport += "Triangulation updated\n";
	else TriangulationReport += "Triangulation not updated\n";
}

bool GoodMergerQ(const Triangle &T1, const Triangle &T2, Triangle &mergedTriangle, const int dim, const double reltol, const double abstol, const int NumChecks){
	
	//assume that T1 and T2 are sorted already
	if(!trianglesOverlap(T1,T2)) return false;
	
	//create mergedTriangle with maximal area
	mergedTriangle = T1;
	int VertexSwaps = 0;
	if(T2[0][0]<T1[0][0]){ mergedTriangle[0] = T2[0]; VertexSwaps++; }
	if(T2[2][0]>T1[2][0]){ mergedTriangle[2] = T2[2]; VertexSwaps++; }
	double Area1 = TriangleArea(mergedTriangle);
	mergedTriangle[1] = T2[1]; VertexSwaps++;
	double Area2 = TriangleArea(mergedTriangle);
	if(Area2<Area1){
		mergedTriangle[1] = T1[1];
		VertexSwaps--;
	}
	
	//check if T2 already contained in T1 (i.e. T1 is the already the triangle with maximal area) or vice versa
	if(VertexSwaps==0 || VertexSwaps==3) return true;
	//check if mergedTriangle is a good triangle
	if(GoodTriangleQ(dim,mergedTriangle,reltol,abstol,NumChecks)) return true;
	return false;
}

