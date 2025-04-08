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
//#include <math.h>//possible conflict with cmath
#include <algorithm>
#include <omp.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

double Orbital(int lambda, vector<double>position, exDFTstruct &ex);
void GetQuantumNumbers(exDFTstruct &ex);
double Energy1pEx(int lambda, exDFTstruct &ex);
void LoadE1p_X2C(exDFTstruct &ex);
void LoadCMatrix_X2C(exDFTstruct &ex);
vector<double> InitializeOccNum(bool noninteractingQ, exDFTstruct &ex);
void NearestProperOccNum(vector<double> &occ, exDFTstruct &ex);
void LoadOptOccNum(exDFTstruct &ex);
bool MonitorEx(int ErrorCode, vector<double> &occ, vector<vector<double>> &rho1p, exDFTstruct &ex);
bool rho1pConsistencyCheck(vector<vector<double>> &rho1p, vector<double> &occ, exDFTstruct &ex);
void GetRho1p(vector<vector<double>> &rho1p, vector<double> occ, exDFTstruct &ex);
void TransformRho1p(vector<vector<double>> &rho1p, vector<vector<double>> &rho1pIm, vector<double> &unitaries, vector<double> &extraphases, exDFTstruct &ex);
bool varianceConvergedQ(vector<double> Vec, int testlength, double crit);
MatrixXd orthogonal_matrix(bool randomSeedQ, exDFTstruct &ex);
MatrixXd rotation_matrix(bool randomSeedQ, exDFTstruct &ex);
bool TestN(vector<double> &occ, exDFTstruct &ex);

double H_int(int i, int j, int k, int l);
inline long double S(int j, int G, int lambda, int alpha){return (long double)(-(2*G+1)*(lambda-j))/(long double)((j+1)*(alpha+j+1)*2);}
inline long double lamb_up(int lambda, int alpha, long double x){return x*sqrt(1.+(long double)(alpha)/(long double)(lambda+1));}
long double T(int l0,int alpha,int alpha1,int l3,long double s00);
double fact_ratio(int b, int a, int N);
double AsymL(int n,int alpha,double x);
double alph_up(double lambda, double alpha, double alpha1, double x);
double integral_div(int l0, int l1, int l2, int l3,double a,double b,double subdiv,int Ns,int Na);
double integral_div2(int l0, int l1, int l2, int l3,double a,double b,double subdiv,int Ns,int Na);
double inf_integral(int i,int j,int k,int l,double b,double tail,double crit,double subdiv,double small,int Ns,int Na,double crit1,double small1,double T0,double Est);
double inf_integral2(int i,int j,int k,int l,double b,double tail,double crit,double subdiv,double small,int Ns,int Na,double crit1,double small1,double T0,double Est);
double JonahIntegral(int i,int j,int k,int l,double a,double b,double crit,double subdiv,double small,int Ns,int Na,double T0,double Est);
double integral2(int i,int j,int k,int l,double a,double b,double crit,double subdiv,double small,int Ns,int Na,double T0,double Est);
double GetPartialIntegral(int i,int j,int k,int l,double a, double b,int MaxIter, int iter, double prevInt, double relCrit,double absCrit,double NumSubDivisions,int Ns,int Na, exDFTstruct &ex);
double GetTotalIntegral(int i,int j,int k,int l,double a,double b,int MaxIter,int Iter,double relCrit,double absCrit,double NumSubDivisions,int Ns,int Na,exDFTstruct &ex);
void testHintIntegral2(exDFTstruct &ex);
void testHintIntegral(exDFTstruct &ex);
void testGetRho(exDFTstruct &ex);
void testrho1pConsistency(exDFTstruct &ex);
void GetHint(exDFTstruct &ex);
double HarmonicInteractionElement(int i, int j, int k, int l);
double HIE(int a, int m, int n);
void exportHint(string filename, exDFTstruct &ex);

double MinimizeEint(vector<vector<double>> &rho1p, exDFTstruct &ex);
double Eint_1pExDFT(vector<double> var, exDFTstruct &ex);
double Etot_1pExDFT(vector<double> var, exDFTstruct &ex);
double Eint_1pExDFT_CombinedMin(vector<double> var, vector<vector<double>> &rho1p, vector<vector<double>> &rho1pIm, exDFTstruct &ex);
double Etot_1pExDFT_CombinedMin(vector<double> AllAngles, exDFTstruct &ex);
void EXprint(int printQ, string str, exDFTstruct &ex);

inline vector<double> OccNumToAngles(vector<double> OccNum){ vector<double> Angles(OccNum.size()); for(int l=0;l<OccNum.size();l++) Angles[l] = acos(max(0.,min(2.,OccNum[l]))-1.); return Angles; }
inline vector<double> AnglesToOccNum(vector<double> Angles, exDFTstruct &ex){ vector<double> OccNum(Angles.size()); for(int l=0;l<Angles.size();l++) OccNum[l] = 1.+cos(Angles[l]); NearestProperOccNum(OccNum,ex); return OccNum; }
