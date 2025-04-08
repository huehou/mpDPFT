#include <vector>
#include <random>
#include <string>
#include <Eigen/Dense>
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

#ifndef MPDPFT_HEADER_ONEPEXDFT
#define MPDPFT_HEADER_ONEPEXDFT
struct exDFTstruct{\
  vector<int> settings; int L; double CompensationFactor = 1.; int System = 0; vector<double> Abundances; double RelAcc; double UNITenergy; double UNITlength; int FuncEvals = 0; int CGDgr = 0; vector<double> OccNum; vector<double> OccNumAngles; vector<double> Phases; vector<double> Unitaries; vector<double> ExtraPhases; double Eint = 0.; double Etot = 0.; vector<vector<int>> QuantumNumbers; vector<vector<double>> Orb; vector<vector<double>> rho1p; vector<vector<double>> rho1pIm; double *ptrHint; vector<double> params; mt19937_64 MTGEN; uniform_real_distribution<double> RNpos; vector<double> ZeroVec; vector<int> Optimizers; double SHdt; int SHmaxCount; double SHConvergenceCrit; bool SHrandomSeedQ = true; MatrixXd RandOrthMat; string control = ""; vector<vector<double>> QuantNumRestrictIndexVectors; vector<string> LevelNames; vector<vector<double>> MonitorMatrix; vector<double> E1pLoaded; vector<vector<double>> CMatrixLoaded; double HintIndexThreshold = 0.; vector<int> FreeIndices; vector<double> UnitariesFlag; vector<double> Iabcd;
};
#endif

