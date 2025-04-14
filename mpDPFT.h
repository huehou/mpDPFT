#include "stdafx.h"
#include "statistics.h"
#include "specialfunctions.h"
#include "solvers.h"
#include "optimization.h"
#include "linalg.h"
#include "interpolation.h"
#include "integration.h"
#include "fasttransforms.h"
#include "diffequations.h"
#include "dataanalysis.h"
#include "ap.h"
#include "alglibmisc.h"
#include "alglibinternal.h"
#include <sstream>
#include <functional>
#include <stdlib.h>
#include <fftw3.h>
#include <omp.h>
#include <gsl/gsl_sf_gamma.h>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "sys/sysinfo.h"

using namespace std;
using namespace alglib;
using namespace Eigen;

//specify processor architecture for ALGLIB
#define AE_CPU = AE_INTEL

struct DynDFTestruct {\
  int mode; bool InitializationPhase = true; vector<vector<double>> n0; vector<vector<double>> nf; vector<vector<vector<double>>> v; vector<vector<vector<double>>> g; vector<int> SnapshotID; vector<vector<double>> VectorScale;
};

struct KDparams {\
  int UseTriangulation = 0; int UpdateTriangulation = 0; int IntermediateCleanUp = 0; int ReevaluateTriangulation = 0; double Result = 1.23456789e+300; double UpperLimit = 1.23456789; double prefactor; double alpha; double beta; int IntLimitCount; double epsIntLimits; double BaseScale; double bStepScale = 1.23456789; double RelCrit; double relCrit; double absCrit; double epsTweeze = -1.0; int subdiv = 4; int printQ = 1; double minDelta = 1.0e-13; double distinguishabilityThreshold = 1.0e-13; int minIter = 3; int maxIter = 100000; double FuncEvals; int Threads = 1; bool Error = false; double betaThreshold = 1.0e-16; bool CompareKD = false; vector<vector<double>> NewPoints; int OrigNumGoodTriangles; vector<vector<vector<double>>> GoodTriangles; vector<vector<double>> Vertices; string GoodTrianglesFilename = ""; string GoodTrianglesFileNameForGnuplot = ""; bool TriangulationUpdatedQ = false; bool TriangulationInitializedQ = false; kdtree Vertextree; real_2d_array pointsArray; integer_1d_array tags; int CoarseGridSize; vector<int> CoarseIndices; string TriangulationReport; int NumChecks = 1; double MergerRatioThreshold = 0.2; double RemovalFraction = 0.1; double RetainSurplusFraction = 0.5; double ExportCleanDensity = 0.; int quality = 1; double MinArea = 0.;
};

struct MLPerceptron {\
  multilayerperceptron mlp; mt19937_64 MTGEN; uniform_real_distribution<double> RNpos; int TrainingDataType; int NumSamples; int NumTrainingSamples; int NumValidationSamples; double TestSetFraction; int NumInputVars; int NumHiddenLayers; int NumWeights; int DefaultNumRestarts; int MaxNumRepeats; int TargetCounter; double TargetSigmaFraction; double WeightChangeStopCriterion; ae_int_t epochs; double DefaultWeightDecay; int print; vector<vector<double>> Seeds; vector<double> par; double shift; double spread; bool ReproductionQ;
};

struct Correlationstruct {\
  int Measure; double POW; double PercentageOfOutliers;
};

struct pulaystruct {\
  vector<double>coeffVec; vector<vector<double>> alphaVec; /*vector<vector<vector<double>>> RMat;*/ vector<vector<double>> ResidualMatrix; vector<vector<vector<double>>> ResidualMatrixSep; vector<vector<vector<double>>> ninVec; vector<vector<vector<double>>> noutVec; vector<vector<vector<double>>> RVec; vector<vector<double>> residualVec; vector<vector<vector<double>>> ResidualVec; vector<vector<vector<vector<double>>>> RVecFFT; vector<vector<vector<double>>> residualVecFFT; vector<vector<vector<vector<double>>>> ResidualVecFFT; int scopeOriginal; int mixer; int scope/*5*/; double weight/* = 50.*/; vector<vector<double>> CoeffVec; bool MomentumSpaceQ = false; bool EagerExecutionQ = false;
};

struct pseudopotstruct {\
  int atom; int valence; spline1dinterpolant PPspline; double rMax;
};

struct GDstruct {\
  vector<double> x; double f;
};

struct flagstruct{\
  int FFTW = 0; int RefData = 0; int CurlyA = 0; int PolyLogs = 0; int Del = 0; int Export = 0; int Gyk = 0; int LIBXC = 0; int ForestGeo = 0; int InitializeDensities = 1; int GetOrbitals = 1; int ChemistryEnvironment = 1; int SCO = 0; int KD = 0; int ExportCubeFile = 0; int vW = 0; int periodic = 0; int AdiabaticEnergy = 0;
};

struct reportstruct {\
  bool activeQ = false;
  int SwarmParticle;
  vector<double> muVec;
  vector<vector<double>> V;
};

struct datastruct {\
  //input parameters:
  int System; int TaskType; int TaskParameterCount; vector<double> TaskHyperBox; int DIM; int Symmetry; int Units; int steps; double edge; int S; vector<double> Abundances; double RelAcc; int K; vector<double> Resources; int DensityExpression; vector<double> mpp; vector<double> Mpp; vector<vector<double>> MPP/*ToDo*/; vector<double> tauVec; vector<double> tVec; vector<double> TVec; vector<int> Environments; double Wall; double Noise; int InterpolVQ; vector<int> Interactions; vector<double> muVec; double DeltamuModifier; vector<double> Mixer; vector<double> thetaVec; double SCcriterion; int maxSCcount; int method; int MovieQ; vector<int> Schedule; vector<int> EkinTypes; double alpha; double beta; double gamma; double gammaH; double degeneracy; vector<int> regularize; int Print;\
  //derived input parameters:
  vector<string> InputParameterNames; double DDIM; int EdgeLength; int GridSize; vector<double> frame; vector<double> kframe; vector<vector<double>> VecAt; vector<vector<double>> kVecAt; vector<double> Norm2kVecAt; int CentreIndex; double area; double rW; double wall; double WallArea; double Deltax; double Deltax2; double k0; double Deltak; vector<double> taupVec; vector<double> ApVec; vector<double> thetaVecOriginal; double RelAccOriginal; double gammaOriginal; double NoiseOriginal; vector<int> HDI; vector<bool> PulayQ; vector<vector<double>> tauVecMatrix; vector<vector<double>> LimitingResource; int HartreeType; vector<vector<double>> MppVec; double AccumulatedAbundances; double U = 1.;\
  //loaded input parameters
  vector<vector<double>> NonUniformResourceDensities; vector<double> ReferenceData; int NumberOfNuclei; vector<int> NucleiTypes; vector<int> ValenceCharges; vector<vector<double>> NucleiPositions; vector<pseudopotstruct> PPlist;\
  //hard-coded input parameters:
  double rpow; double flatEnv = 0.; double stretchfactor = 1.; double AllElectronParam; double zeta; vector<int> ResourceIndexList; vector<vector<double>> Consumption; vector<double> Competition; vector<double> RepMut; vector<double> Amensalism; vector<double> fitness; double sigma; vector<double> sigmaVec; double kappa; double RegularizationThreshold; int DelFieldMethod; int libxcPolarization; double incrementalV; double incrementalVOriginal; double InternalAcc; int MaxIteration; double AutomaticTratio; double ConvolutionSigma; int TaperOff = 1; int RegularizeCoulombSingularity = 1; int AllElectronPP = 1;\
  //streams, memory, handling:
  bool ABORT = false; bool MonitorFlag = false; bool MONITOR = false; ostringstream controlfile; stringstream txtout; stringstream CutDataStream; fftw_plan FFTWPLANFORWARDPARALLEL; fftw_plan FFTWPLANBACKWARDPARALLEL; fftw_complex *inFFTW; fftw_complex *outFFTW; vector<vector<double>> ComplexField; vector<vector<double>> TmpComplexField; vector<double> TmpField1; vector<double> TmpField2; vector<double> TmpField3; vector<vector<double>> TMPField2; vector<vector<double>> MovieStorage; flagstruct FLAGS; int InitializationInProgress = 1;\
  ostringstream epslatexSTART; ostringstream square; ostringstream rectangle; ostringstream epslatexLineStyles; ostringstream epslatexEND; ostringstream TeX1; ostringstream TeX2; ostringstream IncludeGraphicsSTART; ostringstream includeGraphics; ostringstream include3Graphics; ostringstream IncludeGraphicsEND; int SwarmParticle; ostringstream IntermediateEnergies; vector<string> EnergyNames;\
  //support & data fields and special functions:
  vector<double> Lattice; vector<double> kLattice; vector<double> lattice; vector<int> SymmetryMask; vector<int> IndexList; int SplineType; int StrideLevel; int MaxStrideLevel; double InterpolAcc; double UpdateSymMaskFraction; vector<spline1dinterpolant> TmpSpline1D; vector<spline2dinterpolant> TmpSpline2D; vector<spline3dinterpolant> TmpSpline3D; vector<double> Stridelattice; int stride; vector<vector<double>> Env; vector<vector<double>> Den; vector<vector<double>> V; double* LIBXC_rho = NULL; double* LIBXC_sigma = NULL; double* LIBXC_exc = NULL; double* LIBXC_vxc = NULL; double* LIBXC_vsigma = NULL; vector<vector<spline2dinterpolant>> FieldInterpolations; vector<double> SmoothRandomField; vector<vector<double>> Dx; vector<vector<double>> Dy; vector<vector<double>> Dz; vector<vector<double>> D2x; vector<vector<double>> D2y; vector<vector<double>> D2z; vector<vector<double>> GradSquared; vector<vector<double>> Laplacian; vector<vector<double>> SqrtDenDx; vector<vector<double>> SqrtDenDy; vector<vector<double>> SqrtDenDz; vector<vector<double>> SqrtDenGradSquared; spline1dinterpolant PolyLogm1half; spline1dinterpolant PolyLog3half; spline1dinterpolant PolyLog5half; spline1dinterpolant PolyLog1half; spline1dinterpolant PolyLogm3half; spline1dinterpolant AuxDawsonErfc; spline1dinterpolant CurlyA; vector<double> AverageDensity; /*spline1dinterpolant HG1F2_a; spline1dinterpolant HG1F2_b;*/ spline1dinterpolant Gykspline1D; spline1dinterpolant Gykspline2D; spline1dinterpolant Gykspline3D; vector<vector<double>> eta; KDparams KD; \
  //auxilliary variables:
  vector<double> TmpAbundances; vector<vector<double>> OldDen; vector<vector<double>> OldV; vector<double> OldmuVec; vector<double> PreviousmuVec; vector<double> OldAbundances; int SCcount = 0; int FrameCount = 0; int ExitSCloop = 0; int ExitSCcount = 0; double CC; vector<double> CChistory; int ompThreads; int FourierFilterPercentage; int BottomCutoff; int TopCutoff; mt19937_64 MTGEN; uniform_real_distribution<double> RN; uniform_real_distribution<double> RNpos; vector<double> AdmixSignal; vector<double> OldAdmixSignal; double OldCC; double FigureOfMerit; vector<vector<vector<int>>> nAiTevals; vector<double> OldnAiTEVALS; vector<double> booleCoefficients = {{7.,32.,12.,32.,7.}}; pulaystruct Pulay; int Index; int Species; GDstruct GradDesc; int FocalSpecies; double Stride; double StrideOriginal; double StrideMax; double y; int yCount; int ySteps; int yStepsOriginal; int yStepsMax; bool TrappedQ = false; bool BPreductionInProgress = false; vector<double> VEC; double CalculateEnergyQ = true; double lambda; int stall = 0; vector<vector<double>> TestMat; string AuxString = "";\
  //output variables
   time_t Timing0; time_t Timing1; time_t Timing2; int PrintSC = 1; int errorCount; int warningCount; double OutPrecision; vector<double> OutCut; double CutLine; vector<vector<double>> CutPositions; vector<vector<vector<double>>> minmaxfi; double GlobalTimeDenMax = 0.; vector<double> DenMin; vector<double> DenMax; double GlobalDenMin; double GlobalDenMax; double Global3DcolumnDenMin; double Global3DcolumnDenMax; double ScaledDenMin; double ScaledDenMax; vector<double> EnvMin; vector<double> EnvMax; vector<double> EnvAv; vector<double> VMin; vector<double> VMax; vector<double> ResMin; vector<double> ResMax; double RefMin; double RefMax; vector<double> DispersalEnergies; vector<double> EnvironmentalEnergies; vector<double> InteractionEnergies; vector<vector<double>> AuxEkin; vector<double> AlternativeDispersalEnergies; vector<vector<double>> AuxEint; double NucleiEnergy; double Etot; double EInt; vector<double> AvailableResources; vector<double> AverageResourceSquared; vector<double> ConsumedResources; int LinMix = 0; int PulMix = 0;
};

struct taskstruct {\
  vector<datastruct> dataset; ofstream ControlTaskFile; ostringstream controltaskfile; int Type; vector<int> count; int ParameterCount; int AxesPoints; vector<vector<double>> HyperBox; vector<double> VEC = {0.}; double Aux = 0.; vector<double> AuxVec; bool CGinProgress = false; bool ABORT = false; vector<vector<double>> ParameterExplorationContainer; string auxout = ""; bool lastQ = false; vector<double> ConjGradParam = {{0.},{0.},{0.},{0.}}; vector<vector<double>> V; exDFTstruct ex; int fitQ; vector<vector<double>> refData; vector<vector<double>> FOMrefData; vector<double> AbsResL; vector<double> AbsResR; vector<double> AbsRes; int fomType; int refDataType = 0; int S; vector<double> muVec; vector<reportstruct> Report; vector<int> mask; double maskratio = -1.; Correlationstruct Corr; bool PositiveTauQ; int TotalNumFitParams; int NumGlobalParam; int NumSigParam; int NumEnvParam; int NumIntParam; bool CoarseGrain; vector<double> Abundances; vector<double> abundances; bool optInProgress = false; bool GuessMuVecQ = false; MLPerceptron NN; DynDFTestruct DynDFTe;
};


// P R I M A R Y    R O U T I N E S
void GetData(taskstruct &task, datastruct &data);
void GetGlobalMinimum_muVec(taskstruct &task);
void GetGlobalMinimum_Abundances(taskstruct &task);
// taskstruct GetGlobalMinimum_Abundances(taskstruct &task);
void GetAuxilliaryFit(taskstruct &task);
void GetGlobalMinimum_Abundances_AnnealedGradientDescent(taskstruct &task);
vector<vector<double>> GetTaskHyperBox(datastruct &tmpdata);
void GetInputParameters(taskstruct &task, datastruct &data);
//void ExpandInput(int doubleQ, vector<int> &IntVec, vector<double> &DoubleVec, int size);
inline void ExpandInput(vector<auto> &vec, auto val, int size){ vec.resize(size); fill(vec.begin(),vec.end(),val); }
void PrintInputParameters(datastruct &data);
void ActivateDefaultParameters(datastruct &data);
void InitializeHardCodedParameters(datastruct &data, taskstruct &task);
void GetEnv(int s,datastruct &data);
void AddWall(vector<double> &Field, datastruct &data);
void GetResources(datastruct &data);
void GetDensity(int s,datastruct &data);
void getVint(int InteractionType, int ss, datastruct &data);
void GetSCOgradients(vector<double> &infield, vector<double> &outfield, datastruct &data);
void CompetitionIntegrand(int EnergyQ, int a, int ss, datastruct &data);
void ResourceIntegrand(int EnergyQ, int ss, datastruct &data);
void HartreeRepulsion(int EnergyQ, int ss, datastruct &data);
void DipolarInteractionPos(int EnergyQ, int ss, datastruct &data);
void RenormalizedContact(int EnergyQ, int ss, datastruct &data);
void DeltaEkin2D3Dcrossover(int EnergyQ, int ss, datastruct &data);
void Quasi2Dcontact(int EnergyQ, int ss, datastruct &data);
void DipolarInteraction(int EnergyQ, int ss, datastruct &data);
void DFTeAlignment(int EnergyQ, int ss, datastruct &data);
void EkinDynDFTe(int EnergyQ, int ss, datastruct &data);
int IndexOfVec(vector<double> &Vec, datastruct &data);
void DiracExchange(int EnergyQ, int ss, datastruct &data);
void GombasCorrelation(int EnergyQ, int ss, datastruct &data);
void VWNCorrelation(int EnergyQ, int ss, datastruct &data);
void getLIBXC(int EnergyQ, int ss, int neg_func_id, int polarization, datastruct &data);
void MomentumContactInteractionFiniteT(int ss, datastruct &data);

// S P E C I A L   P R E P A R A T I O N S
void SP_fitness(int index, datastruct &data);
void LoadNuclei(datastruct &data);
void LoadPseudoPotentials(datastruct &data);
double AllElectronPP(int Z, double r, datastruct &data);
void GetGuidingVectorField(int s, datastruct &data, taskstruct &task);
vector<int> NeighborhoodIndices(int GridIndex, bool NNN, datastruct &data);

// S E C O N D A R Y    R O U T I N E S
void GetDensities(taskstruct &task, datastruct &data);
void GetTFinspireddensity(int s, int DenExpr, datastruct &data);
void GetSmoothdensity(int s, datastruct &data);
double getDenEnAt(int EnergyQ, int s, int i, datastruct &data);
void GetnAiT(int EnergyQ, int s, datastruct &data);
void AdaptnAiT(datastruct &data);
double AdaptiveBoole_nAiT(int EnergyQ, int s, int maxIter, double a, double b, double prevInt, int iter, double RTPNA, double Compare, double prefactor,double U,double ar,int i,double prefactor2,datastruct &data);
double nAiTintegrand(int EnergyQ, int s, double x,double prefactor,double U,double ar,int i,double prefactor2,datastruct &data);
void Getn3pFFT(int s, datastruct &data);
void Getn3p(int s, datastruct &data);
void Getn3pT(int s, datastruct &data);
void n3pxyFFTPARALLEL(vector<double> &NewField, double prefactor, double yEpsilon, int BisectionCount, double PreviousIntegral, double &AccumulatedPartNum, double yStart, double yEnd, int ySteps, double crit, datastruct &data);
void Adaptn3pT(datastruct &data);
double GetEpsilonNewField(int order, double PreviousResult,double yEpsilon,double crit, vector<double> &InputField, datastruct &data);
spline1dinterpolant gfuncSpline(int order, double yEpsilon, double crit, double maxFk, datastruct &data);
double GetRadialgInt(int order, double yEpsilon, double kappa, double maxFk, int BisectionCount,double xStart, double xEnd, double PreviousIntegral, double AccumulatedIntegral, int BP, double crit, datastruct &data);
void FillComplexField(vector<double> &InputField, datastruct &data);
double getn7AIintegral(double a, double b, double STEPS, double prevres, int iter, int MAXITER, int &NumMAXITER, double dist, double LO, double NLO, datastruct &data);
double GetID(const double tauThreshold, const int s, const int i, const double r, const int ip, const double rp, datastruct &data);
void Getn7(int s, datastruct &data);
void GetSCOdensity(int s, datastruct &data);
void GetTFdensity(int s, int DenExpr, datastruct &data);
void GetOptOccNum(taskstruct &task, datastruct &data);
double PiotrBeta(int derivative, double eta);
void ImposeNoise(int fields, datastruct &data);
void PreparationsForSCdensity(datastruct &data);
void GetFieldInterpolations(datastruct &data);
void EnforceConstraints(datastruct &data);
void GetTmpAbundances(datastruct &data);
void CheckConvergence(datastruct &data);
double RelDev(int s, vector<double> &den1, vector<double> &den2, datastruct &data);
void StoreCurrentIteration(datastruct &data);
void UpdateV(datastruct &data);
void GetVint(int s, datastruct &data);
void Intercept(datastruct &data);
void UpdateSymmetryMask(datastruct &data);
void AdmixDensities(datastruct &data);
void AdmixPotentials(datastruct &data);
int PulayMixer(datastruct &data);
void AdaptPulayMixer(datastruct &data);
void AdaptLinearMixer(datastruct &data);
double PulayScalarProduct(int s, int i, int j, datastruct &data);
void InjectSchedule(datastruct &data);
void PrepareNextIteration(taskstruct &task, datastruct &data);
void PrintIntermediateEnergies(datastruct &data);
void GetEnergy(datastruct &data);
double GetSCOEnergy(vector<double> field/*pass by copy!*/, datastruct &data);
void GetAuxEkin(datastruct &data);
void GetAuxEint(datastruct &data);
double GetDispersalEnergy(int s, datastruct &data);
double GetE1Ai3Dat(datastruct &data);
void E1Ai3DxIntegrand(double x, double xminusa, double bminusx, double &y, void *ptr);
double GetEnvironmentalEnergy(int s, datastruct &data);
double GetInteractionEnergy(int InteractionType, datastruct &data);
double GetNucleiEnergy(datastruct &data);
void FourierFilter(vector<double> &Field, datastruct &data);
void ExpandSymmetry(int ExpandDensityQ, vector<double> &Field, datastruct &data);

// G L O B A L   I N I T I A L I Z A T I O N S
inline void SetOmpThreads(taskstruct &task,datastruct &data){ if(task.Type==44) data.ompThreads = 1; else data.ompThreads = (int)omp_get_max_threads(); /*cout << "SetOmpThreads to " << data.ompThreads;*/ }
void RetrieveTask(datastruct &data,taskstruct &task);
void SetupSCO(datastruct &data);
void SanityChecks(datastruct &data);
void Misc(datastruct &data);
void InitializeGlobalFFTWplans(datastruct &data);
void AllocateTMPmemory(datastruct &data);
void InitializeEnvironments(datastruct &data);
void AllocateDensityMemory(datastruct &data);
void SetupMovieStorage(datastruct &data);
void ImportV(vector<vector<double>> &ImportField, datastruct &data);
void LoadV(datastruct &data, taskstruct &task);
void UpdateApVec(int init, datastruct &data);
void InitializeRandomNumberGenerator(datastruct &data);
void ProcessAuxilliaryData(datastruct &data);
void ReadCubeFile(datastruct &data);
void GetSmoothRandomField(int FromFileQ, double scale, datastruct &data);
void GetSpecialFunctions(datastruct &data);
void InitializeKD(datastruct &data);
void GetOrb(datastruct &data);
spline1dinterpolant GetSpline1D(string FileName, bool sortedQ);
spline1dinterpolant GetSpecialFunction(string TabFunc_Name);
spline1dinterpolant GetPolyLogm3half(void);
spline1dinterpolant GetPolyLogm1half(void);
spline1dinterpolant GetPolyLog1half(void);
spline1dinterpolant GetPolyLog3half(void);
spline1dinterpolant GetPolyLog5half(void);
spline1dinterpolant GetCurlyA(void);
spline1dinterpolant GetAuxDawsonErfc(void);
double polylog(string order, double arg, datastruct &data);
inline double CurlyA(double y, datastruct &data){
  if(y<-1.0e+50) return 1.;
  else if(y<-149.99) return 1.-cos(0.25*3.141592653589793+0.6666666666666666666*(-y)*sqrt(-y))/sqrt(3.141592653589793*(-y)*sqrt(-y));
  else if(y<99.99) return spline1dcalc(data.CurlyA,y);
  else if(y>=99.99) return 0.;
  else return 0.;
}
// inline double HG1F2a(double x, datastruct &data){//HypergeometricPFQ[{1/3}, {2/3, 4/3}, x]
//   if(x<-9.9999999e+9) return 0.;
//   //else if(x>1.24999999e+5) return 1.67747618394342e+303;
//   //else if(x>1.24999999e+5) return 1./0.;
//   else if(x>99.999999e+5) return 1./0.;
//   else return spline1dcalc(data.HG1F2_a,x);
// }
// inline double HG1F2b(double x, datastruct &data){//HypergeometricPFQ[{2/3}, {4/3, 5/3}, x]
//   if(x<-9.9999999e+9) return 0.;
//   //else if(x>1.24999999e+5) return 4.4248785716298e+301;
//   //else if(x>1.24999999e+5) return 1./0.;
//   else if(x>99.999999e+5) return 1./0.;
//   else return spline1dcalc(data.HG1F2_b,x);
// }
void InitializePulayMixer(datastruct &data);

// B A S I C    R O U T I N E S
inline double rn(void){ /*random number between 0 and 1*/ return (double)rand()/((double)RAND_MAX); }
inline double ABS(double x){ if(x<0.) return -x; else return x; }
inline double POS(double x){ if(x>0.) return x; else return 0.; }
inline double Heaviside(double x){ if(x>0.) return 1.; else return 0.; }
inline double Norm(vector<double> v){ int dim = v.size(); double vNorm=0.; for(int i=0;i<dim;i++) vNorm+=v[i]*v[i]; return sqrt(vNorm); }
inline double Norm2(vector<double> v){ int dim = v.size(); double vNorm2=0.; for(int i=0;i<dim;i++) vNorm2+=v[i]*v[i]; return vNorm2; }
inline double Sign(double x){ if(x<0.) return -1.; else return 1.; }
inline int IntSign(int x){ if(x<0) return -1; else return 1; }
inline double POW(double x, int p){ double POW = 1.; for(int d=1;d<=p;d++) POW *= x; return POW; }
inline double MOD(double x, double y){ return Sign(x)*(abs(x)-(double)((int)(abs(x)/y))*y); }
inline double LOG(double x){ if(x<9.859676543759771e-305) return -700.; else if(x<1.0142320547350045e+304) return log(x); else return 700.; }
inline double EXP(double x){ if(x<-700.) return 0.; else if(x<700.) return exp(x); else return 1.0142320547350045e+304; }
inline double EXPt(double x, double threshold){ if(ABS(x)<threshold) return exp(x); else if(x>=threshold) return exp(threshold); else return 0.; }
inline double EXPnonzero(double x){ if(x<-700.) return 9.85967654375977e-305; else if(x<700.) return exp(x); else return 1.0142320547350045e+304; }
inline double COS(double x){ /*maximum absolute error = 10^{-4}*/ if(x>3.141592653589793) x = 2.*3.141592653589793-x; double x2 = x*x, x4 = x2*x2; return 0.000000002087675698786810*x4*x4*x4 - 0.0000002755731922398589*x4*x4*x2 + 0.00002480158730158730*x4*x4 - 0.001388888888888889*x4*x2 + 0.04166666666666667*x4 - 0.5000000000000000*x2 + 1.; }
inline double smoothPOS(double x, double T){ double xoverT = max(x/T,-600.); if(xoverT<-40.) return T*exp(xoverT); else if(xoverT>40.) return x; else return T*log(1.+exp(xoverT)); }
inline double ASYM(double x, double y){ if(x==0. && y==0.) return 0.; return (x-y)/(x+y); }
inline double ASYMcritABS(double x, double y, double crit){ if(ABS(x)<crit && ABS(y)<crit) return 0.; return ABS((x-y)/(x+y)); }
inline double FD(double arg, double T){ return 1./(1.+EXP(-arg/T)); }
inline double RelDiff(double a, double b){ double res = 0.; if(a!=0.) res = ABS(b/a-1.); else if(b!=0.) res = ABS(a/b-1.); return res; }
inline vector<double> VecDiff(vector<double> &v,vector<double> &w){ vector<double> res(v.size()); for(int i=0;i<v.size();i++) res[i] = v[i]-w[i]; return res;}
inline vector<double> VecAbsDiff(vector<double> &v,vector<double> &w){ vector<double> res(v.size()); for(int i=0;i<v.size();i++) res[i] = ABS(v[i]-w[i]); return res;}
inline vector<double> VecRelDiff(vector<double> &v,vector<double> &w){ vector<double> res(v.size()); for(int i=0;i<v.size();i++) res[i] = RelDiff(v[i],w[i]); return res;}
inline vector<double> VecSum(vector<double> &v,vector<double> &w){ vector<double> res(v.size()); for(int i=0;i<v.size();i++) res[i] = v[i]+w[i]; return res;}
inline vector<double> VecFact(vector<double> &v,double factor){ vector<double> res(v.size()); for(int i=0;i<v.size();i++) res[i] = v[i]*factor; return res;}
inline void vecFact(vector<double> &v,double factor){ for(int i=0;i<v.size();i++) v[i] *= factor; }
inline double VecMin(vector<double> &v){ double vm = v[0]; for(int i=0;i<v.size();i++){ if(v[i]<vm) vm = v[i]; } return vm; }
inline double VecAv(vector<auto> &v){ double va = 0.; for(int i=0;i<v.size();i++){ va += v[i]; } return va/((double)(v.size())); }
inline double VecTotal(vector<auto> &v){ return accumulate(v.begin(),v.end(),0.); }
inline double ArrayAv(vector<vector<auto>> &a){
	double aa = 0., NumElements = 0.;
	for(int i=0;i<a.size();i++) for(int j=0;j<a[i].size();j++){ aa += a[i][j]; NumElements += 1.; }
	return aa/NumElements;
}
inline double PartialVecAv(vector<auto> &v, int activeElements){ int L = IntSign(activeElements)*min((int)v.size(),abs(activeElements)); double va = 0.; if(L>0) for(int i=0;i<L;i++){ va += (double)v[i]; } else for(int i=0;i<abs(L);i++){ va += (double)v[v.size()-1-i]; } return va/((double)(abs(L))); }
inline double VecVariance(vector<double> &v){ double res = 0., mean = VecAv(v); for(int i=0;i<v.size();i++){ res += (v[i]-mean)*(v[i]-mean); } return res/((double)v.size()); }
inline double AngularFactor(int dim){ switch(dim){ case 1: {return 2.; break;} case 2: {return 2.*3.141592653589793; break;} case 3: {return 4.*3.141592653589793; break;} default: { return 2.; break; } } };
inline bool IntegerElementQ(int test, vector<int> &vec){ for(int i=0;i<vec.size();i++){ if(test==vec[i]) return true; } return false; }
inline int WhichIntegerElementQ(int test, vector<int> &vec){ for(int i=0;i<vec.size();i++){ if(test==vec[i]) return i+1; } return 0; }
inline double BooleCoefficient(int i, int n, double delta){ if(i==0 || i==n) return delta*(2./45.)*7.; else switch(i % 4){ case 1: {return delta*(2./45.)*32.; break;} case 2: {return delta*(2./45.)*12.; break;} case 3: {return delta*(2./45.)*32.; break;} case 4: {return delta*(2./45.)*14.; break;} default: {return 0.; break;} } }
bool NegligibleMagnitudeQ(double bigQ, double smallQ, double MaxExponentDifference);
double VecMult(vector<double> &v,vector<double> &w, datastruct &data);
inline void Normalize(vector<double> &v){ int dim = v.size(); double tmp=0.; for(int i=0;i<dim;i++) tmp += v[i]*v[i]; tmp = 1./sqrt(tmp); vecFact(v,tmp); }
inline vector<double> real_1d_arrayToVec(const real_1d_array x){ vector<double> v(x.length()); for(int i=0;i<v.size();i++) v[i] = x[i]; return v; }
inline real_1d_array VecTOreal_1d_array(vector<double> x){ real_1d_array v; v.setlength(x.size()); for(int i=0;i<x.size();i++) v[i] = x[i]; return v; }
inline int NearestInt(auto x){ int test = (int)x; if(x-(double)test<0.5) return test; else return test+1; }
inline double xiOverlap(vector<double> &p,vector<double> &d){ double scalarproduct = 0.; for(int i=0;i<d.size();i++) scalarproduct += p[i]*d[i]; return 2.*scalarproduct/(Norm2(p)+Norm2(d)); }
double Overlap(vector<double> &p,vector<double> &d);
vector<size_t> sort_indices(vector<double> &v);
vector<size_t> sort_indices_by_copy(vector<double> v);
vector<size_t> sort_indices_reverse(vector<double> &v);
inline double lnfact(int n){ return (double)gsl_sf_lnfact((unsigned int)n); };
inline double Kd(int m, int n){ if(m==n) return 1.; else return 0.; }
inline bool SameParityQ(int i, int l){
  if(i==l) return true;
  else{
    int a,A; if(i>l){ A=i; a=l; } else{ A=l; a=i; }
    if(a==0){ if(A%2==0) return true; else return false; }
    else{ if(A%2==0 && a%2==0) return true; else return false; }
  }
}
inline double ScalarProduct(vector<double> &v,vector<double> &w){ double res = 0.; for(int i=0;i<v.size();i++) res += v[i]*w[i]; return res; }
inline vector<double> SphericalCoor(vector<double> &v){
  double r = Norm(v), x = v[0], y = v[1], z = v[2], theta, phi;
  if(x==0. && y==0.){ phi = 0.; if(z==0.) theta = 0.; }
  else{ theta=acos(z/r); phi = atan2(y,x); }
  return {{r,theta,phi}};
}
inline void SleepForever(void){ cout << "-> going to sleep forever ... press ctrl+c to abort" << endl; while(true){ usleep(10000); for(int i;i<1000;i++){ double j=i*i; }} }
//inline double KeepInRange(auto x, vector<auto> &range){ if(x<range[0]) return range[0]; else if(x>range[1]) return range[1]; else return x; }
inline void KeepInRange(auto &x, vector<auto> &range){ if(x<range[0]) x = range[0]; else if(x>range[1]) x = range[1]; }
void ReadVec(string FileName, vector<double> &vec);
vector<vector<double>> ReadMat(string FileName);
inline double SigmoidX(double y, double a, double b){ return a+(b-a)/(1.+EXP(-100.*y)); }
inline double Sigmoid(double y, double a, double b, double scale){ return a+(b-a)/(1.+EXP(-scale*y)); }
inline double GetConditionNumber(MatrixXd &A){
	JacobiSVD<MatrixXd> svd(A);
	return svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
}
inline VectorXd VecToVectorXd(vector<double> &vec){
	VectorXd Vec(vec.size());
	for(int i=0;i<vec.size();i++) Vec(i) = vec[i];
	return Vec;
}
inline vector<double> VectorXdToVec(VectorXd &vec){
	vector<double> Vec(vec.size());
	for(int i=0;i<vec.size();i++) Vec[i] = vec(i);
	return Vec;
}
inline vector<double> MergeVecs(vector<double> &v1, vector<double> &v2){
    vector<double> v = v1;
    v.insert(v.end(), v2.begin(), v2.end());
    return v;
}
inline vector<vector<double>> MergeMats(vector<vector<double>> &m1, vector<vector<double>> &m2){
    vector<vector<double>> m = m1;
    m.insert(m.end(), m2.begin(), m2.end());
    return m;
}
inline double SortMedian(vector<double> &v) {
  	sort(v.begin(), v.end());
  	int I = v.size()-1;
  	if(I % 2 == 0) return v[I/2];//last index even
  	else return 0.5*(v[v.size()/2] + v[v.size()/2-1]);//last index odd
}
inline int c2i(double y, int m, int n){//snap unbounded continous variable to integer variable between m and n
  int tmp = m;
  if(m>n){ m = n; n = tmp; }
  int res = m+(int)((double)(n-m+1)*0.5*(1.+y/(ABS(y)+1.)));
  if(res>n) return n;
  else if(res<m) return m;
  else return res;
}
inline double c2b(double y, double a, double b){//unbounded continuous variable y to box-constrained variable a<x<b. Ensure that a<b
  return a + 0.5*(b-a)*(1.+cos(y));
}
inline double c2lc(double y, double a, double b){//unbounded continuous variable to linearly constrained variable. Get linearly constraint x from unbounded variable y, where ax+b>=0, i.e. x>=-b/|a|=-b/a for a>0 and x<=b/|a|=-b/a for a<0, i.e., take x=-b/a + sgn(a) y^2, with -inf<y<inf. This includes semi-unbounded domains like x >= 0 with c2lc(y,1.,0.) and x <= 0 with c2lc(y,-1.,0.). Ensure that the constrain has a!=0
  return -b/a + Sign(a)*y*y;
}
vector<int> primeFactors(const int num, int printQ);
vector<int> GetNestedThreadNumbers(const vector<int> primefactors, int printQ);
inline void simulate_work(int start, int end) {// Simulate work using CPU-intensive computation
  volatile double result = 0.0; // Use volatile to prevent optimization
  for (int i = start; i < end; i++) {
    result += i * 0.0001; // Arbitrary calculation
  }
}
inline double GetKDa(double A, double scale){ return 1./(1.+EXP(-A/scale)); }
inline double GetKDb(double B, double scale){ return 1.-EXP(-B/scale); }
inline double GetKDA(double a, double scale){ if(a<=0. || a>=1.) return Sign(a)*1.0e+300; else return -scale*LOG(1./a-1.); }
inline double GetKDB(double b, double scale){ if(b>=1.) return 1.0e+300; else if(b<=0.) return 0.; else return -scale*LOG(1.-b); }
inline double getFreeRAM(datastruct &data){// The freeram value is given in units specified by mem_unit
  	struct sysinfo memInfo;
	unsigned long long freeRAM = 0;
  	if (sysinfo(&memInfo) == 0) freeRAM = memInfo.freeram * memInfo.mem_unit;
  	else cout << "getFreeRAM: Warning !!! Failed to retrieve memory info" << endl;
    return (double)freeRAM;
}
template <typename Container> inline void freeContainerMemory(Container &container){ Container{}.swap(container); }//Create an empty container and swap with the provided container.

// A U X I L L I A R Y    R O U T I N E S
// inline int HalfDiagonalQ(int index, int dim, int steps, datastruct &data){
//   if(index>=data.CentreIndex && index = )
//   for(int i=0;i<steps+1;i++)
//   int factor=0; for(int i=0;i<dim;i++) factor+=(int)(0.5+pow(double(steps+1),(double)i)); for(int i=steps/2;i<steps+1;i++) if(index==i*factor) return 1; return 0;
// }
bool BoxBoundaryQ(int index, datastruct &data);
vector<double> rGrid(int dim, int steps, vector<double> &frame);
void VecAtIndex(int Index, datastruct &data, vector<double> &out);
void kVecAtIndex(int Index, datastruct &data, vector<double> &out);
vector<vector<double>> GetVecsAt(datastruct &data);
vector<vector<double>> GetkVecsAt(datastruct &data);
void GetNorm2kVecsAt(datastruct &data);
void DelField(vector<double> &InputField, int s, int method, datastruct &data);
void FieldProductToTmpField1(vector<double> &Field1, vector<double> &Field2, datastruct &data);
void MultiplyField(datastruct &data, vector<double> &Field, int dim, int EdgeLength, double factor);
void MultiplyComplexField(datastruct &data, vector<vector<double>> &Field, int dim, int EdgeLength, double factor);
void SetField(datastruct &data, vector<double> &Field, int dim, int EdgeLength, double value);
void AddField(datastruct &data, vector<double> &Field, vector<double> &FieldToAdd, int dim, int EdgeLength);
void SetComplexField(datastruct &data, vector<vector<double>> &Field, int dim, int EdgeLength, double Re, double Im);
void AddFactorComplexField(datastruct &data, vector<vector<double>> &Field, vector<vector<double>> &FieldToAdd, int dim, int EdgeLength, double factorRe, double factorIm);
void CopyComplexField(datastruct &data, vector<vector<double>> &Field, vector<vector<double>> &FieldToCopy, int dim, int EdgeLength);
void CopyRealFieldToComplex(datastruct &data, vector<vector<double>> &Field, vector<double> &FieldToCopy, int Targetcomponent, int dim, int EdgeLength);
void CopyComplexFieldToReal(datastruct &data, vector<double> &Field, vector<vector<double>> &FieldToCopy, int component, int dim, int EdgeLength);
void CheckNAN(string routine, vector<vector<double>> &FieldToCheck, int dim, datastruct &data);
double MaxAbsOfComplexField(datastruct &data, vector<vector<double>> &Field, int dim, int EdgeLength);
double HG1F2(const double a1,const double b1,const double b2,const double x);

// I N T E G R A T I O N ,    I N T E R P O L A T I O N,    F F T ,    A N D    O P T I M I Z A T I O N    R O U T I N E S
double Integrate(int ompThreads, int method, int dim, vector<double> &f, vector<double> &frame);
double Riemann(int ompThreads, int dim, vector<double> &f, vector<double> &frame);
double RiemannBareSum(int ompThreads, int dim, vector<double> &f, vector<double> &frame);
double BooleRule(int ompThreads, int dim, vector<double> &f, vector<double> &frame);
double BooleRule1D(int ompThreads,  vector<double> &f, int Lf, double a1, double b1, int bp);
double BooleRule2D(int ompThreads,  vector<double> &entiref, int Lf, int Lf2, double a1, double b1, double a2, double b2, int bp);
double BooleRule3D(int ompThreads,  vector<double> &entiref, int Lf, int Lf2, double a1, double b1, double a2, double b2, double a3, double b3, int bp);
void SplineOnMainGrid(vector<double> &InputField, datastruct &data);
void SplineOnStrideGrid(vector<double> &InputField, datastruct &data);
void GetStridelattice(datastruct &data);
void fftParallel(int sign, datastruct &data);
int monitorBox(int dim, datastruct &data);
void ConjugateGradientDescent(int tf, int GlobalMinCandidateCounterMax, int countMax, vector<double> &Start, taskstruct &task, datastruct &data);
void FunctionToBeMinimized(const real_1d_array &x, double &func, void *ptr);
double functionToBeMinimized(real_1d_array &x, int globalinteger);
void GradientDescent(taskstruct &task);
void gradient_descent(vector<double> &report, vector<double> &x, double AbsoluteTargetAcc, double gammaStart, int maxIter, double AnnealThreshold, double AnnealMagnitude, taskstruct &task);
// void FitFunction_ForestGeo_ALGLIB(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr);
double FitFunction_ForestGeo_PSO(vector<double> var, int s);
double EquilibriumAbundances_ForestGeo_PSO(vector<double> var, int s);

// T E S T G R O U N D    R O U T I N E S
void RunTests(taskstruct &task, datastruct &data);
void RunAuxTasks(taskstruct &task, datastruct &data);
void testUnitaries(taskstruct &task, datastruct &data);
void FillComplexField(int option, datastruct &data);
void passDoubleOMPloop(int n, vector<double> &AV, vector<double> &AV2);
void testSWAP(int n, datastruct &data);
void testFFT(datastruct &data);
void report_num_threads(int level);
void process_idle_threads(vector<int> &thread_active);
void testOMP(int n, datastruct &data);
void testCPU(int n, datastruct &data);
void testBoxBoundaryQ(datastruct &data);
void testKD(datastruct &data);
void testFitFunction(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr);
//void testALGfit(datastruct &data);
void testisfinite(void);
void testPCA(datastruct &data);
void testRBF(datastruct &data);
void testcec14(datastruct &data, taskstruct &task);
void testOptimizers(datastruct &data, taskstruct &task);
void test1pExDFTN2(datastruct &data, taskstruct &task);
void testALGintegrator(datastruct &data, taskstruct &task);
void testHydrogenicHint(datastruct &data, taskstruct &task);
void testK1(datastruct &data, taskstruct &task);
string exec(const char* cmd);
int testIAMIT(vector<double> &x, datastruct &data, taskstruct &task);
void testScript(void);
void fab(vector<vector<double>> &TargetField, int a, int b, datastruct &data, taskstruct &task);
inline double KDphi(double alpha, double beta, double s){ return alpha*s+POW(beta*s,3)/3.+1./(4.*s); }
double tweeze_phi(double x, double a, double b, bool left, KDparams &p);
double K1integrand(double x, KDparams &p, taskstruct &task);
double GetPartialK1(double a, double b, KDparams &p, taskstruct &task);
double GetBisectionK1(double a, double b, int iter, double prevInt, KDparams &p, taskstruct &task);
double getK1(KDparams &p, datastruct &data, taskstruct &task);
void GetKD(int D, double alpha, double beta, KDparams &p, datastruct &data, taskstruct &task);


// R O U T I N E S    F O R  ---  OPTIMIZATION  ---  MACHINE LEARNING  --- NEURAL NETWORKS
void ParetoFront(datastruct &data, taskstruct &task);
vector<double> Optimize(int func_ID, int opt_ID, int aux, datastruct &data, taskstruct &task);
void testNN(datastruct &data);
void OptimizeNN(datastruct &data);
void SetDefaultNNparams(int dim, MLPerceptron &NN);
double NNtestFunction(vector<double> &x, MLPerceptron &NN);
double NNprocess(vector<double> x, MLPerceptron &NN);
double ProduceNNinstance(vector<double> par);
void InitRandomNumGeneratorNN(MLPerceptron &NN);

// H A N D L I N G    R O U T I N E S
int ManualSCloopBreakQ(datastruct &data);
void abortQ(datastruct &data);

// O U T P U T    R O U T I N E S
void WriteControlFile(datastruct &data);
void StartTimer(string descriptor, datastruct &data);
void EndTimer(string descriptor, datastruct &data);
void StartTiming(datastruct &data);
void EndTiming(datastruct &data);
void PRINT(string str,datastruct &data);
void TASKPRINT(string str,taskstruct &task, int coutQ);
inline bool printQ(int count){ for(int i=1;i<10;i++){ if(i*(int)pow(10.,floor(log10((double)count)))==count || count%1000==0) return true; } return false; }
template <typename T> inline std::string to_string_with_precision(T a_value, int prec){ std::ostringstream out; out << std::setprecision(prec) << a_value; return out.str(); }
template <typename T> inline std::string vec_to_str(vector<T> vec){ std::ostringstream oss; for(int i=0;i<vec.size();i++) oss << to_string_with_precision(vec[i],16) << " "; return oss.str(); }
template <typename T> inline std::string vec_to_str_with_precision(vector<T> vec, int prec){ std::ostringstream oss; for(int i=0;i<vec.size();i++) oss << to_string_with_precision(vec[i],prec) << " "; return oss.str(); }
template <typename T> inline std::string mat_to_str(vector<vector<T>> mat){ std::ostringstream oss; for(int i=0;i<mat.size();i++) oss << vec_to_str_with_precision(mat[i],16) << "\n"; return oss.str(); }
template <typename T> inline std::string partial_vec_to_str_with_precision(vector<T> vec, int start, int end, int prec){ std::ostringstream oss; for(int i=start;i<min(end,(int)vec.size());i++) oss << to_string_with_precision(vec[i],prec) << " "; return oss.str(); }
template <typename T> inline std::string partial_vec_to_str(vector<T> vec, int start, int end){ std::ostringstream oss; for(int i=start;i<min(end,(int)vec.size());i++) oss << to_string(vec[i]) << " "; return oss.str(); }
template <typename T> inline std::string vec_to_CommaSeparatedString_with_precision(vector<T> vec, int prec){ std::ostringstream oss; oss << "{{"; for(int i=0;i<vec.size()-1;i++) oss << to_string_with_precision(vec[i],prec) << ","; oss << to_string_with_precision(vec[vec.size()-1],prec) << "}}"; return oss.str(); }
template <typename T> inline std::string IntVec_to_CommaSeparatedString(vector<T> vec){ std::ostringstream oss; oss << "{{"; for(int i=0;i<vec.size()-1;i++) oss << to_string(vec[i]) << ","; oss << to_string(vec[vec.size()-1]) << "}}"; return oss.str(); }
//string vec_to_str(vector<double> vec);
void COUTalongX(string str,vector<double> &Field,datastruct &data);
string YYYYMMDD(void);
string hhmmss(void);
void VecToFile(vector<double> &vec, string FileName);
void MatrixToFile(vector<vector<double>> &matrix, string FileName, int prec);
void GetFieldStats(int PrintFieldStatsQ, int OmitWallQ, datastruct &data);
bool WithinWallsQ(vector<double> vec, datastruct &data);
void GetCutPositions(datastruct &data);
void GetCutDataStream(vector<vector<vector<double>>> &FI, datastruct &data);
vector<vector<vector<double>>> GetFieldInterpolationsAlongCut(datastruct &data);
void Regularize(vector<int> procedures, vector<double> &Field, double auxVal, datastruct &data);
void GetConsumedResources(datastruct &data);
void StoreMovieData(datastruct &data);
void StoreSnapshotData(datastruct &data);
void SetStreams(datastruct &data);
void ExportData(datastruct &data);
void CleanUp(datastruct &data);
void StoreTASKdata(taskstruct &task, datastruct &data);
void ProcessTASK(taskstruct &task);
void LoadReferenceDensity(datastruct &data);
void LoadFOMrefData(int averageQ, datastruct &data, taskstruct &task);
void LoadrefData(int averageQ, datastruct &data, taskstruct &task);
void LoadFitParameters(vector<double> &LoadedParams);
void LoadFitResults(vector<double> &LoadedParams);
vector<double> LoadIntermediateOPTresults(void);
double AverageOverNeigbours(int row, int col, int nx, int ny, vector<vector<double>> &tmprefData);
void LoadAbundances(datastruct &data);
double fomOverlap(int s, datastruct &data);
double GetFigureOfMerit(datastruct &data);
vector<double> GetNetMagnetization(datastruct &data);
double GetColumnDensityMagnetization(datastruct &data);
std::string getExecutableDirectory(void);
std::string findMatchingFile(const std::string& baseName, const std::string& extension);
vector<vector<double>> readFile(const string &filename);
void ExportTriangulation(datastruct &data);
void CleanUpTriangulation(datastruct &data);
void RemoveSmallTriangles(const double percentage, const vector<vector<vector<double>>> &InTriangles, vector<vector<vector<double>>> &OutTriangles, datastruct &data);
bool ExtractIntegerFromPattern(string filename, int &out);
