#include "stdio.h"
#include <cmath>
//#include <math.h>//possible conflict with cmath
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <limits>
#include <vector>
#include <random>
#include <sstream>
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
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <Eigen/Dense>
#include <Eigen/QR>

using namespace std;
using namespace std::chrono;
using namespace alglib;
using namespace Eigen;

struct MemberShipStruct
{
  double Same;
  double Near;
  double Far;
  double Better;
  double Unvaried;
  double Worse;  
};

struct velocity
{
  int size;
  vector<double> v;
};

struct position
{
  int size;
  double f;
  vector<double> x;
};

struct LCOreport
{
    int FuncEvals;
    double bestf;
    vector<double> OccNum;
    vector<double> Phases;
};

struct PCARBFreport
{
  //move-trajectories
  vector<double> F;//objective-function estimates along move-trajectories
  vector<vector<double>> xVec;
  vector<vector<double>> vVec;
};

struct PCARBFstruct
{
  bool enabled = false;
  bool active = false;
  int dim;//number of dimensions for interpolant == number of principal components used
  int funcDIM = 1;//scalar function; implementation for scalar functions only (for now)
  vector<vector<rbfmodel>> RBFinterpolant;//storage for RBF interpolants
  vector<vector<vector<double>>> W;// PCA transformation matrix [RBFlayer][EigVec(x),EigVec(y),...][IndividualCoordinate]
  int RBFasymptote = 0;//which polynomial the interpolant tends to asymptotically --- 0: zero --- 1: average of the dataset --- 2: linear
  double fshift = 0.;
  vector<vector<vector<double>>> hyperbox;
  double RBFbaseRadiusPrefactor = 4.;//2.;
  int RBFnumLayersPrefactor = 1;//3;
  double smoothing = 0.;//default smoothing choice for RBF: 1.0e-4...1.0e-3 --- put to zero to omit smoothing
  bool exportQ = false;//whether or not to export "RBF_dataset.dat", "RBF_interpol.dat"
  vector<vector<double>> xList;
  vector<vector<double>> fList;
  bool UseSuccessfulPredictions = false;
  vector<position> SuccessfulPredictions;
  double RelDiffCriterion = 0.001;
  vector<vector<vector<double>>> yList;
  int NumPoints = 10000;//(approximate) number of scattered data vectors to be collected before RBF interpolation is executed
  bool NumPointsUpdatedQ = false;
  int pcaType = 1;//0: full PCA --- 1: truncated PCA
  vector<double> PCAeigValues;
  vector<vector<double>> ExtentVec;
  double eps = 1.0e-3;//internal accuracy criterion for PCA
  ae_int_t maxits = 0;//max number of internal iterations; if(maxits==0) then termination according to eps
  int MaxMoveCount = 1000;//max number of allowed (optimizer-dependent) stochastic moves for finding a point in search space worthy of evaluation
  int ActiveLayers = 1;//number of last RBF-interpolants to use for estimating objective function
  double MovePrefactor = 1.0e-3;//for increasing move-range after each unsuccessful move
  double MovePrefactorMin;
  double MovePrefactorMax;
  double fRange;//range of objective function values compared againt test value when accepting (or rejecting) move
  vector<double> fShift;
  vector<double> fSpread;
  vector<double> MoveAcceptanceRate;
  vector<double> moveCount;
  vector<PCARBFreport> report;
  double SuccessfulMoveCriterion = 0.;
  vector<double> successRate;
  vector<double> successfulMoveRate;
  vector<double> successfulJumpRate; 
  vector<int> JumpAt;
  double SuccessRate;
  double SuccessfulMoveRate;
  double SuccessfulJumpRate;
  double AcceptanceRate;
  int AverageMoveCount;  
};

struct CMAstruct
{
	int runs; // Numbers of runs
	vector<int> AbortQ;
	int generation;
	int generationMax;
	int InitialPopulationSize;
	int NewPopulationSize;
	int popExponent;
	double PopulationDecayRate;
	bool PickRandomParamsQ = false;
	bool DelayEigenDecomposition = false;
	bool elitism = false;
	double CheckPopVariance;
	int Constraints = 0;
	bool exit = false;
	int stall;
	vector<string> report;
	vector<vector<vector<double>>> pop;//holds all populations, chromosomes, genes
	vector<vector<vector<double>>> pop2;
	vector<vector<double>> f;
	double bestf = 1.0e+300; // lowest minimum encountered
	vector<double> bestx; // storage for best minimizer  
	int bestp = 0;
	vector<double> bestfVec;
	vector<double> worstfVec;
	vector<vector<double>> bestxVec;
	int populationSize;
	vector<vector<double>> mean;
	vector<vector<double>> meanOld;
	vector<bool> MeanFromAllWeights;
	int mu;
	double muRatio = 0.5;
	int WeightScenario;
	vector<double> betac;
	vector<double> spread;
	vector<double> penaltyFactor;
	vector<double> InitPenaltyFactor;
	double TotalPenalty = 0.;
	vector<double> weights;
	double WeightSum;
	double hsigmafactor;
	double mueff;
	double csigma;
	double dsigma;
	vector<double> cc;
	double cmu;
	double c1;
	vector<double> cm;
	vector<double> stepSize;
	double InitStepSizeFactor;
	vector<int> EVDcount;//counter of eigenvalue decompositions
	vector<int> MaxDelay;
	vector<MatrixXd> Covariance;
	vector<MatrixXd> D;//sqrt of EigenValues
	vector<MatrixXd> B;//EigenVectors
	vector<MatrixXd> CovInvSqrt;
	vector<vector<double>> psigma;//evolution path sigma
	vector<vector<double>> pc;//evolution path c
	vector<double> psigmaNorm;
	int VarianceCheck;
	vector<vector<double>> history;
	double ExpectedValue;
	double alphacov;
	vector<vector<double>> InitBias;
};

struct GAOstruct//MIT
{
  int runs = 1; // Numbers of runs 	
  bool activeQ = false;
  int VarianceCheck;
  bool exit = false;
  vector<string> report;
  bool RandomSearch = false;
  
  double popExponent = 0.25;//between 0. and 1.
  int populationSize = 30;
  int InitialPopulationSize;
  double PopulationDecayRate;
  bool ShrinkPopulationsQ = false;
  int NumParents;
  int generation;
  int generationMax;
  vector<vector<vector<double>>> pop;//holds all populations, chromosomes, genes
  vector<vector<vector<double>>> pop2;//for swap with pop
  vector<vector<double>> f;
  double bestf = 1.0e+300; // lowest minimum encountered
  vector<double> bestx; // storage for best minimizer    
  vector<double> bestfVec;
  vector<double> bestfVecOld;  
  int bestp;
  double Spread = 0.;
  double TargetSpread;
  double InitSpread;
  
  int HyperParamsSchedule = 0;  
  bool HyperParamsDecayQ = true;   
  vector<vector<double>> hp;
  vector<vector<double>> hpInit;
  int numhp;  
  vector<string> hpNames;
  double mutationRate; 
  double mutationRateDecay;
  double mutationStrength;   
  double mutationStrengthDecay;
  double invasionRate;
  double invasionRateDecay;
  double crossoverRate;  
  double crossoverRateDecay;
  double dispersalStrength;
  double dispersalStrengthDecay;
  vector<vector<double>> hpRange;  
  vector<double> MutationRateRange = vector<double>(2);
  vector<double> MutationRateDecayRange = vector<double>(2);  
  vector<double> MutationStrengthRange = vector<double>(2);
  vector<double> MutationStrengthDecayRange = vector<double>(2); 
  vector<double> InvasionRateRange = vector<double>(2);
  vector<double> InvasionRateDecayRange = vector<double>(2);  
  vector<double> CrossoverRateRange = vector<double>(2);
  vector<double> CrossoverRateDecayRange = vector<double>(2);
  vector<double> DispersalStrengthRange = vector<double>(2);
  vector<double> DispersalStrengthDecayRange = vector<double>(2);  
  vector<double> centralhp;
  vector<vector<double>> Successfulhp;  
  double AvBinHeight;
  vector<int> nb_eval;  
  vector<vector<vector<int>>> histogram;
  vector<int> BinHeights;
  vector<vector<double>> bp;
  

//   int NumberOfParents = 2;//ToDo: crossover of multiple parents depending on fitness, and adaptable during evolution to prevent (detrimental) incest
//   int AllowImmigrationAt;//immigration starts after AllowImmigrationAt generations
//   bool FetchImmigrantNow = false;
//   int best_stall_max=20;
//   int average_stall_max=20;
//   double tol_stall_best=1e-9; //MIT: default:1e-6;
//   double tol_stall_average=1e-9; //MIT: default:1e-6;
//   double transferRatio = 0.1;//MIT option1: ratio of fittest chromosomes within current generation to be transferred to next generation
//   double crossover_fraction=0.7;//MIT: default: 0.7
//   double crossover_rate;
//   double VariableMutationRate = -1.;
//   vector<position> X; // Positions
//   int N_threads = 1;
};

struct cSAstruct//MIT
{
  vector<position> X; // Positions
  double bestf = 1.0e+300; // lowest minimum encountered
  vector<double> bestx; // storage for best minimizer  
  int max_iterations = 1000000;//The maximum number of iterations/steps. 
  double max_funcEvals = 1.0e+6;
  float tgen_initial = 0.01;// The initial value of the generation temperature.
  float tgen_schedule = 0.99999;// Determines the factor that `tgen` is multiplied by during each update.
  double LargeStepProb = 0.1;//probability of making a large step (up to tgen_initial)
  double minimal_step = 1.0e-12;
  double AcceptProb = -1.;//constant acceptance probability (disabled if negative)
  float minimal_AcceptProb = 0.;
  float tacc_initial = 0.9;// The initial value of the acceptance temperature.
  float tacc_schedule = 0.01;// Determines the factor by which `tacc` is increased or decreased during each update.
  float desired_variance = 0.99;// The desired variance of the acceptance probabilities.
  vector<int> nb_eval;
  int VarianceCheck = 10;
  double epsf;
  vector<double> averageStepSize;
  int maxStallCount;
};

struct PSOstruct//MIT
{
  int Smax; // maximum swarm size
  int InitialSwarmSize; // Initial number of swarm particles (capped at Smax)  
  int InitialSwarmSizeCurrentRun;
  int CurrentSwarmSize;
  bool NewSwarmSizeQ = false;
  vector<vector<int>> LINKS;
  int MaxLinks; // Max number of particles informed by a given one
  int init_links; // Flag to (re)init or not the information links
  double w = 0.42;//0.6;//1./(2.*log(2.));// First confidence coefficient
  vector<vector<double>> W;
  double c = 1.55;//1.7;//0.5+log(2.);// Second confidence coefficient
  int CoefficientDistribution = 0;
  vector<vector<double>> C1;
  vector<vector<double>> C2;
  double centralW;
  double centralC1;
  double centralC2;
  vector<double> SuccessfulW;
  vector<double> SuccessfulC1;
  vector<double> SuccessfulC2;
  double spread;
  int UnsuccessfulAttempts = 0;
  double AbsAcc = 0.;
  bool UpdateParamsQ = false;
  double InformerProbability = 1.0;
  vector<double> I;
  vector<double> R;
  double epsf = 1.0e-6; // //absolute tolerance for convergence of objective function
  int VarianceCheck; //number of PSO loops to take into account for variance check  
  double elitism; // elitism from none (0.) to full (1.)
  int avPartQ; // 1: replace worst by average particle --- 0: don't
  int PostProcessQ = 0; // post-processing --- 0: none --- 1: only newly found bestx with LCO --- 2:  CurrentBestx with LCO
  bool AlwaysUpdateSearchSpace = false;
  double SwarmDecayRate = 0.;//if >0., decrease swarm size from initial swarm size, reaching S=opt.threads just before opt.VarianceThreshold is reached
  int eval_max_init; // maximum number of evaluations for first run
  double increase; // increase swarm size and eval_max in consecutive runs  
  int TargetMinEncounters; // minimum number of runs that encounter the current optimum within epsf
  int runs; // Numbers of runs  
  int nb_eval;
  double reseed;
  int reseedEvery;
  double reseedFactor = 0.1;
  int loopMax;
  int bestToKeep;
  double alpha = 1.565;  
  int MinimumRandomInitializations;
  vector<vector<double>> bestparams;
  vector<double> ab;
  double bestThreshold;
  vector<int> loopCountHistory;
  vector<double> History;
  vector<double>SortedHistory;
  vector<vector<vector<double>>> PSOparamsHistory;
  vector<vector<vector<double>>> SortedPSOparamsHistory;
  vector<vector<double>> HistoryX;
  int best; // best swarm index
  int worst; // worst swarm index
  double error;// Error for a given position
  double error_prev; // Error after previous iteration  
  double OldBestf;
  vector<int> bestHistory; // collection of best swarm indices
  double bestf; // lowest minimum encountered
  vector<double> bestx; // storage for best minimizer
  vector<position> X; // Positions
  vector<position> Xold; // Positions
  vector<position> P; // Best positions
  vector<velocity> V; // velocities
  double vmax = 1.0;
  double vmin = 1.0e-12;
  vector<double> Vmin;
  vector<double> Vmax;
  vector<double> delta;
  vector<double> phi;
  vector<int> besthistory;
  vector<double> loopcounthistory;
  vector<double> besthistoryf;
};

struct CGDstruct//MIT
{
  ae_int_t maxits;//0 for unlimited number of iterations
  double scale;
  double epsf;
  double epsg;
  double epsx;
  double diffstep;
  int CGDgr = 0; // Total number of gradient evaluations
  double bestf; // lowest minimum encountered
  vector<double> bestx;
};

struct AULstruct//MIT
{
  ae_int_t maxits;//0 for unlimited number of iterations
  double scale;
  double epsx;
  double epsf;
  double diffstep;
  int CGDgr = 0; // Total number of gradient evaluations
  double bestf; // lowest minimum encountered
  vector<double> bestx;
};

struct LCOstruct//MIT
{
  int runs;
  vector<double> evals;
  vector<double> History;
  vector<vector<double>> history;
  vector<double> LocalRuns;
  vector<double> InitParam; //starting point for x
  real_1d_array x;// = "[0,0]";//variables
  real_1d_array s;// = "[1,1]";//scales
  real_1d_array bndl;// = "[-1,-1]";//-INF for unbounded
  real_1d_array bndu;// = "[1,1]";//+INF for unbounded
  real_2d_array c;// = "[[1,-1,1],[1,1,0.],[1,2,0.1]]";
  integer_1d_array ct;// = "[1,0,-1]";//constraint type -> array c means: 1*x-1*y>=1 && 1*x+1*y==0 && 1*x+2*y<=0.1  
  ae_int_t maxits;//0 for unlimited number of iterations
  double scale;
  double epsx;
  double epsf;
  double epsg;
  double diffstep;
  int CGDgr = 0; // Total number of gradient evaluations
  double bestf; // lowest minimum encountered
  vector<double> bestx;
  int VarianceCheck; // how many runs to use for checking variance convergence
  double VarianceThreshold;
  LCOreport rep;
};

struct OPTstruct//MIT
{
  CGDstruct cgd; // Conjugate Gradient Descent
  PSOstruct pso; // Particle Swarm Optimization
  LCOstruct lco; // Linearly-Constrained Optimization
  AULstruct aul; // Augmented Lagrangian Method
  GAOstruct gao; // Genetic Algorithm Optimization  
  cSAstruct csa; // coupled Simulated Annealing
  CMAstruct cma; // Covariance Matrix Adaptation Evolutionary Strategy
  PCARBFstruct pci; // for usage of PCA-based RBF-interpolated objective function
  exDFTstruct ex; // 1p-exact DFT data  
  mt19937_64 MTGEN;
  uniform_real_distribution<double> RNpos;
  normal_distribution<> RNnormal;
  int ActiveOptimizer;
  int TestMode = 0;
  int function; // Code of the objective function
  int D; // Search space dimension  
  double epsf = 1.;// absolute accuracy criterion for function
  double nb_eval = 0.; // Total number evaluations
  double evalMax = 1.0e+8;// Maximum number of function evaluations
  int reportQ = 0;//do report --- 0: nothing --- 1: +exponential loops --- 2: +SearchSpaceUpdates & re-seeding @ exponential loops
  int reportQOriginal = 0;
  bool ExportIntermediateResults = true;
  bool PCARBF = false;
  bool PostProcessQ = false;
  bool PickRandomParamsQ = false;
  vector<vector<double>> SearchSpace; // Search space
  vector<vector<double>> InitSearchSpace;
  double SearchSpaceMin = 0.;
  double SearchSpaceMax = 0.;
  double SearchSpaceExtent;
  vector<double> SearchSpaceCentre;
  vector<double> searchSpaceExtent;
  vector<double> SearchSpaceLowerVec;
  vector<double> SearchSpaceUpperVec;
  vector<bool> VariableLowerBoundaryQ;
  vector<bool> VariableUpperBoundaryQ;
  int UpdateSearchSpaceQ = 0; //0: don't --- 1: update if variable boundary --- 2: in any case, but only within InitSearchSpace
  bool ResetSearchSpaceQ = true;
  int threads = 1;
  int localrun = -1;
  int loopCount;
  vector<double> LoopCounts;  
  int printQ = 1; // whether to print meta information during optimization loop
  ostringstream control;  
  vector<double> Historyf;
  int BreakBadRuns = 0;
  int checkAtLoop;
  vector<double> BestFuncAtLoop;
  bool varianceUpdatedQ = false;
  double CurrentVariance = 1.0e+300;
  double OldVariance = 1.0e+300;
  double VarianceThreshold = 0.;
  double currentBestf = 1.0e+300;
  vector<double> currentBestx;
  int currentBestp = 0;
  vector<double> Start;
  int TerminateQ = 0;
  chrono::high_resolution_clock::time_point Timer;
  vector<double> Timings;
  string timeStamp;
  vector<vector<double>> AuxMat;
  vector<double> AuxParams;
  vector<double> AuxVec;
  double AuxVal;
  int AuxIndex;
  int SubFunc = 0;
  bool NewbestFoundQ = false;
  int FailCount = 0;
  int FailCountThreshold = 10;
  int stallCheck = 100;
  bool ReportX = false;
  string report = "";
  bool finalcalc = false;
  bool postprocess = false;
  int NumEC = 0;//number of equality constraints (==0)
  int NumIC = 0;//number of inequality constraints (>=0)
  vector<vector<double>> maxEC;//maximum violations of equality constraints
  vector<vector<double>> maxEC2;
  vector<vector<double>> maxIC;//maximum violations of inequality constraints
  vector<vector<double>> maxIC2;
  vector<vector<double>> ALmu;//prefactor of quadratic terms in Augmented Lagrangian method
  vector<vector<double>> ALmu2;
  vector<vector<vector<double>>> ALlambda;//prefactor of linear terms in Augmented Lagrangian method
  vector<vector<vector<double>>> ALlambda2;
  vector<double> PenaltyMethod = {0.5,-1.,0.,0.};//(int)PenaltyMethod[0] --- 0: fixed quadratic penalty method --- 1: Augmented Lagrangian Method (Vanilla) --- 2: Augmented Lagrangian Method (MIT) --- PenaltyMethod[1] --- PenaltyScale, initial value for ALmu --- PenaltyMethod[2] --- initial value for all ALlambda --- PenaltyMethod[3] --- percentage of increase of ALmu after each iteration
  double anneal = -1.;//add decaying noise to the objective function, in lieu of standard annealing --- <=0.: don't --- >0.: percentage of spread for the initial amplitude
  int AnnealType;
  bool Rand = false;
  vector<double> RandShift;
  MatrixXd RandMat;
  MatrixXd RandMatInv;
  int inflate = 0;//inflate the dimensionality of the search space in order to lift the algorithm out of local minima - handle within objective function
  double lowerBound;
  double upperBound;
  int homotopy = 0;
};

void CGD(OPTstruct &opt);
void PSO(OPTstruct &opt);
void MakeMove(int s, int best, int g, OPTstruct &opt);
void PSOKeepInBox(int s, OPTstruct &opt);
void ConstrainPSO(int s, OPTstruct &opt);
void InitializeSwarmParticle(int s, OPTstruct &opt);
void FuzzyPSOparams(OPTstruct &opt);
bool breakLoopQ(double test, OPTstruct &opt);
void LCO(OPTstruct &opt);
void setupLCO(int runs, int dim, double acc, OPTstruct &opt);
void reportLCO(OPTstruct &opt);
void AUL(OPTstruct &opt);
void GAO(OPTstruct &opt);
void SetDefaultGAOparams(OPTstruct &opt);
void InitializeGAO(OPTstruct &opt);
void InitializeChromosome(int p, int c, OPTstruct &opt);
void PickHyperParametersGAO(int p, OPTstruct &opt);
void EvaluateGAO(OPTstruct &opt);
void ReportGAO(OPTstruct &opt);
void CrossoverGAO(OPTstruct &opt);
void MutateGAO(OPTstruct &opt);
void ConstrainGAO(OPTstruct &opt);
void constrainGAO(int p, int c, OPTstruct &opt);
void SelectGAO(OPTstruct &opt);
void UpdateGAO(OPTstruct &opt);
void ShrinkPopulationsGAO(OPTstruct &opt);
void cSA(OPTstruct &opt);
double fcSA(void* instance, double* x);
void step(void* instance, double* y, const double* x, float tgen);
void progress(void* instance, double cost, float tgen, float tacc, int opt_id, int iter);

void SetDefaultCMAparams(OPTstruct &opt);
void InitializePopulationSizeCMA(OPTstruct &opt);
void CMA(OPTstruct &opt);
void InitializeCMA(OPTstruct &opt);
void PickParamsCMA(OPTstruct &opt);
void SampleCMA(OPTstruct &opt);
void SampleMultivariateNormalCMA(int p, OPTstruct &opt);
inline bool CheckBoxConstraintViolationCMA(double test, int d, OPTstruct &opt){ if(test < opt.SearchSpace[d][0] || test > opt.SearchSpace[d][1]) return true; else return false; }
void ConstrainCMA(int p, OPTstruct &opt);
void CMAKeepInBox(int p, int c, OPTstruct &opt);
void RepairCMA(int p, OPTstruct &opt);
void ReportCMA(OPTstruct &opt);
void UpdateMeanCMA(OPTstruct &opt);
void UpdateEvolutionPathsCMA(OPTstruct &opt);
void UpdateCovarianceCMA(OPTstruct &opt);
MatrixXd GetInvSqrt(MatrixXd &A, int p, bool validate, bool &terminateQ, OPTstruct &opt);
void UpdateStepSizeCMA(OPTstruct &opt);
void UpdateCMA(OPTstruct &opt);

void GetECIC(vector<double> &EC, vector<double> &IC, vector<double> &x, OPTstruct &opt);
double GetPenalty(int p, vector<double> &x, OPTstruct &opt);
double GetFuncVal( int s, vector<double> &X, int function, OPTstruct &opt); // Fitness evaluation
double ShiftedSphere(vector<double> &x, OPTstruct &opt);
double ConstrainedRastrigin(vector<double> &x, OPTstruct &opt, int s);
double ChemicalProcess(vector<double> &x, OPTstruct &opt, int s);//mixed integer problem (Himmelblau...pdf)
vector<double> x2X_ChemicalProcess(vector<double> &x);//box constraints for ChemicalProcess
double SineEnvelope(vector<double> &x, OPTstruct &opt);
double Rana(vector<double> &x, OPTstruct &opt);
double UnconstrainedRana(vector<double> &x, OPTstruct &opt);
double UnconstrainedEggholder(vector<double> &x, OPTstruct &opt);
double MutuallyUnbiasedBases(vector<double> &x, OPTstruct &opt);
double NYFunction(vector<double> &x, OPTstruct &opt);
double QuantumCircuitIA(vector<double> &x, OPTstruct &opt);
vector<double> SampleFromSphere(OPTstruct &opt);
vector<double> PostProcessNYFunction(double f, OPTstruct &opt);
double DFTe_QPot(vector<double> &x, OPTstruct &opt);
double DynDFTe_TimeSeries(vector<double> &x, OPTstruct &opt);
double ObjFunc(vector<double> &x, OPTstruct &opt);
void Objective_Function(const real_1d_array &x, double &func, void *ptr);

void InitializePSOrun(OPTstruct &opt);
bool ConvergedQ(int n_exec, OPTstruct &opt);
bool PostProcess(OPTstruct &opt);
bool VarianceConvergedQ(OPTstruct &opt);
bool UpdateSearchSpace(OPTstruct &opt);
void InitRandomNumGenerator(OPTstruct &opt);
void SetDefaultOPTparams(OPTstruct &opt);
void RandomizeOptLocation(OPTstruct &opt);
void GetFreeIndices(OPTstruct &opt);
void SetDefaultSearchSpace(OPTstruct &opt);
vector<vector<double>> PCA(OPTstruct &opt);
void UpdatePCARBF(OPTstruct &opt);
bool AcceptMoveQ(int thread, int s, vector<double> &xp, vector<double> &vp, double fbenchmark, OPTstruct &opt);
void BuildRBFinterpolant(vector<vector<double>> &ypList, vector<double> &fxList, OPTstruct &opt);
double EvalRBFonLayer(int layer, int thread, vector<double> &xp, OPTstruct &opt);
void PreparePCARBF(OPTstruct &opt);
void AdaptPCARBF(OPTstruct &opt);

void SetDefaultPCIparams(OPTstruct &opt);
void SetDefaultCGDparams(OPTstruct &opt);
void SetDefaultPSOparams(OPTstruct &opt);
void InitializeSwarmSizeDependentVariables(int &S, OPTstruct &opt);
void ReinitializePSOparams(int s, OPTstruct &opt);
void ShrinkSwarm(OPTstruct &opt);
void UpdatePSOparams(OPTstruct &opt);
void SetDefaultLCOparams(OPTstruct &opt);
void SetLCOvariables(OPTstruct &opt);
void SetDefaultAULparams(OPTstruct &opt);
void SetDefaultCSAparams(OPTstruct &opt);

void LoadOPTparams(OPTstruct &opt);
void SetupOPTloopBreakFile(OPTstruct &opt);
bool manualOPTbreakQ(OPTstruct &opt);
bool ManualOPTbreakQ(OPTstruct &opt);
void OPTprint(string str, OPTstruct &opt);
inline void startTimer(string descriptor, OPTstruct &opt){ if(opt.printQ){ opt.Timer = high_resolution_clock::now(); OPTprint("StartTimer for " + descriptor + ":",opt); } }
inline void endTimer(string descriptor, OPTstruct &opt){
  if(opt.printQ){
    high_resolution_clock::time_point endTimer = high_resolution_clock::now();
    duration<double>time_span = duration_cast<duration<double>>(endTimer - opt.Timer);
    OPTprint("EndTimer for " + descriptor + ": " + to_string(time_span.count()) + " seconds",opt);
    opt.Timings.push_back((double)time_span.count());
  }
}
void ShiftRot(vector<double> &x,OPTstruct &opt);
void ShiftRotInv(vector<double> &z, OPTstruct &opt);

double EqualityConstraintViolation(vector<double> &y, OPTstruct &opt);
bool EqualityConstraintBoundedVariables(vector<double> &y, OPTstruct &opt);

double alea( double a, double b, OPTstruct &opt );
int alea_integer( int a, int b, OPTstruct &opt );
inline double randSign(OPTstruct &opt){ if(alea(0.,1.,opt)<0.5) return -1.; else return 1.; }
inline void GetMemberShip(double delta, double phi, double deltamax, MemberShipStruct &ms){
  double delta1 = 0.2*deltamax, delta2 = 0.4*deltamax, delta3 = 0.6*deltamax;
  
  if(delta<delta1) ms.Same = 1.;
  else if(delta<delta2) ms.Same = (delta2-delta)/(delta2-delta1);
  else ms.Same = 0.;
  
  if(delta>delta1 && delta<delta2) ms.Near = (delta-delta1)/(delta2-delta1);
  else if(delta>delta2 && delta<delta3) ms.Near = (delta3-delta)/(delta3-delta2);
  else ms.Near = 0.;
  
  if(delta>delta3) ms.Far = 1.;
  else if(delta>delta2) ms.Far = (delta-delta2)/(delta3-delta2);
  else ms.Far = 0.;  
  
  if(phi<0.) ms.Better = -phi;
  else ms.Better = 0.;
  
  ms.Unvaried = 1.-abs(phi);
  
  if(phi>0.) ms.Worse = phi;
  else ms.Worse = 0.;
}

inline void MatrixXdToStdMat(const MatrixXd& EigenMat, vector<vector<double>> &StdMat){
	int n = (int)EigenMat.rows(), m = (int)EigenMat.cols();
	n = StdMat.size(); m = StdMat[0].size();
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) StdMat[i][j] = EigenMat(i,j);
}

double Anneal(double f, int s, OPTstruct &opt);



