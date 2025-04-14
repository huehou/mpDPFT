#include "MPDPFT_HEADER_ONEPEXDFT.h"
#include "Plugin_1pEx.h"
#include "Plugin_OPT.h"
#include "mpDPFT.h"
#include "csa_MIT.hpp"
#include "stdio.h"
#include <cmath>
//#include <math.h>//possible conflict with cmath
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <libgen.h>

#include <string>
#include <vector>
//#include <complex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <limits>
#include <algorithm>
#include <functional>
#include <mutex>
#include <atomic>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/MatrixFunctions>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "Plugin_cec14_test_func.h"

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


using namespace alglib;
using namespace Eigen;

vector<OPTstruct> OPT;
bool TerminateOPT = false;

void OPTprint(string str, OPTstruct &opt){
  if(opt.printQ==0) opt.control << std::setprecision(16) << str << "\n";
  else if(opt.printQ==1){
    cout << std::setprecision(16) << str << endl;
    opt.control << std::setprecision(16) << str << "\n";
  }
  else if(opt.printQ==2){
    cout << std::setprecision(16) << str; cout.flush();
    opt.control << std::setprecision(16) << str;
  }
}

//************** begin CSA main code **************

/*
 *     Example of using CSA minimizing the Schwefel function, `f`. The domain of 
 *     `f` is [-500, 500]^d, where `d` is the dimension. The true minimum of `f` 
 *     is 0.0 at x = (420.9687, ..., 420.9687).
 *
 *
 * Copyright (c) 2009 Samuel Xavier-de-Souza, Johan A.K. Suykens,
 *                    Joos Vandewalle, De ́sire ́ Bolle ́
 * Copyright (c) 2018 Evan Pete Walsh 
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

void cSA(OPTstruct &opt)
{
  SetupOPTloopBreakFile(opt);
  
    //initialize variables
    int D = opt.D;
    double* x = new double[D];
    for(int d=0;d<D;d++) x[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
    if(opt.function==-2){
      vector<double> initAngles = OccNumToAngles(InitializeOccNum(0,opt.ex));
      for(int d=0;d<D;d++){
	if(d<D/2) x[d] = initAngles[d];
	else x[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
      }
    }
    opt.csa.bestf = fcSA((void*)&opt, x);

    //initialize CSA solver
    CSA::Solver<double, double> solver;
    solver.m = opt.threads; //number of (coupled) annealers
    solver.max_iterations = opt.csa.max_iterations;//The maximum number of iterations/steps.
    solver.tgen_initial = opt.csa.tgen_initial;// The initial value of the generation temperature.
    solver.tgen_schedule = opt.csa.tgen_schedule;// Determines the factor that `tgen` is multiplied by during each update.
    solver.tacc_initial = opt.csa.tacc_initial;// The initial value of the acceptance temperature.
    solver.tacc_schedule = opt.csa.tacc_schedule;// Determines the percentage by which `tacc` is increased or decreased during each update.
    solver.desired_variance = opt.csa.desired_variance;// The desired variance of the acceptance probabilities.

    //anneal
    solver.minimize(D, x, fcSA, step, progress, (void*)&opt);
    opt.csa.bestf = fcSA((void*)&opt, x);
    opt.csa.bestx.resize(D); for(int d=0;d<D;d++) opt.csa.bestx[d] = x[d];
    opt.nb_eval = (double)accumulate(opt.csa.nb_eval.begin(),opt.csa.nb_eval.end(),0);

    // Clean up
    delete[] x;
}

//objective function
double fcSA(void* instance, double* x){
  OPTstruct *opt = (OPTstruct *)instance;
  int opt_id = omp_get_thread_num();
  for(int i=0;i<opt->D;i++) opt->csa.X[opt_id].x[i] = x[i];
  return GetFuncVal( opt_id, opt->csa.X[opt_id].x, opt->function, *opt );
}

// This function will take a random step from `x` to `y`. The value `tgen`,
// the "generation temperature", determines the variance of the distribution of
// the step. `tgen` will decrease according to fixed a schedule throughout the
// annealing process, which corresponds to a decrease in the variance of steps.
void step(void* instance, double* y, const double* x, float tgen){
  OPTstruct *opt = (OPTstruct *)instance;
  int opt_id = omp_get_thread_num();
  int D = opt->D;
  double stepsize, TGEN = (double)tgen;
  opt->csa.averageStepSize[opt_id] = 0.;
    
  for(int d=0;d<D;d++){  		    
    if(alea(0.,1.,*opt)<opt->csa.LargeStepProb) stepsize = alea(0.,1.,*opt) * (double)(opt->csa.tgen_initial);
    else stepsize = TGEN;
    stepsize = max(stepsize,TGEN);
    stepsize = max(stepsize,opt->csa.minimal_step*(opt->SearchSpace[d][1]-opt->SearchSpace[d][0]));
    stepsize = min(stepsize,1.);
    opt->csa.averageStepSize[opt_id] += stepsize;
    y[d] = x[d] + stepsize*alea( opt->SearchSpace[d][0]-x[d], opt->SearchSpace[d][1]-x[d], *opt );
  }
  opt->csa.averageStepSize[opt_id] /= (double)D;
  
  if(opt->function==-2){
    int D2 = D/2;
    vector<double> ang(D2), occ(D2);
    for(int d=0;d<D2;d++) ang[d] = y[d];
    occ = AnglesToOccNum(ang,opt->ex);
    NearestProperOccNum(occ,opt->ex);
    ang = OccNumToAngles(occ);
    for(int d=0;d<D2;d++) y[d] = ang[d];
    double TwoPi = 2.*3.141592653589793;
    for(int d=D2;d<D;d++){//keep in the box
      while(y[d]<opt->SearchSpace[d][0]) y[d] += TwoPi;
      while(y[d]>opt->SearchSpace[d][1]) y[d] -= TwoPi;
    }
  }  

}

// This will receive progress updates from the CSA process and print updates to the terminal.
void progress(void* instance, double cost, float stepsize, float prob, int opt_id, int iter){
  OPTstruct *opt = (OPTstruct *)instance;
  double TotalNumFuncEvals = 0.; for(int t=0;t<opt->threads;t++) TotalNumFuncEvals += (double)opt->csa.nb_eval[t];
  if(iter>0) OPTprint("\n cSA[thread=" + to_string(opt_id) + "]: function evaluations = " + to_string_with_precision(TotalNumFuncEvals,3) + "/" + to_string_with_precision(opt->csa.max_funcEvals,3) + ", bestf = " + to_string_with_precision(cost,16) + ", stepsize = " + to_string_with_precision((double)stepsize,8) + ", acceptance probability = " + to_string_with_precision((double)prob,8) + "\n                bestx = " + vec_to_str_with_precision(opt->currentBestx,8),*opt);
  if(manualOPTbreakQ(*opt)) opt->TerminateQ = 1;
}

//************** end CSA main code **************








//************** begin MIT CMA main code **************

void SetDefaultCMAparams(OPTstruct &opt){
	opt.ActiveOptimizer = 106;
	opt.printQ = 1;
	opt.reportQ = 2;
	
	SetDefaultSearchSpace(opt);
	
	opt.cma.runs = 10;
	opt.cma.generationMax = 1000*(int)sqrt((double)opt.D);
	opt.cma.popExponent = 5;
	opt.cma.PopulationDecayRate = 1.7;
	opt.cma.muRatio = 0.5;
	opt.cma.VarianceCheck = min(min(10*opt.D,100*(int)sqrt((double)opt.D)),(int)(0.2*(double)opt.cma.generationMax));
	opt.stallCheck = max(10*opt.D,opt.cma.VarianceCheck);
	if(opt.anneal>0. || opt.homotopy>1) opt.stallCheck = opt.cma.generationMax;
	opt.cma.WeightScenario = 2;
	opt.cma.InitStepSizeFactor = 0.3;
	
	InitializePopulationSizeCMA(opt);
}

void InitializePopulationSizeCMA(OPTstruct &opt){
	opt.cma.populationSize = max(4,(int)POW(2.,opt.cma.popExponent)*(4+3*(int)(log((double)opt.D))));
	opt.cma.InitialPopulationSize = opt.cma.populationSize;
	opt.cma.NewPopulationSize = opt.cma.populationSize;
}

void CMA(OPTstruct &opt){//Hansen2016, appendix A for WeightScenario==2
	OPTprint("\n ***** Enter CMA() ... *****",opt);
	InitializeCMA(opt);//(again) after potential manual changes of hyperparameters
	
	while(!ManualOPTbreakQ(opt)){
		SampleCMA(opt);
		ReportCMA(opt);
		if(opt.cma.exit) break;
		UpdateMeanCMA(opt);
		UpdateEvolutionPathsCMA(opt);
		UpdateCovarianceCMA(opt);
		UpdateCMA(opt);
	}
	
	OPTprint("\n ***** ... exit CMA() *****",opt);
}

void InitializeCMA(OPTstruct &opt){
	
	Eigen::setNbThreads(1);
	
	opt.nb_eval = 0.;
	
	opt.cma.generation = 1;
	opt.cma.stall = 0;
	opt.cma.report.resize(3,"");
	
	//fix number of runs (populations)
	if(opt.threads>1){
		if(opt.cma.runs<=opt.threads) opt.cma.runs = opt.threads;
		else if(opt.cma.runs%opt.threads>0) opt.cma.runs = max(opt.threads,opt.cma.runs-(opt.cma.runs%opt.threads)+opt.threads);
		if(opt.function+2014>=1 && opt.function+2014<=30) opt.threads = 1;
	}
	
	if(opt.cma.runs==1) opt.cma.CheckPopVariance = 0.;
	
	//initialize further default values
	opt.cma.InitPenaltyFactor = vector<double>(opt.cma.runs,1.0e+3);
	opt.cma.betac = vector<double>(opt.cma.runs,1.);
	opt.cma.cm = vector<double>(opt.cma.runs,1.);//learning rate for the update of the mean
	opt.cma.MeanFromAllWeights = vector<bool>(opt.cma.runs,false);//true (false): update the mean with all (only the positive) weights
	opt.cma.penaltyFactor = opt.cma.InitPenaltyFactor;
	opt.cma.stepSize = vector<double>(opt.cma.runs,opt.cma.InitStepSizeFactor*opt.SearchSpaceExtent);
	//opt.cma.penaltyFactor.clear(); opt.cma.penaltyFactor.resize(opt.cma.runs); for(int p=0;p<opt.cma.runs;p++) opt.cma.penaltyFactor[p] = opt.cma.InitPenaltyFactor[p];
	
	if(opt.cma.PickRandomParamsQ) PickParamsCMA(opt);
	
	InitializePopulationSizeCMA(opt);
	
	opt.cma.AbortQ = vector<int>(opt.cma.runs,0);
	opt.cma.mean = vector<vector<double>>(opt.cma.runs,opt.SearchSpaceCentre);
	opt.cma.meanOld = vector<vector<double>>(opt.cma.runs,opt.SearchSpaceCentre);
	opt.cma.Covariance = vector<MatrixXd>(opt.cma.runs,MatrixXd::Identity(opt.D,opt.D));
	opt.cma.D = vector<MatrixXd>(opt.cma.runs,MatrixXd::Identity(opt.D,opt.D));
	opt.cma.B = vector<MatrixXd>(opt.cma.runs,MatrixXd::Identity(opt.D,opt.D));
	opt.cma.CovInvSqrt = vector<MatrixXd>(opt.cma.runs,MatrixXd::Identity(opt.D,opt.D));
	opt.cma.EVDcount = vector<int>(opt.cma.runs,0);
	opt.cma.MaxDelay = vector<int>(opt.cma.runs,1);
	opt.cma.psigma = vector<vector<double>>(opt.cma.runs,vector<double>(opt.D,0.));
	opt.cma.pc = vector<vector<double>>(opt.cma.runs,vector<double>(opt.D,0.));
	opt.cma.psigmaNorm = vector<double>(opt.cma.runs,0.);
	opt.cma.pop = vector<vector<vector<double>>>(opt.cma.runs,vector<vector<double>>(opt.cma.populationSize,vector<double>(opt.D)));
	opt.cma.pop2 = vector<vector<vector<double>>>(opt.cma.runs,vector<vector<double>>(opt.cma.populationSize,vector<double>(opt.D)));
	opt.cma.f = vector<vector<double>>(opt.cma.runs,vector<double>(opt.cma.populationSize));
	opt.cma.bestfVec.resize(opt.cma.runs,1.0e+300);
	opt.cma.worstfVec.resize(opt.cma.runs,-1.0e+300);
	opt.cma.bestxVec = vector<vector<double>>(opt.cma.runs,vector<double>(opt.D,0.));
	opt.cma.bestx = vector<double>(opt.D,1.23456789);
	opt.cma.spread = vector<double>(opt.cma.runs,0.);
	opt.cma.history = vector<vector<double>>(opt.cma.runs,vector<double>(0,0.));

	opt.maxEC = vector<vector<double>>(opt.cma.runs,vector<double>(opt.NumEC,0.));
	opt.maxIC = vector<vector<double>>(opt.cma.runs,vector<double>(opt.NumIC,0.));
	if((int)opt.PenaltyMethod[0]>0){
		opt.ALmu = vector<vector<double>>(opt.cma.runs,vector<double>(opt.cma.populationSize,opt.PenaltyMethod[1]));
		opt.ALmu2 = vector<vector<double>>(opt.cma.runs,vector<double>(opt.cma.populationSize,opt.PenaltyMethod[1]));
		opt.ALlambda = vector<vector<vector<double>>>(opt.NumEC+opt.NumIC,vector<vector<double>>(opt.cma.runs,vector<double>(opt.cma.populationSize,opt.PenaltyMethod[2])));
		opt.ALlambda2 = vector<vector<vector<double>>>(opt.NumEC+opt.NumIC,vector<vector<double>>(opt.cma.runs,vector<double>(opt.cma.populationSize,opt.PenaltyMethod[2])));
	}

	opt.cma.ExpectedValue = sqrt((double)opt.D)*(1.-1./((double)(4*opt.D)) + 1./((double)(21*opt.D*opt.D)));
	opt.cma.alphacov = 2.;
	
	double alphamu, alphamueff, alphaposdef, minalpha, RawPosWeightSum = 0., RawNegWeightSum = 0., PosWeightSum = 0., NegWeightSum = 0.;
	
	if(opt.cma.WeightScenario==0) opt.cma.muRatio = 0.25;
	opt.cma.mu = min(opt.cma.populationSize-1,max(2,(int)(opt.cma.muRatio*(double)opt.cma.populationSize)));
	
	if(opt.cma.WeightScenario==0){//parameters for constant weights
		
		opt.cma.weights = vector<double>(opt.cma.mu,1./((double)opt.cma.mu));
		opt.cma.mueff = (double)opt.cma.mu;
	}
	else if(opt.cma.WeightScenario==1){//parameters for Wikipedia(==Hansen2016, MatLab-Code) weights
		double weightSum = 0.;
		opt.cma.weights.clear(); opt.cma.weights.resize(0);
		for(int c=0;c<opt.cma.mu;c++){
			opt.cma.weights.push_back(log((double)opt.cma.mu+0.5)-log(1.+(double)c));
			weightSum += opt.cma.weights[c];
		}
		for(int c=0;c<opt.cma.mu;c++) opt.cma.weights[c] /= weightSum;
		opt.cma.mueff = 0.;
		for(int c=0;c<opt.cma.mu;c++) opt.cma.mueff += opt.cma.weights[c]*opt.cma.weights[c];
		opt.cma.mueff = 1./opt.cma.mueff;
		opt.cma.c1 = opt.cma.alphacov/(POW((double)opt.D+1.3,2)+opt.cma.mueff);
		opt.cma.cmu = min(1.-opt.cma.c1, opt.cma.alphacov * (0.25+opt.cma.mueff+1./opt.cma.mueff-2.) / (POW((double)opt.D+2.,2)+0.5*opt.cma.alphacov*opt.cma.mueff));
	}
	else if(opt.cma.WeightScenario==2){//parameters for Hansen2016 with negative weights
		//... negative weights will be triggered only for opt.D>5 or so, depending on population size
		opt.cma.weights.clear(); opt.cma.weights.resize(0);
		double mueffm = 0.;
		opt.cma.mueff = 0.;
		//set up preliminary convex shape of weights
		double crossing = (double)opt.cma.mu/(double)opt.cma.populationSize;
		for(int c=0;c<opt.cma.populationSize;c++){
			opt.cma.weights.push_back(log(crossing*((double)opt.cma.populationSize+1.))-log(1.+(double)c));
			//OPTprint(to_string(opt.cma.weights[c]),opt);
			if(c<opt.cma.mu){
				RawPosWeightSum += opt.cma.weights[c];
				opt.cma.mueff += opt.cma.weights[c]*opt.cma.weights[c];
			}
			else{
				RawNegWeightSum -= opt.cma.weights[c];
				mueffm += opt.cma.weights[c]*opt.cma.weights[c];
			}
		}
		if(opt.cma.weights[opt.cma.mu-1]<0. || opt.cma.weights[opt.cma.mu]>0.){
			OPTprint("InitializeCMA: wrong weights !!! " + to_string(opt.cma.weights[opt.cma.mu-1]) + " " + to_string(opt.cma.weights[opt.cma.mu]) + " ",opt);
		}
		opt.cma.mueff = RawPosWeightSum*RawPosWeightSum/opt.cma.mueff;
		mueffm = RawNegWeightSum*RawNegWeightSum/mueffm;
		opt.cma.c1 = opt.cma.alphacov/(POW((double)opt.D+1.3,2)+opt.cma.mueff);
		opt.cma.cmu = min(1.-opt.cma.c1, opt.cma.alphacov * (0.25+opt.cma.mueff+1./opt.cma.mueff-2.) / (POW((double)opt.D+2.,2)+0.5*opt.cma.alphacov*opt.cma.mueff));
		alphamu = 1.+opt.cma.c1/opt.cma.cmu;
		alphamueff = 1.+2.*mueffm/(opt.cma.mueff+2.);
		alphaposdef = (1.-opt.cma.c1-opt.cma.cmu)/((double)opt.D*opt.cma.cmu);
		minalpha = min(min(alphamu,alphamueff),alphaposdef);
		//finalize weights
		for(int c=0;c<opt.cma.mu;c++){
			opt.cma.weights[c] /= RawPosWeightSum;
			PosWeightSum += opt.cma.weights[c];
		}
		for(int c=opt.cma.mu;c<opt.cma.populationSize;c++){
			opt.cma.weights[c] *= minalpha/RawNegWeightSum;
			NegWeightSum += opt.cma.weights[c];
		}
	}
	
	opt.cma.WeightSum = PosWeightSum + NegWeightSum;
	opt.cma.csigma = (opt.cma.mueff+2.)/((double)opt.D+opt.cma.mueff+5.);
	opt.cma.dsigma = 1. + 2.*max(0., sqrt((opt.cma.mueff-1.)/((double)opt.D+1.))-1.) + opt.cma.csigma;
	double alphac = pow(10.,1.-pow((double)opt.D,-1./3.));
	
	opt.cma.cc.clear(); opt.cma.cc.resize(opt.cma.runs);
	for(int p=0;p<opt.cma.runs;p++) opt.cma.cc[p] = (alphac+pow(opt.cma.mueff/((double)opt.D),opt.cma.betac[p])) / (pow((double)opt.D,opt.cma.betac[p])+alphac+2.*pow(opt.cma.mueff/((double)opt.D),opt.cma.betac[p]));//Hansen2016_Eq.(61) with hyperparameter betac<=1
	
	OPTprint(" ----- InitializeCMA ----- ",opt);
	OPTprint("       function ID                       = " + to_string(opt.function),opt);
	OPTprint("       search space dimension            = " + to_string(opt.D),opt);
	OPTprint("       parallel threads                  = " + to_string(opt.threads),opt);
	OPTprint(" ----- CMA hyperparameters (user) ----- ",opt);
	OPTprint("       runs (#populations)               = " + to_string(opt.cma.runs),opt);
	OPTprint("       popExponent                       = " + to_string(opt.cma.popExponent),opt);
	OPTprint("       --> (initial) population size     = " + to_string(opt.cma.populationSize),opt);
	OPTprint("       muRatio                           = " + to_string(opt.cma.muRatio),opt);
	OPTprint("       VarianceCheck                     = " + to_string(opt.cma.VarianceCheck),opt);
	OPTprint("       PopulationDecayRate               = " + to_string(opt.cma.PopulationDecayRate),opt);
	OPTprint("       elitism                           = " + to_string(opt.cma.elitism),opt);
	OPTprint("       Constraints                       = " + to_string(opt.cma.Constraints),opt);
	OPTprint("       DelayEigenDecomposition           = " + to_string(opt.cma.DelayEigenDecomposition),opt);
	OPTprint("       WeightScenario                    = " + to_string(opt.cma.WeightScenario),opt);
	OPTprint("       MeanFromAllWeights (best pop)     = " + to_string(opt.cma.MeanFromAllWeights[opt.cma.bestp]),opt);
	OPTprint("       betac (best pop)                  = " + to_string(opt.cma.betac[opt.cma.bestp]),opt);
	OPTprint("       InitPenaltyFactor (best pop)      = " + to_string(opt.cma.InitPenaltyFactor[opt.cma.bestp]),opt);
	OPTprint(" ----- CMA miscellaneous parameters (fixed) ----- ",opt);
	OPTprint("       mu (#parents)                     = " + to_string(opt.cma.mu),opt);
	OPTprint("       mueff                             = " + to_string_with_precision(opt.cma.mueff,4),opt);
	OPTprint("       c1                                = " + to_string_with_precision(opt.cma.c1,4),opt);
	OPTprint("       cmu                               = " + to_string_with_precision(opt.cma.cmu,4),opt);
	OPTprint("       csigma                            = " + to_string_with_precision(opt.cma.csigma,4),opt);
	OPTprint("       cc (best pop)                     = " + to_string_with_precision(opt.cma.cc[opt.cma.bestp],4) + " (ideally in [" + to_string_with_precision(2./((double)opt.D),3) + "," + to_string_with_precision(1./sqrt((double)opt.D),3) + "])",opt);
	OPTprint("       cm (best pop)                     = " + to_string_with_precision(opt.cma.cm[opt.cma.bestp],4),opt);
	if(opt.cma.WeightScenario==2){
		OPTprint("       alphamu                           = " + to_string_with_precision(alphamu,6) + " ~ " + to_string_with_precision(NegWeightSum,6) + " ~ 0 ?",opt);
		OPTprint("       alphamueff                        = " + to_string_with_precision(alphamueff,4),opt);
		OPTprint("       alphaposdef                       = " + to_string_with_precision(alphaposdef,4),opt);
	}
	OPTprint("       weights        = " + vec_to_str_with_precision(opt.cma.weights,6),opt);

}

void PickParamsCMA(OPTstruct &opt){
	//opt.cma.InitBias.clear(); opt.cma.InitBias.resize(opt.cma.runs);
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		//for(int d=0;d<opt.D;d++) opt.cma.InitBias[p].push_back(alea(-0.5,0.5,opt)*opt.searchSpaceExtent[d]);
		opt.cma.stepSize[p] *= 2.*pow(10.,-alea(0.,2.,opt));
		opt.cma.cm[p] = alea(0.95,1.05,opt);//alea(0.1,2.,opt);//
		opt.cma.betac[p] = alea(0.8,1.,opt);//alea(0.1,1.,opt);//
		opt.cma.InitPenaltyFactor[p] = pow(10.,alea(3.,6.,opt));//pow(10.,alea(0.,6.,opt));//
		if(alea(0.,1.,opt)<0.5) opt.cma.MeanFromAllWeights[p] = true;
	}
}

void SampleCMA(OPTstruct &opt){
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			//for each population, sample new chromosomes
			SampleMultivariateNormalCMA(p,opt);
			ConstrainCMA(p,opt);
			double bestf = 1.0e+300, worstf = -bestf;
			for(int c=0;c<opt.cma.populationSize;c++){
				opt.cma.f[p][c] = GetFuncVal(p*opt.cma.populationSize+c, opt.cma.pop[p][c], opt.function, opt);
				if(opt.cma.f[p][c]>worstf) worstf = opt.cma.f[p][c];
				if(opt.cma.f[p][c]<bestf) bestf = opt.cma.f[p][c];
				if(worstf-bestf>opt.cma.spread[p]) opt.cma.spread[p] = worstf-bestf;
			}
			RepairCMA(p,opt);
			//sort all chromosomes
			int count = 0;
			for(auto c: sort_indices(opt.cma.f[p])){
				opt.cma.pop2[p][count] = opt.cma.pop[p][c];//assign new (and ordered with ascending f) chromosomes to pop2
				if((int)opt.PenaltyMethod[0]>0){
					for(int i=0;i<opt.NumEC+opt.NumIC;i++){
						opt.ALlambda2[i][p][count] = opt.ALlambda[i][p][c];
						opt.ALmu2[p][count] = opt.ALmu[p][c];
					}
				}
				count++;
			}
			if((int)opt.PenaltyMethod[0]>0){
				for(int i=0;i<opt.NumEC+opt.NumIC;i++){
					opt.ALlambda2[i][p].swap(opt.ALlambda[i][p]);
					opt.ALmu2[p].swap(opt.ALmu[p]);
				}
			}
			sort(opt.cma.f[p].begin(),opt.cma.f[p].end());//now, f[p] is in the order consistent with opt.cma.pop2[p]
			//elitism: retain best chromosomes
			if(opt.cma.elitism && opt.cma.generation>1){
				if(opt.cma.f[p][0]>opt.cma.bestfVec[p]){
					double test = GetFuncVal(p*opt.cma.populationSize+0, opt.cma.bestxVec[p], opt.function, opt);//recalculate (important if (e.g.) penalties are generation-dependent)
					if(test<opt.cma.f[p][0]){
						opt.cma.f[p][0] = test;
						opt.cma.pop2[p][0] = opt.cma.bestxVec[p];
					}
				}
				double fAtMean = GetFuncVal(p*opt.cma.populationSize+0, opt.cma.mean[p], opt.function, opt);
				if(opt.cma.f[p][0]>fAtMean){
					opt.cma.f[p][0] = fAtMean;
					opt.cma.pop2[p][0] = opt.cma.mean[p];
				}			
			}
			//collect best result from each population
			opt.cma.bestfVec[p] = opt.cma.f[p][0];
			opt.cma.bestxVec[p] = opt.cma.pop2[p][0];
		}
	}
	opt.cma.pop2.swap(opt.cma.pop);
	//if(opt.cma.generation==1) MatrixToFile(opt.cma.pop[0],"Samples.dat",16);
	opt.nb_eval += (double)(opt.cma.runs*opt.cma.populationSize);
}

void SampleMultivariateNormalCMA(int p, OPTstruct &opt){//sample d-dimensional points around the d-dimensional mean 

	// Sample from standard normal distribution -> the matrix 'Samples' holds d-dimensional vectors in its COLUMNS!
	MatrixXd Samples(opt.D,opt.cma.populationSize);
	for(int c=0;c<opt.cma.populationSize;++c) for(int d=0;d<opt.D;++d) Samples(d,c) = opt.RNnormal(opt.MTGEN);
	
    // Transform standard normal samples to multivariate normal samples
    Samples = opt.cma.stepSize[p] * opt.cma.B[p] * opt.cma.D[p] * Samples;
	
    //Shift and store sampled points in population matrix, which holds one d-dimensional vector in each ROW!
    for(int d=0;d<opt.D;++d){
		for(int c=0;c<opt.cma.populationSize;++c){
			opt.cma.pop[p][c][d] = opt.cma.mean[p][d] + Samples(d,c);
			if(opt.anneal>0. && opt.AnnealType==0) opt.cma.pop[p][c][d] += Anneal(0.,p*opt.cma.populationSize+c,opt);
		}
	}

	//cout << "SampleMultivariateNormalCMA " << vec_to_str(opt.cma.pop[p][0]) << endl;
	
	//if(opt.cma.generation==1 && opt.cma.PickRandomParamsQ) for(int d=0;d<opt.D;++d) for(int c=0;c<opt.cma.populationSize;++c) opt.cma.pop[p][c][d] += opt.cma.InitBias[p][d];//doesn't seem to do any good...
}

void ConstrainCMA(int p, OPTstruct &opt){
	for(int c=0;c<opt.cma.populationSize;c++){
		if(opt.function==-1 || opt.function==-2){
			double Pi = 3.141592653589793, TwoPi = 2.*Pi;
			if(opt.function==-2) CMAKeepInBox(p,c,opt);
			//properize positions:
			int L = opt.ex.L;
			vector<double> ang(L), occ(L);
			for(int d=0;d<L;d++) ang[d] = opt.cma.pop[p][c][d];
			occ = AnglesToOccNum(ang,opt.ex);//properized internally
			if(ABS(opt.ex.Abundances[0]-accumulate(occ.begin(),occ.end(),0.))>10*opt.ex.Abundances[0]){ cout << "ConstrainCMA: NearestProperOccNum (ex.settings[5]==" << opt.ex.settings[5] << ") Warning !!! " << accumulate(occ.begin(),occ.end(),0.) << " != " << opt.ex.Abundances[0] << endl; usleep(1000000); }
			ang = OccNumToAngles(occ);//revert to angles
			for(int d=0;d<L;d++) opt.cma.pop[p][c][d] = ang[d];//will automatically be within [0,Pi]
			if(opt.function==-2){	  
				for(int d=L;d<opt.D;d++){//keep Phases within [0,TwoPi]  
					while(opt.cma.pop[p][c][d]<0.){ opt.cma.pop[p][c][d] += TwoPi; }
					while(opt.cma.pop[p][c][d]>TwoPi){ opt.cma.pop[p][c][d] -= TwoPi; }
				}
			}
		}
		else if(opt.function==100){//mutually unbiased bases
			int NB = (int)(opt.AuxParams[1]+0.5), NV = (int)(opt.AuxParams[2]+0.5), dim = NV;
			for(int b=0;b<NB;b++){
				for(int w=0;w<NV;w++){
					double invnormw = 0.;
					for(int d=0;d<dim;d++){
						double wR = 0., wI = 0.;
						if(b>0){
							wR = opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2];
							if(d>0) wI = opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2+1];
							else{//fix global phase of each vector by putting imaginary part of first component to zero
								opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2+1] = 0.;
							}
						}
						else{//fix first basis to canonical basis
							opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2+1] = 0.;
							if(d==w){
								opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2] = 1.;
								wR = 1.;
							}
							else opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2] = 0.;
						}
						invnormw += wR*wR+wI*wI;
					}
					invnormw = 1./sqrt(invnormw);
					for(int d=0;d<dim;d++){
						opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2] *= invnormw;
						opt.cma.pop[p][c][b*NV*dim*2+w*dim*2+d*2+1] *= invnormw;
					}
				}
			}
		}
		else if(opt.cma.Constraints==1) CMAKeepInBox(p,c,opt);
	}
}

void CMAKeepInBox(int p, int c, OPTstruct &opt){
	for(int d=0;d<opt.D;d++){
		if ( opt.cma.pop[p][c][d] < opt.SearchSpace[d][0] ){ opt.cma.pop[p][c][d] = opt.SearchSpace[d][0]; }
		if ( opt.cma.pop[p][c][d] > opt.SearchSpace[d][1] ){ opt.cma.pop[p][c][d] = opt.SearchSpace[d][1]; }
	}
}

void RepairCMA(int p, OPTstruct &opt){
	//box constraint penalty --- 2: smooth (useful if optimizer may be on boundary) --- 3: sharp (useful if optimizer likely not on boundary)
	if(opt.cma.Constraints>=2 && opt.cma.Constraints<=4){
		//penalize all populations, but monitor total penalty only for the best population
		if(p==opt.cma.bestp) opt.cma.TotalPenalty = 0.;
		for(int c=0;c<opt.cma.populationSize;c++){
			double penalty = 0.;
			for(int d=0;d<opt.D;++d){//accumulate penalties from all search space dimensions
				double x = opt.cma.pop[p][c][d];
				double addon = 0.;
				if(x < opt.SearchSpace[d][0]){
					if(opt.cma.Constraints==2) addon = POW((opt.SearchSpace[d][0]-x)/opt.searchSpaceExtent[d],2);
					else if(opt.cma.Constraints==3) addon = sqrt((opt.SearchSpace[d][0]-x)/opt.searchSpaceExtent[d]);
					else addon = 1.0e+100;
				}
				else if(x > opt.SearchSpace[d][1]){
					if(opt.cma.Constraints==2) addon = POW((x-opt.SearchSpace[d][1])/opt.searchSpaceExtent[d],2);
					else if(opt.cma.Constraints==3) addon = sqrt((x-opt.SearchSpace[d][1])/opt.searchSpaceExtent[d]);
					else addon = 1.0e+100;
				}
				penalty += addon;
			}
			penalty *= opt.cma.penaltyFactor[p]*opt.cma.spread[p];
			if(p==opt.cma.bestp) opt.cma.TotalPenalty += penalty;
			opt.cma.f[p][c] += penalty;
		}
		
	}
}

void ReportCMA(OPTstruct &opt){
	if(opt.printQ>-2){//determine whether the report will be printed
    	int gen = opt.cma.generation;
		double expon = floor(log10((double)gen));
		opt.printQ = -1;
    	for(int i=1;i<10;i++){
      		if(i*(int)pow(10.,expon)==gen || gen%1000==0 || gen==opt.cma.generationMax-1){
			  	opt.printQ = 1;
			  	break;
      		}
    	}
    	if(opt.printQ==-1){ cout << "."; cout.flush(); }
	}
    
    //OPTprint("\n\n ***** begin CMA report *****",opt);
	double bestfOld = opt.currentBestf;
	vector<double> bestfVecSorted(opt.cma.bestfVec);
	int count = 0;
	for(auto p: sort_indices(opt.cma.bestfVec)){
		bestfVecSorted[count] = opt.cma.bestfVec[p];
		if(count==0){
			opt.cma.bestf = opt.cma.bestfVec[p];
			opt.cma.bestx =	opt.cma.pop[p][0];
			opt.cma.bestp = p;
			if(opt.cma.bestf<opt.currentBestf){
				opt.NewbestFoundQ = true;
				opt.currentBestf = opt.cma.bestf;
				opt.currentBestx = opt.cma.bestx;
				opt.currentBestp = opt.cma.bestp;
				opt.cma.report[0] = "  ... new bestf found : " + to_string_with_precision(opt.currentBestf,16) + " @ x=" + vec_to_str_with_precision(opt.currentBestx,16) + "\n";
			}
			else opt.NewbestFoundQ = false;
		}
		count++;
	}

	opt.cma.exit = false;
	bool varianceConvergedQ = false;
	double Mean = Norm(opt.cma.mean[opt.cma.bestp]);
	double MeanOld = Norm(opt.cma.meanOld[opt.cma.bestp]);
	opt.OldVariance = opt.CurrentVariance;
	
	if(RelDiff(bestfOld,opt.currentBestf)>1.0e-12){
		if(opt.epsf>0.) varianceConvergedQ = VarianceConvergedQ(opt);
		opt.cma.stall = 0;
	}

	if( opt.cma.generation>opt.cma.VarianceCheck && RelDiff(opt.OldVariance,opt.CurrentVariance)<1.0e-12 /* && RelDiff(Mean,MeanOld)<1.0e-12*/){
		opt.cma.stall++;
		if(opt.cma.stall==opt.stallCheck){
			opt.cma.report[1] += "\n ||||| CMA has been stalling for " + to_string(opt.stallCheck) + " generations -> exit @ CMA-generation = " + to_string(opt.cma.generation) + "/" + to_string(opt.cma.generationMax);
			opt.cma.exit = true;
		}
	}
	
	if(opt.NewbestFoundQ || opt.printQ==1 || varianceConvergedQ){
		if(varianceConvergedQ){
			opt.cma.report[1] += "\n ||||| Variance converged @ CMA-generation = " + to_string(opt.cma.generation) + "/" + to_string(opt.cma.generationMax);
			opt.cma.exit = true;
		}
		if(opt.NewbestFoundQ) opt.NewbestFoundQ = false;//reset for next generation
	}
	
	if(opt.cma.generation==opt.cma.generationMax){
		opt.cma.report[1] += "\n ##### Warning: Exit @ CMA-generation = " + to_string(opt.cma.generation) + "/" + to_string(opt.cma.generationMax);
		opt.cma.exit = true;
	}
	else if(opt.homotopy>1) opt.cma.exit = false;

	if(opt.cma.elitism) if( opt.cma.bestf - opt.cma.bestfVec[opt.cma.bestp] > 1.0e-12 ) OPTprint("ReportCMA: Warning !!! bestf lost?",opt);
	
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==1){
			//opt.cma.report[1] += "\n   ||| run # " + to_string(p+1) + " terminated @ f=" + to_string_with_precision(opt.currentBestf,16);
			opt.cma.AbortQ[p] = 2;
		}
	}	
	
	opt.cma.report[2] += opt.report;
	opt.report = "";

	OPTprint("\n generation " + to_string(opt.cma.generation) + "/" + to_string(opt.cma.generationMax),opt);
	if(opt.cma.generation>1){
		int activepops = 0; for(int p=0;p<opt.cma.runs;p++) if(opt.cma.AbortQ[p]==0) activepops++;
		if(activepops==0) opt.cma.exit = true;
		OPTprint(" ----- CMA params ----- ",opt);
		OPTprint("       function ID                       = " + to_string(opt.function),opt);
		OPTprint("       search space dimension            = " + to_string(opt.D),opt);
		OPTprint("       parallel threads                  = " + to_string(opt.threads),opt);
		OPTprint("       VarianceCheck                     = " + to_string(opt.cma.VarianceCheck),opt);
		OPTprint(" ----- CMA stats ----- ",opt);
		OPTprint("       # active populations              = " + to_string(activepops) + "/" + to_string(opt.cma.runs),opt);
		OPTprint("       population size                   = " + to_string(opt.cma.populationSize),opt);
		OPTprint("       new population size (suggested)   = " + to_string(opt.cma.NewPopulationSize),opt);
		OPTprint("       Mean    (norm)                    = " + to_string_with_precision(Mean,16),opt);
		OPTprint("       MeanOld (norm)                    = " + to_string_with_precision(MeanOld,16),opt);
		string ov = "pending...", cv = "pending...";
		if(opt.varianceUpdatedQ){
			ov = to_string_with_precision(opt.OldVariance,4);
			cv = to_string_with_precision(opt.CurrentVariance,4);
		}
		OPTprint("       OldVariance (global history)      = " + cv,opt);
		OPTprint("       CurrentVariance (global history)  = " + cv,opt);
		OPTprint("       VarianceThreshold                 = " + to_string_with_precision(opt.VarianceThreshold,4),opt);
		OPTprint("       # FuncEvals                       = " + to_string_with_precision(opt.nb_eval,2),opt);
		OPTprint(" ----- CMA stats of best population (p=" + to_string(opt.cma.bestp) + ") ----- ",opt);
		OPTprint("       MeanFromAllWeights                = " + to_string(opt.cma.MeanFromAllWeights[opt.cma.bestp]),opt);
		OPTprint("       betac                             = " + to_string(opt.cma.betac[opt.cma.bestp]),opt);
		OPTprint("       InitPenaltyFactor                 = " + to_string(opt.cma.InitPenaltyFactor[opt.cma.bestp]),opt);
		OPTprint("       cc                                = " + to_string_with_precision(opt.cma.cc[opt.cma.bestp],4),opt);
		OPTprint("       cm                                = " + to_string_with_precision(opt.cma.cm[opt.cma.bestp],4),opt);
		OPTprint("       TotalPenalty                      = " + to_string_with_precision(opt.cma.TotalPenalty,4),opt);			
		OPTprint("       EvolutionPath ps (norm)           = " + to_string_with_precision(Norm(opt.cma.psigma[opt.cma.bestp]),4),opt);
		OPTprint("       EvolutionPath pc (norm)           = " + to_string_with_precision(Norm(opt.cma.pc[opt.cma.bestp]),4),opt);
		OPTprint("       Average StepSize                  = " + to_string_with_precision(opt.cma.stepSize[opt.cma.bestp],4),opt);
		OPTprint("       MaxDelay (covariance update)      = " + to_string_with_precision(opt.cma.MaxDelay[opt.cma.bestp],4),opt);
		OPTprint("       spread                            = " + to_string_with_precision(opt.cma.spread[opt.cma.bestp],4),opt);
	}
	
	if(opt.printQ>-2){
		if(opt.cma.exit) opt.printQ = 1;
		if(opt.printQ==1){
			for(int r=0;r<opt.cma.report.size();r++){
				if(opt.cma.report[r]!=""){
					OPTprint(opt.cma.report[r],opt);
					opt.cma.report[r] = "";
				}
			}
		}
	}
	
	sort(bestfVecSorted.begin(),bestfVecSorted.end());
	if(opt.cma.runs>6) OPTprint(" bestfVec(sorted) = " + partial_vec_to_str_with_precision(bestfVecSorted,0,3,16) + " .......... " +  partial_vec_to_str_with_precision(bestfVecSorted,bestfVecSorted.size()-4,bestfVecSorted.size()-1,16) + "\n",opt);
	else OPTprint(" bestfVec(sorted) = " + vec_to_str_with_precision(bestfVecSorted,16) + "\n",opt);
	if(opt.ReportX) OPTprint(" current bestf @ x = " + vec_to_str_with_precision(opt.cma.bestx,6) + "\n",opt);
	if(opt.cma.CheckPopVariance>0.){
		int P;
		if(opt.cma.CheckPopVariance > 1.) P = (int)opt.cma.CheckPopVariance;//absolute number of populations
		else P = max(2,(int)(opt.cma.CheckPopVariance*(double)opt.cma.runs));//fraction of populations
		double PopVariance = 0., PopMean = 0.;
		for(int p=0;p<P;p++) PopMean += bestfVecSorted[p]/((double)P);
		for(int p=0;p<P;p++) PopVariance += (bestfVecSorted[p]-PopMean)*(bestfVecSorted[p]-PopMean)/((double)P);
		double threshold = (double)opt.cma.runs*opt.cma.CheckPopVariance*opt.VarianceThreshold;
		if(opt.cma.generation > opt.cma.VarianceCheck && PopVariance < threshold){
			if(opt.printQ>-2){
				opt.printQ = 1;
				OPTprint("||||| Population Variance = " + to_string_with_precision(PopVariance,3) + " <(" + to_string_with_precision(threshold,3) + ") -> exit @ CMA-generation = " + to_string(opt.cma.generation) + "/" + to_string(opt.cma.generationMax) + " @ f=" + to_string_with_precision(opt.cma.bestfVec[opt.cma.bestp],16),opt);
			}
			opt.cma.exit = true;
		}
	}
	
}

void UpdateMeanCMA(OPTstruct &opt){
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			for(int d=0;d<opt.D;d++){
				opt.cma.meanOld[p][d] = opt.cma.mean[p][d];
				opt.cma.mean[p][d] = 0.;
				int cUpper = opt.cma.mu;
				if(opt.cma.MeanFromAllWeights[p] && opt.cma.WeightScenario==2) cUpper = opt.cma.populationSize;
				for(int c=0;c<cUpper;c++) opt.cma.mean[p][d] += opt.cma.cm[p]*opt.cma.weights[c]*opt.cma.pop[p][c][d];
			}
			if(!std::isfinite(Norm(opt.cma.mean[p]))){
				opt.cma.AbortQ[p] = 1;
				opt.cma.mean[p] = opt.cma.meanOld[p];
			}
		}
	}
}

void UpdateEvolutionPathsCMA(OPTstruct &opt){
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			double bs = sqrt(opt.cma.csigma*(2.-opt.cma.csigma)*opt.cma.mueff)/(opt.cma.cm[p]*opt.cma.stepSize[p]);
			for(int d=0;d<opt.D;d++){
				opt.cma.psigma[p][d] *= 1.-opt.cma.csigma;
				for(int j=0;j<opt.D;j++) opt.cma.psigma[p][d] += bs * opt.cma.CovInvSqrt[p](d,j) * (opt.cma.mean[p][j] - opt.cma.meanOld[p][j]);
				opt.cma.psigmaNorm[p] = Norm(opt.cma.psigma[p]);
			}
		}
	}
	
	UpdateStepSizeCMA(opt);//update step size here, according to Hansen2016
	
	opt.cma.hsigmafactor = sqrt(1.-EXP(2.*(double)(opt.cma.generation+1)*log(1.-opt.cma.csigma)));
	
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			double bc = sqrt(opt.cma.cc[p]*(2.-opt.cma.cc[p])*opt.cma.mueff)/(opt.cma.cm[p]*opt.cma.stepSize[p]);
			for(int d=0;d<opt.D;d++){
				opt.cma.pc[p][d] *= 1.-opt.cma.cc[p];
				if( opt.cma.psigmaNorm[p] < (1.4 + 2./((double)(opt.D+1)))*opt.cma.ExpectedValue*opt.cma.hsigmafactor ) opt.cma.pc[p][d] += bc * (opt.cma.mean[p][d] - opt.cma.meanOld[p][d]);
			}
		}
	}	
}

void UpdateStepSizeCMA(OPTstruct &opt){
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			double Candidate = opt.cma.stepSize[p] * EXP((opt.cma.psigmaNorm[p]/opt.cma.ExpectedValue-1.)*opt.cma.csigma/opt.cma.dsigma);
			if(std::isfinite(Candidate)) opt.cma.stepSize[p] = max(Candidate,1.0e-16);
			else if(opt.BreakBadRuns>0) opt.cma.AbortQ[p] = 1;
		}
	}
}

void UpdateCovarianceCMA(OPTstruct &opt){
	#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
	for(int p=0;p<opt.cma.runs;p++){
		if(opt.cma.AbortQ[p]==0){
			//update covariance matrix
			double hsigma = 0.;
			if( opt.cma.psigmaNorm[p] < (1.4 + 2./((double)(opt.D+1)))*opt.cma.ExpectedValue*opt.cma.hsigmafactor ) hsigma = 1.;
			double deltahsigma = (1.-hsigma)*opt.cma.cc[p]*(2.-opt.cma.cc[p]);
			double a = 1.+opt.cma.c1*deltahsigma-opt.cma.c1-opt.cma.cmu*opt.cma.WeightSum;
			vector<double> weightModifier(opt.cma.populationSize,1.);
			int cUpper = opt.cma.mu;
			if(opt.cma.WeightScenario==2){
				cUpper = opt.cma.populationSize;
				for(int c=opt.cma.mu;c<opt.cma.populationSize;c++){
					VectorXd y(opt.D);
					for(int d=0;d<opt.D;d++) y(d) = (opt.cma.pop[p][c][d]-opt.cma.mean[p][d]) / (opt.cma.cm[p]*opt.cma.stepSize[p]);
					y = opt.cma.CovInvSqrt[p] * y;
					weightModifier[c] = (double)opt.D/Norm2(VectorXdToVec(y));
				}
			}
			for(int i=0;i<opt.D;i++){
				for(int j=0;j<=i;j++){
					opt.cma.Covariance[p](i,j) *= a;
					opt.cma.Covariance[p](i,j) += opt.cma.c1*opt.cma.pc[p][i]*opt.cma.pc[p][j];
					for(int c=0;c<cUpper;c++){
						opt.cma.Covariance[p](i,j) += (opt.cma.pop[p][c][i]-opt.cma.mean[p][i]) * (opt.cma.pop[p][c][j]-opt.cma.mean[p][j]) * opt.cma.cmu*weightModifier[c]*opt.cma.weights[c]/POW(opt.cma.cm[p]*opt.cma.stepSize[p],2);
					}
				}
			}
			for(int i=0;i<opt.D;i++) for(int j=i+1;j<opt.D;j++) opt.cma.Covariance[p](i,j) = opt.cma.Covariance[p](j,i);
			
			//update inverse of sqrt of covariance matrix
			opt.cma.EVDcount[p]++;
			if(opt.cma.DelayEigenDecomposition) opt.cma.MaxDelay[p] = max(1,(int)(1./(10.*opt.D*(opt.cma.c1+opt.cma.cmu))));
			if(!opt.cma.DelayEigenDecomposition || opt.cma.EVDcount[p]>=opt.cma.MaxDelay[p]){
				bool success = true;
				MatrixXd Candidate = GetInvSqrt(opt.cma.Covariance[p], p, /*true*/false, success, opt);
				if(success){
					opt.cma.CovInvSqrt[p] = Candidate;
					opt.cma.EVDcount[p] = 0;//reset
				}
				else if(opt.BreakBadRuns>0 && opt.cma.AbortQ[p]==0) opt.cma.AbortQ[p] = 1;
			}
			for(int i=0;i<opt.D;i++){
				if(opt.cma.AbortQ[p]==0) for(int j=i+1;j<opt.D;j++){
					if(!std::isfinite(opt.cma.CovInvSqrt[p](i,j))){
						OPTprint("UpdateCovarianceCMA: Warning !!! CovInvSqrt not finite",opt);
						if(opt.BreakBadRuns>0) opt.cma.AbortQ[p] = 1;
					}
				}
			}
		}
	}
}

MatrixXd GetInvSqrt(MatrixXd &A, int p, bool validate, bool &success, OPTstruct &opt){
	
    SelfAdjointEigenSolver<MatrixXd> solver(A);
    if(solver.info() != Eigen::Success){
		OPTprint("GetInvSqrt: Warning !!! Eigenvalue decomposition failed",opt);
		success = false;
		return MatrixXd::Identity(A.rows(),A.rows());
	}
    MatrixXd InvSqrtEigenVals = solver.eigenvalues().asDiagonal();
	opt.cma.B[p] = solver.eigenvectors();//normalized eigenvectors in columns
	vector<double> diag(0);
	for(int i=0;i<A.rows();i++){
		diag.push_back(InvSqrtEigenVals(i,i));//EigenValues
		opt.cma.D[p](i,i) = sqrt(diag[i]);//sqrt of EigenValues
		InvSqrtEigenVals(i,i) = 1./opt.cma.D[p](i,i);//inverse of sqrt of EigenValues
	}
	if(ABS(*max_element(diag.begin(),diag.end()))>1.0e-16){
		double InverseConditionNumber = *min_element(diag.begin(),diag.end()) / *max_element(diag.begin(),diag.end());
		if(!(std::isfinite(InverseConditionNumber)) || InverseConditionNumber<1.0e-14){
			//OPTprint("GetInvSqrt: Warning !!! Covariance matrix is ill-conditioned: InverseConditionNumber = " + to_string_with_precision(InverseConditionNumber,3),opt);
			success = false;
			return MatrixXd::Identity(A.rows(),A.rows());		
		}
	}
	//MatrixXd InvSqrtofA = opt.cma.B[p] * InvSqrtEigenVals * opt.cma.B[p].inverse();//for general A
	MatrixXd InvSqrtofA = opt.cma.B[p] * InvSqrtEigenVals * opt.cma.B[p].transpose();//for positive semidefinite A

	if(validate){
		MatrixXd TestMat = A * InvSqrtofA * InvSqrtofA - MatrixXd::Identity(A.rows(),A.rows());
		for(int i=0;i<A.rows();i++) for(int j=0;j<A.rows();j++) if(!(std::isfinite(TestMat(i,j)))) cout << i << " " << j << " " << TestMat(i,j) << endl;
		cout << TestMat << endl;
		double TestVal = TestMat.squaredNorm();
		if(TestVal>1.0e-8){
			OPTprint("GetInvSqrt validation failed: TestVal = " + to_string_with_precision(TestVal,16) + " (not 0)",opt);
			success = false;
		}
	}	
	
	return InvSqrtofA;
}



void UpdateCMA(OPTstruct &opt){
	
	//UpdateStepSizeCMA(opt);//update step size here, according to Wikipedia
	
	//terminate populations
	if(opt.BreakBadRuns>1){
		#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
		for(int p=0;p<opt.cma.runs;p++){
			if(opt.cma.AbortQ[p]==0){
				int N = opt.cma.VarianceCheck, n = opt.cma.history[p].size();
				if(n<N) opt.cma.history[p].push_back(opt.cma.bestfVec[p]);
				else{
					double StdDevThreshold = sqrt((double)N)*opt.epsf;
					sort(opt.cma.history[p].begin(),opt.cma.history[p].end());
					if(opt.cma.bestfVec[p]<opt.cma.history[p][N-1]) opt.cma.history[p][N-1] = opt.cma.bestfVec[p];
					double mean = accumulate(opt.cma.history[p].begin(),opt.cma.history[p].end(),0.)/((double)N), StdDev = 0.;
					for(int h=0;h<N;h++) StdDev += (opt.cma.history[p][h]-mean)*(opt.cma.history[p][h]-mean);
					StdDev = sqrt(StdDev/((double)N));
					if(StdDev<StdDevThreshold && opt.homotopy<=1) opt.cma.AbortQ[p] = 1;
				}
			}
		}
	}
	
	//shrink populations
	if(opt.cma.PopulationDecayRate>0. && opt.varianceUpdatedQ){
		double c = log10(opt.CurrentVariance), t = log10(opt.VarianceThreshold);
		opt.cma.NewPopulationSize = max(4,min(opt.cma.InitialPopulationSize,(int)((double)opt.cma.InitialPopulationSize*(1.-opt.cma.PopulationDecayRate*c/t))));
		if(opt.cma.generation%opt.cma.VarianceCheck==0 && opt.cma.NewPopulationSize<(int)(0.9*(double)opt.cma.populationSize)){
			opt.cma.populationSize = opt.cma.NewPopulationSize;
			opt.cma.report[2] += "  >>>>> ShrinkPopulation @ generation = " + to_string(opt.cma.generation) + ": NewPopulationSize = " + to_string(opt.cma.populationSize);
			#pragma omp parallel for schedule(dynamic) if(opt.threads>1)
			for(int p=0;p<opt.cma.runs;p++){
				if(opt.cma.AbortQ[p]==0){
					opt.cma.pop[p].resize(opt.cma.populationSize);
					opt.cma.pop2[p].resize(opt.cma.populationSize);
					opt.cma.f[p].resize(opt.cma.populationSize);
					if((int)opt.PenaltyMethod[0]>0){
						opt.ALmu[p].resize(opt.cma.populationSize);
						opt.ALmu2[p].resize(opt.cma.populationSize);
						for(int i=0;i<opt.NumEC+opt.NumIC;i++){
							opt.ALlambda[i][p].resize(opt.cma.populationSize);
							opt.ALlambda2[i][p].resize(opt.cma.populationSize);
						}
					}
				}
			}
		}
	}
	
	//prepare next generation
	bool b = UpdateSearchSpace(opt);	
	opt.cma.generation++;
	for(int p=0;p<opt.cma.runs;p++) opt.cma.penaltyFactor[p] = 1.0e+3 * (double)opt.cma.generation/(double)opt.cma.generationMax * opt.cma.InitPenaltyFactor[p];
}

//************** end MIT CMA main code **************











//************** begin MIT GAO main code **************

void GAO(OPTstruct &opt){
	OPTprint("\n ***** Enter GAO() ... *****",opt);
	int D = opt.D; // Search space dimension
	
	startTimer("\n InitializeGAO",opt);
	InitializeGAO(opt);
	endTimer(" InitializeGAO",opt);
	
	while(!ManualOPTbreakQ(opt)){
		
		EvaluateGAO(opt);
		ReportGAO(opt);
		if(opt.gao.exit) break;
		
		SelectGAO(opt);
		CrossoverGAO(opt);
		MutateGAO(opt);
		ConstrainGAO(opt);
		
		UpdateGAO(opt);
	}
	
	OPTprint("\n ***** ... exit GAO() *****",opt);	
}

void SetDefaultGAOparams(OPTstruct &opt){//needs as input opt.evalMax, opt.gao.runs, and opt.gao.popExponent
	opt.ActiveOptimizer = 104;
	opt.gao.activeQ = true;
	opt.printQ = 1;
	opt.reportQ = 2;
	opt.FailCountThreshold = 10;
	
	SetDefaultSearchSpace(opt);

	if(opt.threads>1){
		if(opt.gao.runs<=opt.threads) opt.gao.runs = opt.threads;
		else if(opt.gao.runs%opt.threads>0) opt.gao.runs = max(opt.threads,opt.gao.runs-(opt.gao.runs%opt.threads)+opt.threads);
		if(opt.function+2014>=1 && opt.function+2014<=30){ opt.threads = 1; cout << "opt.threads -> 1 !!!!!!!!!!!!!!!!!!!!!" << endl; }
	}
	
	if(opt.gao.RandomSearch){
		opt.gao.runs = opt.threads;
		opt.gao.popExponent = 0.5;
		opt.epsf = -1.;
	}
	
	opt.gao.populationSize = (int)pow(opt.evalMax/(double)opt.gao.runs,opt.gao.popExponent);
	opt.gao.generationMax = (int)pow(opt.evalMax/(double)opt.gao.runs,1.-opt.gao.popExponent);
	opt.gao.InitialPopulationSize = opt.gao.populationSize;
	opt.gao.PopulationDecayRate = 1.7;
	opt.gao.VarianceCheck = min(opt.D,(int)(0.2*(double)opt.gao.generationMax));
	opt.gao.AvBinHeight = (double)opt.gao.runs;
	opt.gao.NumParents = 2;//choose two or three, 2 seems better
	
	//default hyperparameters; if Decay==1.0 => decay to zero if opt.gao.HyperParamsSchedule==1
	opt.gao.mutationRate = 0.02;//0.1	
	opt.gao.mutationRateDecay = 1.0;
	opt.gao.mutationStrength = 10.;
	opt.gao.mutationStrengthDecay = 1.0;
	opt.gao.invasionRate = 0.02;
	opt.gao.invasionRateDecay = 1.0;
	opt.gao.crossoverRate = 0.8;
	opt.gao.crossoverRateDecay = 0.;
	opt.gao.dispersalStrength = 1.5;
	opt.gao.dispersalStrengthDecay = -(double)opt.gao.runs;//0.;//
	
	//hyperparameter ranges
	opt.gao.MutationRateRange = {{0.},{0.2}};
	opt.gao.MutationRateDecayRange = {{-0.2},{1.}};	
	opt.gao.MutationStrengthRange = {{0.},{100.}};
	opt.gao.MutationStrengthDecayRange = {{-0.2},{1.}};
	opt.gao.InvasionRateRange = {{0.},{0.05}};
	opt.gao.InvasionRateDecayRange = {{-0.2},{1.}};
	opt.gao.CrossoverRateRange = {{0.2},{1.0}};
	opt.gao.CrossoverRateDecayRange = {{0.},{1.}};
	opt.gao.DispersalStrengthRange = {{0.},{(double)opt.gao.runs}};
	opt.gao.DispersalStrengthDecayRange = {{-(double)opt.gao.runs},{0.}};
	
	if(opt.gao.RandomSearch){
		opt.gao.mutationRate = 0.;
		opt.gao.mutationStrength = 0.;
		opt.gao.invasionRate = 1.23456789;
		opt.gao.crossoverRate = 0.;
		opt.gao.dispersalStrength = 0.;
	}
}

void InitializeGAO(OPTstruct &opt){
	
	opt.gao.generation = 1;
	
	if(opt.gao.HyperParamsSchedule==2) opt.PickRandomParamsQ = true;
	
	if(opt.gao.RandomSearch){
		opt.PickRandomParamsQ = false;
		opt.gao.HyperParamsDecayQ = false;
		opt.gao.ShrinkPopulationsQ = false;
		opt.gao.HyperParamsSchedule = 0;
	}

	opt.gao.report.resize(3,"");
	
	opt.gao.pop.resize(opt.gao.runs);
	opt.gao.pop2.resize(opt.gao.runs);
	opt.gao.f.resize(opt.gao.runs);
	opt.gao.bestfVec.resize(opt.gao.runs,1.0e+300);
	opt.gao.bestfVecOld.resize(opt.gao.runs,1.0e+300);
	opt.gao.nb_eval.resize(opt.gao.runs,0);
	opt.gao.bp.resize(opt.gao.runs);//breeding probabilities <=> crossover rates
	
	vector<double> hyperparams = {{opt.gao.mutationRate},{opt.gao.mutationRateDecay},{opt.gao.mutationStrength},{opt.gao.mutationStrengthDecay},{opt.gao.invasionRate},{opt.gao.invasionRateDecay},{opt.gao.crossoverRate},{opt.gao.crossoverRateDecay},{opt.gao.dispersalStrength},{opt.gao.dispersalStrengthDecay}};
	opt.gao.numhp = hyperparams.size();
	
	opt.gao.hpNames.resize(opt.gao.numhp);
	opt.gao.hpNames = {{"           MutationRate"},{"      MutationRateDecay"},{"       MutationStrength"},{"  MutationStrengthDecay"},{"           InvasionRate"},{"      InvasionRateDecay"},{"          CrossoverRate"},{"     CrossoverRateDecay"},{"      DispersalStrength"},{" DispersalStrengthDecay"}};

	opt.gao.hpRange.resize(opt.gao.numhp);
	opt.gao.hpRange = {opt.gao.MutationRateRange,opt.gao.MutationRateDecayRange,opt.gao.MutationStrengthRange,opt.gao.MutationStrengthDecayRange,opt.gao.InvasionRateRange,opt.gao.InvasionRateDecayRange,opt.gao.CrossoverRateRange,opt.gao.CrossoverRateDecayRange,opt.gao.DispersalStrengthRange,opt.gao.DispersalStrengthDecayRange};
	
	opt.gao.hp.resize(opt.gao.runs);
	opt.gao.hpInit.resize(opt.gao.runs);
	
	#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++){
		opt.gao.hp[p].resize(opt.gao.numhp);
		opt.gao.hpInit[p].resize(opt.gao.numhp);
		opt.gao.hp[p] = hyperparams;
		opt.gao.hpInit[p] = hyperparams;
		opt.gao.pop[p].resize(2*opt.gao.populationSize);
		opt.gao.pop2[p].resize(2*opt.gao.populationSize);
		opt.gao.f[p].resize(2*opt.gao.populationSize);
		opt.gao.bp[p].resize(opt.gao.populationSize);
		for(int c=0;c<2*opt.gao.populationSize;c++){
			opt.gao.pop[p][c].resize(opt.D);
			opt.gao.pop2[p][c].resize(opt.D);
			InitializeChromosome(p, c, opt);//create first parents and first children
			constrainGAO(p,c,opt);
			if(c<opt.gao.populationSize) opt.gao.f[p][c] = GetFuncVal(0, opt.gao.pop[p][c], opt.function, opt);//evaluate first parents
		}
	}
	if(opt.PickRandomParamsQ) for(int p=0;p<opt.gao.runs;p++) PickHyperParametersGAO(p,opt);
	
	double fmin = 1.0e+300, fmax = -fmin;
	for(int p=0;p<opt.gao.runs;p++){
		for(int c=0;c<opt.gao.populationSize;c++){
			if(opt.gao.f[p][c]<fmin) fmin = opt.gao.f[p][c];
			else if(opt.gao.f[p][c]>fmax) fmax = opt.gao.f[p][c];
		}
	}
	opt.gao.InitSpread = fmax-fmin;
	
	opt.gao.centralhp.resize(hyperparams.size(),0.);
	if(opt.PickRandomParamsQ){
		for(int k=0;k<hyperparams.size();k++){
			for(int p=0;p<opt.gao.runs;p++) opt.gao.centralhp[k] += opt.gao.hp[p][k];
			opt.gao.centralhp[k] /= (double)opt.gao.runs;
		}
	}
	else opt.gao.centralhp = hyperparams;
	opt.gao.Successfulhp.resize(0);	
	
	opt.nb_eval = 0.;

	OPTprint("---max. #evaluations            = " + to_string_with_precision(opt.evalMax,2),opt);
	OPTprint("---runs (#Populations)          = " + to_string(opt.gao.runs),opt);
	OPTprint("---popExponent                  = " + to_string(opt.gao.popExponent),opt);
	OPTprint("---populationSize (initial)     = " + to_string(opt.gao.populationSize),opt);
	OPTprint("---total #Chromosomes(initial)  = " + to_string(opt.gao.runs*2*opt.gao.populationSize),opt);
	OPTprint("---ShrinkPopulationsQ           = " + to_string(opt.gao.ShrinkPopulationsQ),opt);
	OPTprint("---PickRandomParamsQ            = " + to_string(opt.PickRandomParamsQ),opt);
	OPTprint("---HyperParamsDecayQ            = " + to_string(opt.gao.HyperParamsDecayQ),opt);
	OPTprint("---HyperParamsSchedule          = " + to_string(opt.gao.HyperParamsSchedule),opt);
	OPTprint("---VarianceCheck/generationMax  = " + to_string(opt.gao.VarianceCheck) + "/" + to_string(opt.gao.generationMax),opt);
	OPTprint("---(" + to_string(opt.gao.runs) + ") configurations for HyperParameters\n" + vec_to_str(opt.gao.hpNames),opt);
	for(int p=0;p<min(100,opt.gao.runs);p++) OPTprint(vec_to_str_with_precision(opt.gao.hp[p],2),opt);
	if(opt.gao.runs>100) OPTprint("...................",opt);
	OPTprint("GAO initialized\n",opt);
}

void InitializeChromosome(int p, int c, OPTstruct &opt){
	for(int g=0;g<opt.D;g++) opt.gao.pop[p][c][g] = alea(opt.SearchSpace[g][0],opt.SearchSpace[g][1],opt);
}

void PickHyperParametersGAO(int p, OPTstruct &opt){
	for(int k=0;k<opt.gao.numhp;k++) opt.gao.hp[p][k] = alea(opt.gao.hpRange[k][0],opt.gao.hpRange[k][1],opt);	
}

void EvaluateGAO(OPTstruct &opt){//evaluate only the children (the parents had been evaluated already), and store the (sorted) total populations in pop2
	#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++){
		for(int c=opt.gao.populationSize;c<2*opt.gao.populationSize;c++) opt.gao.f[p][c] = GetFuncVal(0, opt.gao.pop[p][c], opt.function, opt);
		//sort all parents and children
		int count = 0;
		for(auto c: sort_indices(opt.gao.f[p])){
			opt.gao.pop2[p][count] = opt.gao.pop[p][c];
			count++;
		}
		sort(opt.gao.f[p].begin(),opt.gao.f[p].end());//now, f[p] is in the order consistent with opt.gao.pop2[p]
		//collect best result from each population
		opt.gao.bestfVecOld[p] = opt.gao.bestfVec[p];
		opt.gao.bestfVec[p] = opt.gao.f[p][0];
	}
	opt.nb_eval += (double)(opt.gao.runs*opt.gao.populationSize);
}

void ReportGAO(OPTstruct &opt){
	//determine whether the report will be printed
    int gen = opt.gao.generation;
	double expon = floor(log10((double)gen));
	opt.printQ = -1;
    for(int i=1;i<10;i++){
      if(i*(int)pow(10.,expon)==gen || gen%1000==0 || gen==opt.gao.generationMax-1){
		  opt.printQ = 1;
		  break;
      }
    }
    if(opt.printQ==-1){ cout << "."; cout.flush(); }
    
	OPTprint("\n\n ***** begin GAO report *****",opt);
	OPTprint(" generation " + to_string(opt.gao.generation) + "/" + to_string(opt.gao.generationMax) + ": #FuncEvals = " + to_string_with_precision(opt.nb_eval,2),opt);
	
	double bestfOld = opt.currentBestf;
	//report details
	vector<double> bestfvec(opt.gao.bestfVec);
	int count = 0;
	for(auto p: sort_indices(opt.gao.bestfVec)){
		if(count==0){
			opt.gao.bestf = opt.gao.bestfVec[p];
			opt.gao.bestx =	opt.gao.pop2[p][0];
			opt.gao.bestp = p;
			if(opt.gao.bestf<opt.currentBestf){
				opt.NewbestFoundQ = true;
				opt.gao.report[0] = "  new bestf found. ";
				opt.currentBestf = opt.gao.bestf;
				opt.gao.Successfulhp.push_back(opt.gao.hp[p]);
				if(opt.gao.HyperParamsSchedule==3){
					opt.gao.report[0] += " successful (central) hyper parameters:\n";
					for(int k=0;k<opt.gao.numhp;k++) opt.gao.report[0] += opt.gao.hpNames[k] + " = " + to_string_with_precision(opt.gao.hp[p][k],2) + "(" + to_string_with_precision(opt.gao.centralhp[k],2) + ")\n";
				}
				else{
					opt.gao.report[0] += " successful hyper parameters:\n";
					for(int k=0;k<opt.gao.numhp/2;k++) opt.gao.report[0] += opt.gao.hpNames[2*k] + " = " + to_string_with_precision(opt.gao.hp[p][2*k],2) + " (" + to_string_with_precision(opt.gao.hp[p][2*k+1],2) + ")\n";
				}
			}
			else opt.NewbestFoundQ = false;
		}
		bestfvec[count] = opt.gao.bestfVec[p];
		count++;
	}
	if(opt.printQ==1){
		for(int r=0;r<opt.gao.report.size();r++){
			if(opt.gao.report[r]!=""){
				OPTprint(opt.gao.report[r],opt);
				opt.gao.report[r] = "";
			}
		}
	}
	
	opt.gao.exit = false;
	bool varianceConvergedQ = false;
	if(opt.epsf>0. && RelDiff(bestfOld,opt.currentBestf)>1.0e-12) varianceConvergedQ = VarianceConvergedQ(opt);
	if(opt.NewbestFoundQ || opt.printQ==1 || varianceConvergedQ){
		if(varianceConvergedQ){
			opt.printQ = 1;
			OPTprint("\n ##### Variance converged @ GAO-generation = " + to_string(opt.gao.generation) + "/" + to_string(opt.gao.generationMax),opt);
			opt.gao.exit = true;
		}
		if(opt.NewbestFoundQ) opt.NewbestFoundQ = false;//reset for next generation
	}
	if(opt.gao.generation==opt.gao.generationMax){
		opt.printQ = 1;
		OPTprint("\n ##### Warning: Exit @ GAO-generation = " + to_string(opt.gao.generation) + "/" + to_string(opt.gao.generationMax),opt);
		opt.gao.exit = true;
	}	
	
	sort(bestfvec.begin(),bestfvec.end());
	if(opt.gao.runs>6) OPTprint(" bestfVec(sorted) = " + partial_vec_to_str_with_precision(bestfvec,0,3,12) + " .......... " +  partial_vec_to_str_with_precision(bestfvec,bestfvec.size()-4,bestfvec.size()-1,12) + "\n",opt);
	else OPTprint(" bestfVec(sorted) = " + vec_to_str_with_precision(bestfvec,12) + "\n",opt);
	
	sort(opt.gao.bestfVecOld.begin(),opt.gao.bestfVecOld.end());
	if(opt.gao.bestf-opt.gao.bestfVecOld[0]>1.0e-12){
		OPTprint("ReportGAO: Warning !!! bestf has been lost! ",opt);
		SleepForever();
	}	
}

void SelectGAO(OPTstruct &opt){//select new parents from (sorted) pop2 and store in pop
	if(opt.gao.RandomSearch){
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++){
			for(int c=0;c<opt.gao.populationSize;c++){
				opt.gao.pop[p][c] = opt.gao.pop2[p][c];
			}
		}
	}
	else{
		//determine new spread
		double fmin = 1.0e+300, fmax = -fmin;
		for(int p=0;p<opt.gao.runs;p++){
			auto fInterval = minmax_element(begin(opt.gao.f[p]),end(opt.gao.f[p]));
			if(*fInterval.first<fmin) fmin = *fInterval.first;
			else if(*fInterval.second>fmax) fmax = *fInterval.second;
		}	
		if(!(std::isfinite(fmax))) fmax = 1.0e+300;
		opt.gao.Spread = fmax-fmin;
		opt.gao.InitSpread = opt.gao.Spread;
		if(opt.gao.Spread>opt.gao.InitSpread) opt.gao.InitSpread = opt.gao.Spread;
		opt.gao.TargetSpread = opt.gao.InitSpread*pow(1.-(double)opt.gao.generation/(double)opt.gao.generationMax,4.);
		double ShrinkSelection = 0.;//try to maintain full (current) f-spread
		if(opt.gao.Spread>0. && opt.gao.Spread>opt.gao.TargetSpread) ShrinkSelection = 1.-opt.gao.TargetSpread/opt.gao.Spread;//try to shrink f-spread

		//select new parents in each population
		vector<double> stretchRange{1.,2.}, Stretch(opt.gao.runs);
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++){
			vector<double> fpop2(opt.gao.f[p]);
			//int i=0;
			//while(i<fpop2.size() && fpop2[i]<fmin+opt.gao.TargetSpread) i++;
			//Stretch[p] = 2.*(double)(i-1)/((double)(2*opt.gao.populationSize-1));
			//KeepInRange(Stretch[p],stretchRange);
			//Stretch[p] = 1.8;
			for(int c=0;c<opt.gao.populationSize;c++){
				//int index = c;
				//int index = 2*c;
				int index = (int)((2.-ShrinkSelection)*(double)c);
				//int index = (int)(Stretch[p]*(double)c);//?
				opt.gao.pop[p][c] = opt.gao.pop2[p][index];
				opt.gao.f[p][c] = fpop2[index];
			}
		}
		int MinUpperIndex = (int)(*min_element(Stretch.begin(),Stretch.end())*(double)(opt.gao.populationSize-1));

		//set up histogram bins, fill histogram with the chromosome indices, and store the number of chromosomes in each bin
		int bins = (int)sqrt((double)(opt.gao.runs*opt.gao.populationSize));//(int)((double)(opt.gao.runs*opt.gao.populationSize)/opt.gao.AvBinHeight);
		vector<int> binRange{0,bins-1};
		opt.gao.histogram.clear();
		opt.gao.histogram.resize(bins,vector<vector<int>>(0));
		for(int b=0;b<bins;b++) opt.gao.histogram[b].resize(0);
		double BinWidth = opt.gao.Spread/(double)bins;
		//double BinWidth = opt.gao.TargetSpread/(double)bins;//Why is this not better?
		for(int p=0;p<opt.gao.runs;p++){
			for(int c=0;c<2*opt.gao.populationSize;c++){//inclusion of All f-values gives a more comprehensive / more accurate picture of their distribution
				vector<int> v{p,c};
				int B = (int)((opt.gao.f[p][c]-fmin)/BinWidth);
				KeepInRange(B,binRange);
				opt.gao.histogram[B].push_back(v);
			}
		}
		opt.gao.BinHeights.clear();
		opt.gao.BinHeights.resize(0);
		for(int b=0;b<bins;b++) opt.gao.BinHeights.push_back(opt.gao.histogram[b].size());
		int MaxBinHeight = *max_element(opt.gao.BinHeights.begin(),opt.gao.BinHeights.end());
	
		//assign breeding probability opt.gao.bp to each new parent, akin to the fitness uniform deletion scheme
		vector<double> TotalProbability(0); TotalProbability.resize(opt.gao.runs,0.);
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++){
			for(int c=0;c<opt.gao.populationSize;c++){
				int b = (int)((opt.gao.f[p][c]-fmin)/BinWidth);
				KeepInRange(b,binRange);
				double frequency = 1., minFrequency = 0.01;
				if(opt.gao.BinHeights[b]>0){ frequency = (double)(opt.gao.BinHeights[b]-1)/(double)MaxBinHeight;
					opt.gao.bp[p][c] = max(minFrequency,1.-sqrt(frequency));
					TotalProbability[p] += opt.gao.bp[p][c];
				}
			}
			TotalProbability[p] /= (double)opt.gao.populationSize;
		}
		double AvBreedingProbability = accumulate(TotalProbability.begin(),TotalProbability.end(),0.)/(double)opt.gao.runs;
		vector<double> avbp(opt.gao.runs,0.);
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++){
			double normalization = opt.gao.hp[p][6]/AvBreedingProbability;//
			for(int c=0;c<opt.gao.populationSize;c++){
				opt.gao.bp[p][c] *= normalization;
				if(opt.gao.bp[p][c]>1.) opt.gao.bp[p][c] = 1.;
				avbp[p] += opt.gao.bp[p][c];
			}
			avbp[p] /= (double)opt.gao.populationSize;
		}
	
		OPTprint(" SelectGAO:",opt);
		OPTprint("                  current [fmin,fmax] = [" + to_string(fmin) + "," + to_string(fmax) + "]",opt);
		OPTprint("                           InitSpread = " + to_string(opt.gao.InitSpread),opt);
		OPTprint("                Spread (TargetSpread) = " + to_string(opt.gao.Spread) + " (" + to_string(opt.gao.TargetSpread) + ")",opt);
		OPTprint("  average (min) upper selected index  = " + to_string((int)(VecAv(Stretch)*(double)(opt.gao.populationSize-1))) + " (" + to_string(MinUpperIndex) + ")" + " / " + to_string(2*opt.gao.populationSize-1),opt);
		OPTprint("         average breeding probability = " + to_string_with_precision(AvBreedingProbability,2) + " -> " + to_string_with_precision(VecAv(avbp),2),opt);
		OPTprint("                   NumBins (BinWidth) = " + to_string(bins) + " (" + to_string(BinWidth) + ")",opt);
		OPTprint("                         MaxBinHeight = " + to_string(MaxBinHeight),opt);	
		OPTprint("                           BinHeights = " + vec_to_str(opt.gao.BinHeights),opt);
	}
}

void CrossoverGAO(OPTstruct &opt){//produce all children ...
	if(!opt.gao.RandomSearch){
		int pm = opt.gao.runs-1, cm = opt.gao.populationSize-1;
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p1=0;p1<opt.gao.runs;p1++){// ... in all populations pop
			int pl = max(0,p1-(int)opt.gao.hp[p1][8]), pu = min(p1+(int)opt.gao.hp[p1][8],pm);
			for(int child=opt.gao.populationSize;child<2*opt.gao.populationSize;child++){
				bool success = false;
				while(!success){
					//choose 1st parent (from population p1)
					int c1 = alea_integer(0,cm,opt);
					if(alea(0.,1.,opt)<opt.gao.bp[p1][c1]){//proceed if c1 is allowed to breed
						//choose 2nd parent
						int p2 = alea_integer(pl,pu,opt);
						int c2 = alea_integer(0,cm,opt);
						if(alea(0.,1.,opt)<opt.gao.bp[p2][c2]){//proceed if c2 is allowed to breed
							if(opt.gao.NumParents==2){
								for(int g=0;g<opt.D;g++){//create child
									if(alea(0.,1.,opt)<0.5) opt.gao.pop[p1][child][g] = opt.gao.pop[p1][c1][g];
									else opt.gao.pop[p1][child][g] = opt.gao.pop[p2][c2][g];
								}
								success = true;
							}
							else{
								//choose 3rd parent
								int p3 = alea_integer(pl,pu,opt);
								int c3 = alea_integer(0,cm,opt);
								if(alea(0.,1.,opt)<opt.gao.bp[p3][c3]){//proceed if c3 is allowed to breed
									for(int g=0;g<opt.D;g++){//create child
										double rn = alea(0.,1.,opt);
										if(rn<0.333) opt.gao.pop[p1][child][g] = opt.gao.pop[p1][c1][g];
										else if(rn<0.666) opt.gao.pop[p1][child][g] = opt.gao.pop[p2][c2][g];
										else opt.gao.pop[p1][child][g] = opt.gao.pop[p3][c3][g];
									}
									success = true;
								}
							}
						}
					}
				}
			}
		}
	}
}

void MutateGAO(OPTstruct &opt){//mutate children in pop
	#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++){
		for(int c=opt.gao.populationSize;c<2*opt.gao.populationSize;c++){
			if(alea(0.,1.,opt)<opt.gao.invasionRate) InitializeChromosome(p, c, opt);//occasional random invasion
			else for(int g=0;g<opt.D;g++) if(alea(0.,1.,opt)<opt.gao.hp[p][0]) opt.gao.pop[p][c][g] += alea(-1.,1.,opt) * opt.gao.hp[p][2] * 0.5*(opt.SearchSpace[g][1]-opt.SearchSpace[g][0]) * sqrt(-2.*log(alea(1.0e-15,0.999999999999999,opt)));
		}
	}
}

void ConstrainGAO(OPTstruct &opt){//constrain children
	#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++){
		for(int c=opt.gao.populationSize;c<2*opt.gao.populationSize;c++) constrainGAO(p,c,opt);
	}	
}

void constrainGAO(int p, int c, OPTstruct &opt){	
	int L = opt.ex.L;
	if(opt.function==-440 || opt.function==-44 || opt.function==-20 || opt.function==-11 || opt.function==0 || (opt.function+2014>=1 && opt.function+2014<=30)){//keep in the box
		for(int d=0;d<opt.D;d++){
			if ( opt.gao.pop[p][c][d] < opt.SearchSpace[d][0] ){ opt.gao.pop[p][c][d] = opt.SearchSpace[d][0]; }
			if ( opt.gao.pop[p][c][d] > opt.SearchSpace[d][1] ){ opt.gao.pop[p][c][d] = opt.SearchSpace[d][1]; }
		}	
	}
	else if(opt.function==-1 || opt.function==-2){
		double Pi = 3.141592653589793, TwoPi = 2.*Pi;
		if(opt.function==-2){
			for(int d=0;d<opt.D;d++){//restrict to (possibly updated) search space
				if ( opt.gao.pop[p][c][d] < opt.SearchSpace[d][0] ){ opt.gao.pop[p][c][d] = opt.SearchSpace[d][0]; }
				if ( opt.gao.pop[p][c][d] > opt.SearchSpace[d][1] ){ opt.gao.pop[p][c][d] = opt.SearchSpace[d][1]; }		  
			}
		}
		//properize positions:
		vector<double> ang(L), occ(L);
		for(int d=0;d<L;d++) ang[d] = opt.gao.pop[p][c][d];
		occ = AnglesToOccNum(ang,opt.ex);//properized internally
		if(ABS(opt.ex.Abundances[0]-accumulate(occ.begin(),occ.end(),0.))>10*opt.ex.Abundances[0]){ cout << "constrainGAO: NearestProperOccNum (ex.settings[5]==" << opt.ex.settings[5] << ") Warning !!! " << accumulate(occ.begin(),occ.end(),0.) << " != " << opt.ex.Abundances[0] << endl; usleep(1000000); }
		ang = OccNumToAngles(occ);//revert to angles
		for(int d=0;d<L;d++) opt.gao.pop[p][c][d] = ang[d];//will automatically be within [0,Pi]
		if(opt.function==-2){	  
			for(int d=L;d<opt.D;d++){//keep Phases within [0,TwoPi]  
				while(opt.gao.pop[p][c][d]<0.){ opt.gao.pop[p][c][d] += TwoPi; }
				while(opt.gao.pop[p][c][d]>TwoPi){ opt.gao.pop[p][c][d] -= TwoPi; }
			}
		}
	}
}

void UpdateGAO(OPTstruct &opt){
	//ToDo: Adaptive/scheduled population size simply with resize()
	if(opt.gao.HyperParamsSchedule==0){
		//default hyperparameters
	}
	else if(opt.gao.HyperParamsSchedule==1){//constant hyperparameters
		//#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++) for(int k=0;k<opt.gao.numhp/2;k++) opt.gao.hp[p][2*k+1] = 0.;
	}
	else if(opt.gao.HyperParamsSchedule==2){//inter-population broadcasting of successful hyperparameters
		if(opt.NewbestFoundQ){
			int P = opt.gao.bestp;
			//#pragma omp parallel for schedule(static) if(opt.threads>1)
			for(int p=0;p<opt.gao.runs;p++){
				for(int k=0;k<opt.gao.numhp;k++) opt.gao.hp[p][k] += alea(0.,1.,opt)*(opt.gao.hp[P][k]-opt.gao.hp[p][k]);
			}
		}
	}	
	else if(opt.gao.HyperParamsSchedule==3){//self-adapting hyperparameters, stochastically drawn from itinerant Gaussians
		//double admixture = 0., width = 0.;
		double admixture = 0.2, width = 2.*0.05*0.05;
		//double admixture = 0.2, width = 2.*0.005*0.005;
		int activeHistory = min(10,(int)opt.gao.Successfulhp.size());
		for(int k=0;k<opt.gao.numhp;k++){//draw from Gaussian centered at (updated) central values, etc.
			double SuccessfulHistoryAv = 0.;
			for(int a=0;a<activeHistory;a++) SuccessfulHistoryAv += opt.gao.Successfulhp[opt.gao.Successfulhp.size()-1-a][k]/(double)activeHistory;
			opt.gao.centralhp[k] = (1.-admixture) * opt.gao.centralhp[k] + admixture*SuccessfulHistoryAv;
		}
		//#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++) for(int k=0;k<opt.gao.numhp;k++) opt.gao.hp[p][k] = opt.gao.centralhp[k] + randSign(opt)*sqrt(-width*log(alea(1.0e-16,1.,opt)));		
	}
	
	int dispersalOld = (int)opt.gao.hp[0][8];
	if(opt.gao.HyperParamsDecayQ){//decay (or increase) according to DecayRates
		//#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int p=0;p<opt.gao.runs;p++){
			for(int k=0;k<opt.gao.numhp/2-2;k++) opt.gao.hp[p][2*k] *= 1.-opt.gao.hp[p][2*k+1]/(opt.gao.generationMax-opt.gao.generation);
			opt.gao.hp[p][8] = 1.5;//max(1.5,0.01*(double)opt.gao.runs);
			//if(alea(0.,1.,opt)<0.01) opt.gao.hp[p][8] = floor((double)opt.gao.runs*sqrt((double)opt.gao.generation/(double)opt.gao.generationMax));
			//else opt.gao.hp[p][8] = 0.;
 			//for(int k=0;k<opt.gao.numhp/2;k++) opt.gao.hp[p][2*k] = opt.gao.hpInit[p][2*k]*(1.-opt.gao.hp[p][2*k+1]*(double)opt.gao.generation/(double)opt.gao.generationMax);
			
// 			if(opt.gao.Spread<opt.gao.TargetSpread){
// 				opt.gao.hp[p][0] *= 1.1;
// 				opt.gao.hp[p][8] -= 1.;
// 			}
// 			else{
// 				opt.gao.hp[p][0] /= 1.1;
// 				opt.gao.hp[p][8] += 1.;
// 			}
		}
	}
	
	//#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++) for(int k=0;k<opt.gao.numhp;k++) KeepInRange(opt.gao.hp[p][k],opt.gao.hpRange[k]);
	
	if((int)opt.gao.hp[0][8]!=dispersalOld) opt.gao.report[1] = "  << >> UpdateGAO: dispersal " + to_string(dispersalOld) + " -> " + to_string((int)opt.gao.hp[0][8]);
	
	if(opt.gao.ShrinkPopulationsQ) ShrinkPopulationsGAO(opt);
	
	bool b = UpdateSearchSpace(opt);
	
	opt.gao.generation++;
}

void ShrinkPopulationsGAO(OPTstruct &opt){
  double c = log10(opt.CurrentVariance), t = log10(opt.VarianceThreshold);
  int NewPopulationSize = max(4,min(opt.gao.InitialPopulationSize,(int)((double)opt.gao.InitialPopulationSize*(1.-opt.gao.PopulationDecayRate*c/t))));
  if(opt.gao.generation%opt.gao.VarianceCheck==0 && NewPopulationSize<(int)(0.9*(double)opt.gao.populationSize)){
	opt.gao.populationSize = NewPopulationSize;
    opt.gao.report[2] = "  >>>>> ShrinkPopulation @ generation = " + to_string(opt.gao.generation) + ": NewPopulationSize = " + to_string(opt.gao.populationSize);
	#pragma omp parallel for schedule(static) if(opt.threads>1)
	for(int p=0;p<opt.gao.runs;p++){
		opt.gao.pop[p].resize(2*opt.gao.populationSize);
		opt.gao.pop2[p].resize(2*opt.gao.populationSize);
		opt.gao.f[p].resize(2*opt.gao.populationSize);
		opt.gao.bp[p].resize(opt.gao.populationSize);
	}
  }
}

//************** end MIT GAO main code **************




//************** begin PSO main code **************

void PSO(OPTstruct &opt){
	
	//----------------------------------------------------- INITIALISATION
	int breakQ = 0;
	SetupOPTloopBreakFile(opt);
	LoadOPTparams(opt);
	int eval_max = opt.pso.eval_max_init; // Max number of evaluations for each run
	double TwoPi = 2.*3.141592653589793;
	int D = opt.D; // Search space dimension	
	opt.pso.CurrentSwarmSize = opt.pso.InitialSwarmSize;
	if(opt.pso.CurrentSwarmSize>opt.pso.Smax) opt.pso.CurrentSwarmSize = opt.pso.Smax;
	int S = opt.pso.CurrentSwarmSize; // Swarm size
	int K = opt.pso.MaxLinks; // Max number of particles informed by a given one
	int n_exec = 0; // current number of executions
	opt.nb_eval = 0.;

	opt.pso.bestx.resize(opt.D); fill(opt.pso.bestx.begin(),opt.pso.bestx.end(),0.);
	opt.pso.bestHistory.clear(); opt.pso.loopCountHistory.clear();
	opt.pso.bestparams.clear(); opt.pso.bestparams.resize(3); for(int i=0;i<3;i++) opt.pso.bestparams[i].resize(4);
	opt.pso.PSOparamsHistory.clear();
	opt.pso.ab.clear(); opt.pso.ab.resize(7);
	opt.pso.ab[0] = opt.pso.alpha;//rate alpha
	opt.pso.ab[1] = 0.05; opt.pso.ab[2] = 0.95; opt.pso.ab[3] = 1.01; opt.pso.ab[4] = 1.99; opt.pso.ab[5] = 1.01; opt.pso.ab[6] = 1.99;//intervals for W,C1,C2
	opt.pso.MinimumRandomInitializations = (int)(0.1*(double)opt.pso.runs);

	while( n_exec < opt.pso.runs && !manualOPTbreakQ(opt)){
		n_exec++;
		opt.localrun = n_exec-1;
		bool BreakLoopQ = false;
		opt.pso.nb_eval = 0;
		opt.loopCount = 0;
  
		InitializePSOrun(opt);

		//---------------------------------------------- ITERATIONS
		while(!BreakLoopQ && opt.pso.nb_eval < eval_max){
			
			// ... set PSO hyperparameters
			if(opt.pso.SwarmDecayRate>0. && opt.varianceUpdatedQ) ShrinkSwarm(opt);
			S = opt.pso.CurrentSwarmSize;
			if(opt.pso.CoefficientDistribution==1 || opt.pso.CoefficientDistribution==4 || opt.pso.CoefficientDistribution==5) UpdatePSOparams(opt);
			else if(opt.pso.CoefficientDistribution==2){
				for(int s=0;s<S;s++){
					opt.pso.C1[s][0] = max(1.05,3.-(double)opt.loopCount/200.);
					opt.pso.C2[s][0] = min(3.,1.05+(double)opt.loopCount/200.);
					double sum = opt.pso.C1[s][0]+opt.pso.C2[s][0]; if(sum>4.){ opt.pso.C1[s][0] *= 4./sum; opt.pso.C2[s][0] *= 4./sum; }
				}
			}
			else if(opt.pso.CoefficientDistribution==3){//adaptive fuzzy determination of PSO parameters
				FuzzyPSOparams(opt);
			}
  
			// ... update PCA-RBF-routine parameters, and report
			UpdatePCARBF(opt);
  
			// ... determine reporting needs
			opt.loopCount++;
			bool ReportQ = false;
			if(opt.reportQ>0){
				double expon = floor(log10((double)opt.loopCount));
				for(int i=1;i<10;i++){ if(i*(int)pow(10.,expon)==opt.loopCount){ opt.reportQ = 2; ReportQ = true; break; } }
			}  

			// ... re-seed a percentage (opt.pso.reseed) of the (historically) worst swarm particles
			if(opt.pso.reseed>0. && opt.loopCount>opt.pso.reseedEvery && (opt.loopCount%opt.pso.reseedEvery==0 || ReportQ)){	
				int NumReseed = max(1,(int)(opt.pso.reseed*(double)S)), count = 0;
				vector<double> flist(S), fl; for(int s=0;s<S;s++) flist[s] = opt.pso.P[s].f;//collect personal bests of swarm...
				vector<int> reseeded; 
				for(auto s: sort_indices_reverse(flist)){//...sort them (worst first)...
					if(count<NumReseed){//...save those to be reseeded...
						reseeded.push_back(s);				
						fl.push_back(opt.pso.P[s].f);
					}
					count++;
				}
				double reseedEverySurivors = 0., fgkj = 0.;
				for(int s=0;s<S;s++){
					for(int i=0;i<reseeded.size();i++){
						if(s!=i){
							reseedEverySurivors += opt.pso.R[s];
							fgkj += 1.;
						}
						reseedEverySurivors /= fgkj;
					}
				}
				#pragma omp parallel for schedule(static) if(opt.threads>1)
				for(int i=0;i<reseeded.size();i++){
					InitializeSwarmParticle(reseeded[i],opt);
					if(opt.pso.CoefficientDistribution==1 || opt.pso.CoefficientDistribution==4 || opt.pso.CoefficientDistribution==5) ReinitializePSOparams(reseeded[i], opt);
				}
				opt.pso.reseedEvery = max(1,(int)VecAv(opt.pso.R));
				sort(reseeded.begin(),reseeded.end());
				if(opt.printQ && opt.reportQ>1){
					OPTprint("\n ***** Re-seeding of particles " + vec_to_str(reseeded) + "\n" + vec_to_str(fl),opt);
					OPTprint(" --> reseed every " + to_string(opt.pso.reseedEvery) + "-th loop --- ",opt);
				}
			}
  
			// ... replace worst particle by newly constructed average particle, a sort of mild elitism
			int avPart = opt.pso.worst;
			if(opt.pso.avPartQ){
				opt.pso.X[avPart].f = 0.;
				#pragma omp parallel for schedule(static) if(opt.threads>1)
				for (int d = 0; d < D; d++ ){
					opt.pso.X[avPart].x[d] = 0.;
					opt.pso.V[avPart].v[d] = 0.;
					for (int s = 0; s < S; s++ ){
						if(s!=avPart){
							opt.pso.X[avPart].x[d] += opt.pso.X[s].x[d];
							opt.pso.V[avPart].v[d] += opt.pso.V[s].v[d];
							opt.pso.X[avPart].f += opt.pso.X[s].f;
						}
					}
					opt.pso.X[avPart].x[d] /= (double)(S-1);
					opt.pso.V[avPart].v[d] /= (double)(S-1);
					opt.pso.X[avPart].f /= (double)(S-1);
				}
			}
  
			// ... determine mandatory informants
			#pragma omp parallel for schedule(static) if(opt.threads>1)
			for(int s=0;s<S;s++){
				if(opt.pso.init_links==1) for(int m=0;m<S;m++) opt.pso.LINKS[m][s] = 0; // Init to "no link"
				opt.pso.LINKS[s][s] = 1; // Each particle informs itself
				if(alea(0.,1.,opt)<opt.pso.elitism) opt.pso.LINKS[opt.pso.best][s] = 1;//MIT: Best particle informs all others if opt.pso.elitism==1.
			}
			int KK = max(1,K - (int)(opt.pso.elitism+0.5) - opt.pso.avPartQ);
			// ... and other links (who informs whom, at random)
			if(opt.pso.init_links==1){
				#pragma omp parallel for schedule(static) if(opt.threads>1)
				for(int Informer=0;Informer<S;Informer++){
					for(int i=0;i<KK;i++){//KK: maximum number of links per informer
						int ToBeInformed = Informer;
						while(ToBeInformed==Informer) ToBeInformed = alea_integer( 0, S - 1, opt );
						if( ToBeInformed!=opt.pso.best || opt.pso.elitism<alea(0.,1.,opt) ){//MIT: best particle is only informed by itself if opt.pso.elitism==1.
							if(alea(0.,1.,opt)<opt.pso.I[Informer]) opt.pso.LINKS[Informer][ToBeInformed] = 1;
						}
					}
				}
			}

			// ... move the swarm
			#pragma omp parallel for schedule(static) if(opt.threads>1)
			for(int s=0;s<S;s++){// For each particle ...
				int g=s;// .. find the best admissible informant
				for(int m=0;m<S;m++) if(opt.pso.LINKS[m][s]==1 && opt.pso.P[m].f<opt.pso.P[g].f) g = m;
				bool AcceptMove = false;
				while(!AcceptMove){//accumulate moves until acceptance
					//...compute the new velocity, and move
					MakeMove(s,opt.pso.best,g,opt);
					//...interval confinements:   
					ConstrainPSO(s,opt);
					//...accept/reject move
					AcceptMove = AcceptMoveQ(omp_get_thread_num(),s,opt.pso.X[s].x,opt.pso.V[s].v,opt.pso.P[s].f,opt);
				}
				//...evaluate the new position
				opt.pso.X[s].f = GetFuncVal( s, opt.pso.X[s].x, opt.function, opt );
			}
			opt.nb_eval += (double)S;
			opt.pso.nb_eval += S;       
    
			// ... adapt PCA-RBF-routine parameters, and report
			AdaptPCARBF(opt);  
    
			// ... check for new (both local and global) bestf
			bool NewbestFoundQ = false, localrunNewbestFoundQ = false;
			for(int s=0;s<S;s++){
				if(opt.pso.X[s].f<opt.pso.P[opt.pso.best].f){
					localrunNewbestFoundQ = true;
					if(opt.pso.X[s].f<opt.pso.bestf){
						NewbestFoundQ = true;
						opt.pso.bestparams = {{opt.pso.W[s],opt.pso.C1[s],opt.pso.C2[s]}};
						opt.pso.bestf = opt.pso.X[s].f;
						opt.pso.bestx = opt.pso.X[s].x;
						if(opt.ExportIntermediateResults){
							ofstream IntermediateOPTresults;
							IntermediateOPTresults.open("mpDPFT_IntermediateOPTresults.dat");      
							IntermediateOPTresults << to_string_with_precision(opt.pso.bestf,16) << endl;
							IntermediateOPTresults << endl << vec_to_str_with_precision(opt.pso.bestx,16) << endl;
							IntermediateOPTresults << endl << vec_to_CommaSeparatedString_with_precision(opt.pso.bestx,16) << endl;
							IntermediateOPTresults.close(); 
						}
					}
				}
			}
    
			// ... update the best previous position    
			for (int s = 0; s < S; s++ ) if(opt.pso.X[s].f<opt.pso.P[s].f) opt.pso.P[s] = opt.pso.X[s];
   
			// ... get the current best ... and current worst
			int currentBest = 0;
			opt.currentBestf = opt.pso.X[0].f;
			double currentWorstf = opt.currentBestf;
			opt.currentBestx = opt.pso.X[0].x;
			for(int s=1;s<S;s++){
				if(opt.pso.X[s].f<opt.currentBestf){ currentBest = s; opt.currentBestf = opt.pso.X[s].f; opt.currentBestx = opt.pso.X[s].x; }
				else if(opt.pso.X[s].f>currentWorstf){ currentWorstf = opt.pso.X[s].f; opt.pso.worst = s; }
			}
    
			// ... post-process current best
			if(opt.pso.PostProcessQ>0/*&& (opt.pso.VarianceCheck==0 || opt.Historyf.size()==opt.pso.VarianceCheck) && opt.CurrentVariance<max(1.0e-4*VecAv(opt.Historyf),1.0e+3*opt.VarianceThreshold)*/){
				//if(opt.currentBestf<opt.Historyf[opt.pso.VarianceCheck-1]){
				double bf = opt.currentBestf;
				vector<double> bx = opt.currentBestx;
				if(PostProcess(opt)){//update opt.currentBestf and opt.currentBestx
					opt.pso.X[currentBest].f = opt.currentBestf;
					opt.pso.X[currentBest].x = opt.currentBestx;
					if(currentBest!=opt.pso.best || opt.pso.X[currentBest].f<opt.pso.P[opt.pso.best].f) opt.pso.P[currentBest] = opt.pso.X[currentBest];
				}
				else{
					opt.currentBestf = bf;
					opt.currentBestx = bx;
				}
			}
    
			// ... update the best index
			for(int s=0;s<S;s++) if(opt.pso.P[s].f<=opt.pso.P[opt.pso.best].f) opt.pso.best = s;
			
			// ... determine if PSO hyperparameters need to be updated in the next iteration
			opt.pso.UpdateParamsQ = false;
			if(/*NewbestFoundQ*/localrunNewbestFoundQ){
				int h = opt.pso.History.size()-1;
				opt.pso.History[h] = opt.pso.P[opt.pso.best].f;
				opt.pso.HistoryX[h] = opt.pso.P[opt.pso.best].x;
				if(opt.pso.CoefficientDistribution==5){
					double threshold = opt.pso.OldBestf-max(opt.pso.AbsAcc,opt.pso.spread*EXP(-0.3*(double)opt.pso.UnsuccessfulAttempts));
					if(opt.pso.P[opt.pso.best].f<threshold){
						opt.pso.UnsuccessfulAttempts = min(0,--opt.pso.UnsuccessfulAttempts);
						opt.pso.SuccessfulW.push_back(opt.pso.W[opt.pso.best][0]);
						opt.pso.SuccessfulC1.push_back(opt.pso.C1[opt.pso.best][0]);
						opt.pso.SuccessfulC2.push_back(opt.pso.C2[opt.pso.best][0]);	
						opt.pso.spread = opt.pso.OldBestf - opt.pso.P[opt.pso.best].f;
						opt.pso.OldBestf = opt.pso.P[opt.pso.best].f;
						//cout << "opt.pso.spread (@loopCount==" << opt.loopCount << ") = " << opt.pso.spread << " with threshold = " << threshold << endl;
						opt.pso.UpdateParamsQ = true;
					}
					else opt.pso.UnsuccessfulAttempts++;
					//cout << "...#UnsuccessfulAttempts = " << opt.pso.UnsuccessfulAttempts << endl;
				}
			}
    
			// ... intermediate report
			bool varianceConvergedQ = VarianceConvergedQ(opt);
			opt.reportQ = opt.reportQOriginal;
			if(opt.printQ && ReportQ) OPTprint("\n PSO-run " + to_string(n_exec) + "/" + to_string(opt.pso.runs) + " loop " + to_string(opt.loopCount) + " [SwarmSize=" + to_string(S) + "] --- total func evals " + to_string(opt.nb_eval) + " --- bestf = " + to_string_with_precision(opt.pso.P[opt.pso.best].f,16) + " --- current worstf = " + to_string_with_precision(opt.pso.X[opt.pso.worst].f,16) + " --- current bestf = " + to_string_with_precision(opt.currentBestf,16) +" @ current bestx = \n" + vec_to_str_with_precision(opt.currentBestx,8) + "\n",opt);  
    
			// ... check if finished; if no improvement, information links will be reinitialized
			//opt.pso.error=opt.currentBestf;
			opt.pso.error=opt.pso.P[opt.pso.best].f;
			if(opt.pso.error>=opt.pso.error_prev) opt.pso.init_links = 1;
			else opt.pso.init_links = 0;
			opt.pso.error_prev = opt.pso.error;

			// ... update search space
			bool UpdatedQ = false;
			if(opt.loopCount>opt.pso.VarianceCheck){
				if(opt.pso.AlwaysUpdateSearchSpace || NewbestFoundQ || localrunNewbestFoundQ) UpdatedQ = UpdateSearchSpace(opt);
			}
			
			// ... check termination criteria
			if(varianceConvergedQ/* && !UpdatedQ*/){
				BreakLoopQ = true;
				opt.reportQ = opt.reportQOriginal;
				OPTprint("Variance converged. Exit @ PSO-loop = " + to_string(opt.loopCount) + "/" + to_string(opt.pso.loopMax),opt);
				//opt.Historyf.resize(0);     
			}
			else if(opt.loopCount==opt.pso.loopMax){
				BreakLoopQ = true;
			}
			if(!BreakLoopQ) BreakLoopQ = manualOPTbreakQ(opt);  
			if(!BreakLoopQ && opt.BreakBadRuns>0) BreakLoopQ = breakLoopQ(opt.pso.P[opt.pso.best].f,opt);
    
		}//individual run finished here

		OPTprint("Final Search Space:",opt);
		for (int d=0;d<D;d++) OPTprint("param " + to_string(d) + " -> [" + to_string_with_precision(opt.SearchSpace[d][0],8) + "," + to_string_with_precision(opt.SearchSpace[d][1],8) + "]",opt);
    
		if( ConvergedQ(n_exec,opt) || breakQ==2 ) n_exec = opt.pso.runs;//no further runs
		else{//for the next run
			opt.CurrentVariance = 1.0e+100;
			opt.varianceUpdatedQ = false;
			opt.Historyf.clear();
			opt.Historyf.resize(0);
			opt.pso.PSOparamsHistory.push_back({{opt.pso.W[opt.pso.best],opt.pso.C1[opt.pso.best],opt.pso.C2[opt.pso.best]}});
			opt.pso.SortedHistory.clear();
			opt.pso.SortedPSOparamsHistory.clear();
			for(auto h: sort_indices(opt.pso.History)){
				opt.pso.SortedHistory.push_back(opt.pso.History[h]);
				opt.pso.SortedPSOparamsHistory.push_back(opt.pso.PSOparamsHistory[h]);
			}
			if(opt.pso.CoefficientDistribution>0){
				OPTprint("Best PSO-parameters:",opt);
				for(int h=0;h<min(10,(int)opt.pso.SortedPSOparamsHistory.size());h++){
					OPTprint(to_string(h) + ": bestf = " + to_string_with_precision(opt.pso.SortedHistory[h],16),opt);
					OPTprint("{ W , W_a , W_b , W_alpha }     = " + vec_to_CommaSeparatedString_with_precision(opt.pso.SortedPSOparamsHistory[h][0],6),opt);
					OPTprint("{ C1 , C1_a , C1_b , C1_alpha } = " + vec_to_CommaSeparatedString_with_precision(opt.pso.SortedPSOparamsHistory[h][1],6),opt);
					OPTprint("{ C2 , C2_a , C2_b , C2_alpha } = " + vec_to_CommaSeparatedString_with_precision(opt.pso.SortedPSOparamsHistory[h][2],6),opt);
				}
			}
			opt.LoopCounts.push_back((double)opt.loopCount);
			opt.pso.reseedEvery = max(1,(int)(2.*opt.pso.reseedFactor*VecAv(opt.LoopCounts)));
			//if(S==max(opt.threads,10)) opt.pso.SwarmDecayRate *= 0.95; else opt.pso.SwarmDecayRate *= 1.05;//try to reach minimum swarm size, eventually.... NONSENSE
			//OPTprint("New SwarmDecayRate = " + to_string_with_precision(opt.pso.SwarmDecayRate,3),opt);
		}
	}//all runs finished here

	//report best PSO hyperparameters
	if(opt.pso.CoefficientDistribution>0){
		OPTprint("Best PSO-parameters { W , W_a , W_b , W_alpha }     = " + vec_to_CommaSeparatedString_with_precision(opt.pso.bestparams[0],16),opt);
		OPTprint("Best PSO-parameters { C1 , C1_a , C1_b , C1_alpha } = " + vec_to_CommaSeparatedString_with_precision(opt.pso.bestparams[1],16),opt);
		OPTprint("Best PSO-parameters { C2 , C2_a , C2_b , C2_alpha } = " + vec_to_CommaSeparatedString_with_precision(opt.pso.bestparams[2],16),opt);
	}

}

void MakeMove(int s, int best, int g, OPTstruct &opt){
	if(opt.pci.enabled) opt.pci.moveCount[s]++;
	if(opt.ActiveOptimizer==101){
		for(int d=0;d<opt.D;d++){
			opt.pso.V[s].v[d] = opt.pso.W[s][0]*opt.pso.V[s].v[d] + alea(0,opt.pso.C1[s][0],opt)*(opt.pso.P[s].x[d]-opt.pso.X[s].x[d])+alea(0,opt.pso.C2[s][0],opt)*(opt.pso.P[g].x[d]-opt.pso.X[s].x[d]);
			//if(ABS(opt.pso.V[s].v[d])>opt.pso.Vmax[s]*opt.searchSpaceExtent[d]) opt.pso.V[s].v[d] = Sign(opt.pso.V[s].v[d])*opt.pso.Vmax[s]*opt.searchSpaceExtent[d];//not good, apparently...
			if(opt.pci.active && s!=best && opt.pci.moveCount[s]>1){
				if(alea(0.,1.,opt)<1./((double)opt.D)){//make big jump
					opt.pso.V[s].v[d] += alea(0,opt.pci.MovePrefactor,opt)*opt.searchSpaceExtent[d];
					opt.pci.JumpAt[s] = opt.pci.moveCount[s];
				}
				else{//make small (but incrementally larger) move
					opt.pso.V[s].v[d] += alea(0,opt.pci.MovePrefactor,opt)*opt.searchSpaceExtent[d]*(double)(opt.pci.moveCount[s]-1)/((double)opt.pci.MaxMoveCount);
				}
			}
			opt.pso.X[s].x[d] = opt.pso.X[s].x[d] + opt.pso.V[s].v[d];
		}
	}
}

void PSOKeepInBox(int s, OPTstruct &opt){
	for(int d=0;d<opt.D;d++){
		if ( opt.pso.X[s].x[d] < opt.SearchSpace[d][0] ){ opt.pso.X[s].x[d] = opt.SearchSpace[d][0]; opt.pso.V[s].v[d] = 0.; }
		if ( opt.pso.X[s].x[d] > opt.SearchSpace[d][1] ){ opt.pso.X[s].x[d] = opt.SearchSpace[d][1]; opt.pso.V[s].v[d] = 0.; }
	}
}

void ConstrainPSO(int s, OPTstruct &opt){
    if(opt.function==-1 || opt.function==-2){
		int L = opt.ex.L;
		double Pi = 3.141592653589793, TwoPi = 2.*Pi;
		if(opt.function==-2) PSOKeepInBox(s,opt);
		//properize positions and make velocities consistent with properized positions:
		//current position xp_new=x_old+vp -> proper target position x_new=Properized(xp_new)=xp_new+(x_new-xp_new)=!=x_old+v -> v=x_new-x_old=x_new-(xp_new-vp)=vp+x_new-xp_new
		vector<double> ang(L), occ(L);
		for(int d=0;d<L;d++) ang[d] = opt.pso.X[s].x[d];
		occ = AnglesToOccNum(ang,opt.ex);
		if(ABS(opt.ex.Abundances[0]-accumulate(occ.begin(),occ.end(),0.))>10*opt.ex.Abundances[0]){ cout << "PSO: NearestProperOccNum (ex.settings[5]==" << opt.ex.settings[5] << ") Warning !!! " << accumulate(occ.begin(),occ.end(),0.) << " != " << opt.ex.Abundances[0] << endl; usleep(1000000); }
		ang = OccNumToAngles(occ);
		for(int d=0;d<L;d++){
			opt.pso.V[s].v[d] += ang[d] - opt.pso.X[s].x[d];
			opt.pso.X[s].x[d] = ang[d];//will automatically be within [0,Pi]
		}
		if(opt.function==-2){	  
			for(int d=L;d<opt.D;d++){//keep Phases within [0,TwoPi]  
				while(opt.pso.X[s].x[d]<0.){ opt.pso.X[s].x[d] += TwoPi; opt.pso.V[s].v[d] += TwoPi; }
				while(opt.pso.X[s].x[d]>TwoPi){ opt.pso.X[s].x[d] -= TwoPi; opt.pso.V[s].v[d] -= TwoPi; }
			}
		}
    }
    else if(opt.function==100){//mutually unbiased bases
		opt.pso.V[s].v = VecDiff(opt.pso.V[s].v,opt.pso.X[s].x);//v=vp-xp_new
		int NB = (int)(opt.AuxParams[1]+0.5), NV = (int)(opt.AuxParams[2]+0.5), dim = NV;
		for(int b=0;b<NB;b++){
			for(int w=0;w<NV;w++){
				double invnormw = 0.;
				for(int d=0;d<dim;d++){
					double wR = 0., wI = 0.;
					if(b>0){
						wR = opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2];
						if(d>0) wI = opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2+1];
						else{//fix global phase of each vector by putting imaginary part of first component to zero
							opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2+1] = 0.;
						}
					}
					else{//fix first basis to canonical basis
						opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2+1] = 0.;
						if(d==w){
							opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2] = 1.;
							wR = 1.;
						}
						else opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2] = 0.;
					}
					invnormw += wR*wR+wI*wI;
				}
				invnormw = 1./sqrt(invnormw);
				for(int d=0;d<dim;d++){
					opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2] *= invnormw;
					opt.pso.X[s].x[b*NV*dim*2+w*dim*2+d*2+1] *= invnormw;
				}
			}
		}//x_new=Normalized(xp_new)	
		opt.pso.V[s].v = VecSum(opt.pso.V[s].v,opt.pso.X[s].x);//v=vp-xp_new+x_new
	}
	else if(opt.function==10 || opt.function==200){
		//do nothing
	}
	else PSOKeepInBox(s,opt);
}  

void InitializeSwarmParticle(int s, OPTstruct &opt){  
	if(opt.function==-1 || opt.function==-2 || opt.function==-11){
		vector<double> initAngles = OccNumToAngles(InitializeOccNum(0,opt.ex));
		for (int d = 0; d < opt.D; d++ ){
			if( ((opt.function==-1 || opt.function==-11)) || d<opt.ex.L ) opt.pso.X[s].x[d] = initAngles[d];
			else if(opt.function==-2 && d>=opt.ex.L) opt.pso.X[s].x[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
			else opt.pso.X[s].x[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
			opt.pso.V[s].v[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt ) - 0.5*opt.pso.X[s].x[d];
		}
	}
    else{
		for (int d = 0; d < opt.D; d++ ){
			opt.pso.X[s].x[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
			opt.pso.V[s].v[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt ) - 0.5*opt.pso.X[s].x[d];
		}
    }
    ConstrainPSO(s, opt);
}

void FuzzyPSOparams(OPTstruct &opt){
	int S = opt.pso.CurrentSwarmSize;
	if(opt.loopCount>0){
		double fdelta, deltamax = sqrt((double)opt.D);
		if(opt.function==-44) fdelta = 1.;
		#pragma omp parallel for schedule(static) if(opt.threads>1)
		for(int s=0;s<S;s++){
			vector<double> normalizedDelta, normalizedDeltaBest;
			for(int d=0;d<opt.D;d++){
				normalizedDelta.push_back((opt.pso.X[s].x[d]-opt.pso.Xold[s].x[d])/opt.searchSpaceExtent[d]);
				normalizedDeltaBest.push_back((opt.pso.X[s].x[d]-opt.pso.P[opt.pso.best].x[d])/opt.searchSpaceExtent[d]);
			}
			opt.pso.delta[s] = Norm(normalizedDeltaBest);
			opt.pso.phi[s] = Norm(normalizedDelta)/deltamax * (opt.pso.X[s].f-opt.pso.Xold[s].f)/fdelta;
			//OUTPUT:
			double Low,Medium,High,outputNormalization,tmp;
			MemberShipStruct ms;
			GetMemberShip(opt.pso.delta[s],opt.pso.phi[s],deltamax,ms);
			//Interia
			opt.pso.W[s][0] = 0.;
			Low = 0.4; Medium = opt.pso.w; High = 0.9; outputNormalization = 0.;//default 0.3 0.5 1.0
			//1 if (φ is Worse or δ is Same) then (Inertia is Low)
			if(ms.Worse>0. || ms.Same>0.){ tmp = ms.Worse+ms.Same; opt.pso.W[s][0] += tmp*Low; outputNormalization += tmp; }
			//2 if (φ is Unvaried or δ is Near) then (Inertia is Medium)
			if(ms.Unvaried>0. || ms.Near>0.){ tmp = ms.Unvaried+ms.Near; opt.pso.W[s][0] += tmp*Medium; outputNormalization += tmp; }
			//3 if (φ is Better or δ is Far) then (Inertia is High)
			if(ms.Better>0. || ms.Far>0.){ tmp = ms.Better+ms.Far; opt.pso.W[s][0] += tmp*High; outputNormalization += tmp; }
			if(outputNormalization>0.) opt.pso.W[s][0] /= outputNormalization; else opt.pso.W[s][0] = opt.pso.w;
			//Social
			opt.pso.C2[s][0] = 0.;
			Low = 1.01; Medium = opt.pso.c; High = 1.9; outputNormalization = 0.;//default 1.0 2.0 3.0
			//4 if (φ is Better or δ is Near) then (Social is Low)
			if(ms.Better>0. || ms.Near>0.){ tmp = ms.Better+ms.Near; opt.pso.C2[s][0] += tmp*Low; outputNormalization += tmp; }
			//5 if (φ is Unvaried or δ is Same) then (Social is Medium)
			if(ms.Unvaried>0. || ms.Same>0.){ tmp = ms.Better+ms.Far; opt.pso.C2[s][0] += tmp*Medium; outputNormalization += tmp; }
			//6 if (φ is Worse or δ is Far) then (Social is High)
			if(ms.Worse>0. || ms.Far>0.){ tmp = ms.Better+ms.Far; opt.pso.C2[s][0] += tmp*High; outputNormalization += tmp; }
			if(outputNormalization>0.) opt.pso.C2[s][0] /= outputNormalization; else opt.pso.C2[s][0] = opt.pso.c;
			//Cognitive
			opt.pso.C1[s][0] = 0.;
			Low = 1.01; Medium = opt.pso.c; High = 1.9; outputNormalization = 0.;//default 0.1 1.5 3.0
			//7 if (δ is Far) then (Cognitive is Low)
			if(ms.Far>0.){ tmp = ms.Far; opt.pso.C1[s][0] += tmp*Low; outputNormalization += tmp; }
			//8 if (φ is Worse or φ is Unvaried or δ is Same or δ is Near) then (Cognitive is Medium)
			if(ms.Worse>0. || ms.Unvaried>0. || ms.Same>0. || ms.Near>0.){ tmp = ms.Worse+ms.Unvaried+ms.Same+ms.Near; opt.pso.C1[s][0] += tmp*Medium; outputNormalization += tmp; }
			//9 if (φ is Better) then (Cognitive is High)
			if(ms.Better>0.){ tmp = ms.Better; opt.pso.C1[s][0] += tmp*High; outputNormalization += tmp; }
			if(outputNormalization>0.) opt.pso.C1[s][0] /= outputNormalization; else opt.pso.C1[s][0] = opt.pso.c;
			//L
			opt.pso.Vmin[s] = 0.;
			Low = 0.; Medium = 1.0e-8; High = 1.0e-2; outputNormalization = 0.;//default 0. 0.001 0.01
			//10 if (φ is Unvaried or φ is Better or δ is Far) then (L is Low)
			if(ms.Unvaried>0. || ms.Better>0. || ms.Far>0.){ tmp = ms.Unvaried+ms.Better+ms.Far; opt.pso.Vmin[s] += tmp*Low; outputNormalization += tmp; }
			//11 if (δ is Same or δ is Near) then (L is Medium)
			if(ms.Same>0. || ms.Near>0.){ tmp = ms.Same+ms.Near; opt.pso.Vmin[s] += tmp*Medium; outputNormalization += tmp; }
			//12 if (φ is Worse) then (L is High)
			if(ms.Worse>0.){ tmp = ms.Worse; opt.pso.Vmin[s] += tmp*High; outputNormalization += tmp; }
			if(outputNormalization>0.) opt.pso.Vmin[s] /= outputNormalization; else opt.pso.Vmin[s] = opt.pso.vmin;	
			//U
			opt.pso.Vmax[s] = 0.;
			Low = 0.01; Medium = 0.1; High = 1.0; outputNormalization = 0.;//default 0.1 0.15 0.2
			//13 if (δ is Same) then (U is Low)
			if(ms.Same>0.){ tmp = ms.Same; opt.pso.Vmax[s] += tmp*Low; outputNormalization += tmp; }
			//14 if (φ is Unvaried or φ is Better or δ is Near) then (U is Medium)
			if(ms.Unvaried>0. || ms.Better>0. || ms.Near>0.){ tmp = ms.Unvaried+ms.Better+ms.Near; opt.pso.Vmax[s] += tmp*Medium; outputNormalization += tmp; }
			//15 if (φ is Worse or δ is Far) then (U is High)
			if(ms.Worse>0. || ms.Far>0.){ tmp = ms.Worse+ms.Far; opt.pso.Vmax[s] += tmp*High; outputNormalization += tmp; }
			if(outputNormalization>0.) opt.pso.Vmax[s] /= outputNormalization; else opt.pso.Vmax[s] = opt.pso.vmax;
		}
		OPTprint("\n ***** Adaptive PSO Parameter Determination",opt);
	}
	for(int s=0;s<S;s++){ opt.pso.Xold[s].x = opt.pso.X[s].x; opt.pso.Xold[s].f = opt.pso.X[s].f; }
}
	
//************** end PSO main code **************



bool breakLoopQ(double test, OPTstruct &opt){
  	if(opt.localrun==0) opt.checkAtLoop = (int)(0.05*(double)opt.loopCount);
  	else if(opt.loopCount>0 && opt.loopCount%opt.checkAtLoop==0){
    	if(opt.localrun==1) opt.BestFuncAtLoop.push_back(test);
    	else{
      		int Loop = opt.loopCount/opt.checkAtLoop;
      		if(opt.BestFuncAtLoop.size()<Loop) opt.BestFuncAtLoop.push_back(test);
      		else{
				if(test<opt.BestFuncAtLoop[Loop-1]){
	  				opt.BestFuncAtLoop[Loop-1] = test;
	  				OPTprint("breakLoopQ: new best = " + to_string(test) + " @loop=" + to_string(Loop),opt);
				}
				else{
	  				double diff = test-opt.BestFuncAtLoop[Loop-1], compare = opt.BestFuncAtLoop[Loop-2]-opt.BestFuncAtLoop[Loop-1];
	  				double ProbCriterion = 0.25*min(1.,1.-ASYM(compare,diff));
	  				if(alea(0.,1.,opt)<ProbCriterion){
	    				OPTprint("breakLoopQ: break@ Loop = " + to_string(Loop) + " (test=" + to_string(test) + " vs " + to_string(opt.BestFuncAtLoop[Loop-1]) + ") with probability = " + to_string(ProbCriterion),opt);
	    				return true;
	  				}
				}
      		}
    	}
  	}
  	return false;
}

  //===========================================================
  double alea( double a, double b, OPTstruct &opt ){// random number (uniform distribution) in [a,b]
    return a + opt.RNpos(opt.MTGEN) * ( b - a );
  }
  //===========================================================
  int alea_integer( int a, int b, OPTstruct &opt ){// Integer random number in [a,b]
    int ir = (int)((double)a+alea(0.,1.,opt)*(double)(b-a+1));
    if((double)ir>b) return b;
    else return ir;
  }

void GetECIC(vector<double> &EC, vector<double> &IC, vector<double> &x, OPTstruct &opt){
	if(opt.function==10){
		vector<double> X(x2X_ChemicalProcess(x));

		//equality constraints   (EC[..]==0)
		EC[0] = X[3]-0.9*(X[4]+X[5]+X[6]);

		//inequality constraints (IC[..]>=0)
		IC[0] = 4.*X[1]-X[4];
		IC[1] = 5.*X[2]-X[5];
		IC[2] = 2.*X[0]-X[3];
	}
}

double GetPenalty(int s, vector<double> &x, OPTstruct &opt){
	int p,c;
	if(opt.ActiveOptimizer==101){
		c = 0;
		p = s;
	}
	else if(opt.ActiveOptimizer==106){
		c = s%opt.cma.populationSize;
		p = (s-c)/opt.cma.populationSize;
	}

	vector<double> EC(opt.NumEC);
	vector<double> IC(opt.NumIC);
	GetECIC(EC,IC,x,opt);

	double Penalty = 0.;
	if((int)opt.PenaltyMethod[0]==0){//fixed quadratic penalty method
		// double spread = 1.;//doesn't seem to be useful this way...
		// if(opt.ActiveOptimizer==101 && opt.loopCount>1) spread = opt.pso.X[opt.pso.worst].f-opt.currentBestf;
		// else if(opt.ActiveOptimizer==106 && opt.cma.generation>1) spread = opt.cma.spread[p];
		// if(opt.varianceUpdatedQ && spread>0. && opt.CurrentVariance>0.) PenaltyScale = PenaltyScale+spread*sqrt(opt.VarianceThreshold)/opt.CurrentVariance;
		for(int i=0;i<opt.NumEC;i++){
			double violation = ABS(EC[i]);
			if(violation > opt.maxEC[p][i]) opt.maxEC[p][i] = violation;
			if(!std::isfinite(EC[i])) Penalty += 1.;
			else if(violation>0.){
				double penalty = violation/opt.maxEC[p][i];
				Penalty += penalty*penalty;
			}
		}
		for(int i=0;i<opt.NumIC;i++){
			double violation = ABS(min(0.,IC[i]));
			if(violation > opt.maxIC[p][i]) opt.maxIC[p][i] = violation;
			if(!std::isfinite(IC[i])) Penalty += 1.;
			else if(violation>0.){
				double penalty = violation/opt.maxIC[p][i];
				Penalty += penalty*penalty;
			}
		}
		Penalty *= opt.PenaltyMethod[1];
	}
	else if((int)opt.PenaltyMethod[0]==1){// Augmented Lagrangian Method (Vanilla)
		for(int i=0;i<opt.NumEC+opt.NumIC;i++){
			double violation = 0.;
			if(i<opt.NumEC) violation = ABS(EC[i]);
			else violation = ABS(min(0.,IC[i-opt.NumEC]));
			if(violation > 0.){
				Penalty += opt.ALlambda[i][p][c]*violation+0.5*opt.ALmu[p][c]*violation*violation;
				opt.ALlambda[i][p][c] += opt.ALmu[p][c]*violation;
			}
		}
		if(Penalty>0.) opt.ALmu[p][c] *= 1.+opt.PenaltyMethod[3];
	}
	else if((int)opt.PenaltyMethod[0]==2){//Augmented Lagrangian Method (MIT)
		for(int i=0;i<opt.NumEC+opt.NumIC;i++){
			double violation = 0., penalty = 0.;
			if(i<opt.NumEC){
				violation = ABS(EC[i]);
				if(violation > opt.maxEC[p][i]) opt.maxEC[p][i] = violation;
				penalty = violation/opt.maxEC[p][i];
			}
			else{
				violation = ABS(min(0.,IC[i-opt.NumEC]));
				if(violation > opt.maxIC[p][i-opt.NumEC]) opt.maxIC[p][i-opt.NumEC] = violation;
				penalty = violation/opt.maxIC[p][i-opt.NumEC];
			}
			if(!std::isfinite(penalty)) penalty = 1.;
			if(penalty > 0.){
				Penalty += opt.ALlambda[i][p][c]*penalty+opt.ALmu[p][c]*penalty*penalty;
				opt.ALlambda[i][p][c] += opt.ALmu[p][c]*penalty;
			}
		}
		if(Penalty>0.) opt.ALmu[p][c] *= 1.+opt.PenaltyMethod[3];
	}
	else if((int)opt.PenaltyMethod[0]==3 && opt.ALmu[p][c]>0.){//Unconstrained Augmented Lagrangian Method (Nocedal2006, pp. 523-524)
		for(int i=0;i<opt.NumEC+opt.NumIC;i++){
			double violation = 0., psi;
			if(i<opt.NumEC) violation = EC[i];
			else violation = IC[i-opt.NumEC];
			if(violation < opt.ALlambda[i][p][c] / opt.ALmu[p][c]){
				psi = -opt.ALlambda[i][p][c]*violation+0.5*opt.ALmu[p][c]*violation*violation;
				opt.ALlambda[i][p][c] -= opt.ALmu[p][c]*violation;
			}
			else{
				psi = -0.5 * opt.ALlambda[i][p][c]*opt.ALlambda[i][p][c] / opt.ALmu[p][c];
				opt.ALlambda[i][p][c] = 0.;
			}
			Penalty += psi;
		}
		Penalty *= opt.PenaltyMethod[1];
	}


	return Penalty;
}


//===========================================================
double GetFuncVal( int s, vector<double> &X, int function, OPTstruct &opt ){//Evaluate the fitness value for the particle of rank s
	double f = 0.123456789;

	vector<double> x(X);
	if(opt.Rand) ShiftRot(x,opt);

	if(function>=-2014+1 && function<=-2014+30){//cec14 test functions, function=-2014+id(cec14), with id(cec14)=1...30
		vector<double> func(1);
		vector<double> var(x);
		cec14_test_func(&var[0], &func[0], opt.D, 1, function+2014, opt.TestMode);
		f = func[0];
	}
	else if(function==-440) f = EquilibriumAbundances_ForestGeo_PSO(x,s);
	else if(function==-100) f = ObjFunc(x,opt);//costum objective function
	else if(function==-44) f = FitFunction_ForestGeo_PSO(x,s);
	else if(function==-20) f = ProduceNNinstance(x);//neural network
	else if(function==-11) f = Eint_1pExDFT(x,opt.ex);
	else if(function==-2) f = Etot_1pExDFT_CombinedMin(x,opt.ex);
	else if(function==-1){
		vector<double> var(x);
		var = AnglesToOccNum(var,opt.ex);
		f = Etot_1pExDFT(var,opt.ex);
	}
	else if(function==0){ for(int d=0;d<opt.D;d++) f += (x[d]-1./((double)(d+1)))*(x[d]-1./((double)(d+1))); }//template, shifted sphere function
	else if(function==10) f = ChemicalProcess(x,opt,s);
	else if(function==100) f = MutuallyUnbiasedBases(x,opt);
	else if(function==101) f = ShiftedSphere(x,opt);
	else if(function==102) f = SineEnvelope(x,opt);
	else if(function==103) f = Rana(x,opt);
	else if(function==104) f = UnconstrainedRana(x,opt);
	else if(function==105) f = UnconstrainedEggholder(x,opt);
	else if(function==200) f = NYFunction(x,opt);
	else if(function==201) f = QuantumCircuitIA(x,opt);
	else if(function==300) f = ConstrainedRastrigin(x,opt,s);
	else if(function>=1000 && function<=1001) f = DFTe_QPot(x,opt);
	else if(function>=1002 && function<=1005) f = DynDFTe_TimeSeries(x,opt);
    
    if(!std::isfinite(f)){
		//OPTprint("GetFuncVal: Error !!!\n",opt);
		return 1.2345e+300;
	}

    if(opt.anneal>0. && opt.AnnealType==-1) f += Anneal(0.,s,opt);

	if(opt.homotopy>0) f += Anneal(f,s,opt);

    return f;
}

double ShiftedSphere(vector<double> &x, OPTstruct &opt){//shifted sphere (on: opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;)
	double res = 0.;
	for(int d=0;d<opt.D;d++) res += (x[d]-0.5)*(x[d]-0.5);
	return res;	
}

void ShiftRot(vector<double> &x, OPTstruct &opt){
	vector<double> y(VecSum(x,opt.RandShift));
	vector<double> z(y.size());
	for(int i=0;i<opt.D;i++){
		z[i] = 0.;
		for(int j=0;j<opt.D;j++){
			z[i] += opt.RandMat(i,j) * y[j];
		}
	}
	x = z;
}

void ShiftRotInv(vector<double> &z, OPTstruct &opt){
	vector<double> y(z.size());
	for(int i=0;i<opt.D;i++){
		y[i] = 0.;
		for(int j=0;j<opt.D;j++){
			y[i] += opt.RandMatInv(i,j) * z[j];
		}
	}
	vector<double> x(VecDiff(y,opt.RandShift));
	z = x;
}

double EqualityConstraintViolation(vector<double> &y, OPTstruct &opt){
	if(opt.function==300) return VecTotal(y)-opt.AuxVal;
	else return 0.;
}

bool EqualityConstraintBoundedVariables(vector<double> &y, OPTstruct &opt){
	double a = opt.AuxParams[1], b = opt.AuxParams[2], scale = opt.AuxParams[3], threshold = opt.AuxParams[4];

	int failsafe = 10000, count = 0;

	vector<double> xlow(y), xmid(y), xupp(y);

	for(int d=0;d<opt.D;d++) y[d] = Sigmoid(y[d],a,b,scale);//unconstrained y to constrained y

	bool nlowFound = false, nuppFound = false;
	while(ABS(EqualityConstraintViolation(y,opt))>threshold && (!nlowFound || !nuppFound) && count<failsafe){//find lower and upper bounds for y_i
		//cout << "EqualityConstraintBoundedVariables (find lower and upper bounds for y_i) " << VecTotal(y) << " ---- " << vec_to_str(x) << endl;
		count++;
		if(!nuppFound) while(EqualityConstraintViolation(y,opt)<0. && count<failsafe){
			count++;
			for(int d=0;d<opt.D;d++){
				xupp[d] += max(threshold,1.23456789*ABS(xupp[d]));
				y[d] = Sigmoid(xupp[d],a,b,scale);
			}
			nuppFound = true;
			//cout << "EqualityConstraintBoundedVariables (upp) " << VecTotal(y) << " -- " << vec_to_str(xupp) << endl; usleep(100000);
		}
		if(!nlowFound) while(EqualityConstraintViolation(y,opt)>0. && count<failsafe){
			count++;
			for(int d=0;d<opt.D;d++){
				xlow[d] -= max(threshold,1.23456789*ABS(xlow[d]));
				y[d] = Sigmoid(xlow[d],a,b,scale);
			}
			nlowFound = true;
			//cout << "EqualityConstraintBoundedVariables (low) " << VecTotal(y) << " -- " << vec_to_str(xlow) << endl; usleep(100000);
		}
	}
	while(ABS(EqualityConstraintViolation(y,opt))>threshold && count<failsafe){//enforce equality constraint sum_i n_i via bisection
		count++;
		for(int d=0;d<opt.D;d++){
			xmid[d] = 0.5*(xlow[d]+xupp[d]);
			y[d] = Sigmoid(xmid[d],a,b,scale);
		}
		if(VecTotal(y)<opt.AuxVal) xlow = xmid;
		else xupp = xmid;
		//cout << "EqualityConstraintBoundedVariables (mid) " << VecTotal(y) << " -- " << vec_to_str(xmid) << endl; usleep(100000);
	}
	if(count>=failsafe){
		OPTprint("EqualityConstraintBoundedVariables: Warning !!! equality constraints violated",opt);
		return false;
	}
	return true;
}

double ConstrainedRastrigin(vector<double> &x, OPTstruct &opt, int s){//Rastrigin constrained on polytope, unconstrained x
	double f = 0., Y = opt.AuxVal * (double)opt.inflate/(double)opt.D;

	vector<double> y(x);

	int p = -1, c = -1;
	if(opt.ActiveOptimizer==101){
		c = 0;
		p = s;
	}
	else if(opt.ActiveOptimizer==106){
		c = s%opt.cma.populationSize;
		p = (s-c)/opt.cma.populationSize;
	}
	if(!EqualityConstraintBoundedVariables(y,opt)){
		OPTprint("ConstrainedRastrigin: abort run #" + to_string(p),opt);
		if(opt.ActiveOptimizer==106) opt.cma.AbortQ[p] = 1;
	}

	vector<double> test(0);
	for(int d=0;d<opt.D/opt.inflate;d++){
		double var = -Y;
		for(int i=0;i<opt.inflate;i++) var += y[opt.inflate*d+i];//opt.inflate auxilliary variables y[] yield the actual variable var @ dimension d
		if(p==0) test.push_back(var);
		f += var*var + 10. - 10.*cos(opt.AuxParams[0]*3.141592653589793*var);
	}
	if(p==0){
		test.push_back(f);
		opt.AuxMat.push_back(test);
	}

	return f;
}

double ChemicalProcess(vector<double> &x, OPTstruct &opt, int s){//x unbounded
	vector<double> X(x2X_ChemicalProcess(x));//X bounded

	double A2 = EXP(X[4])-1.;
	double A3 = EXP(X[5]/1.2)-1.;

	double Penalty = 0.;
	if(s>=0) Penalty = GetPenalty(s,x,opt);

	return -(11.*X[3]-3.5*X[0]-X[1]-X[4]-1.5*X[2]-1.2*X[5]-7.*X[6]-1.8*A2-1.8*A3) + Penalty;
}

vector<double> x2X_ChemicalProcess(vector<double> &x){
	double Y1 = (double)c2i(x[0],0,1);//continuous unbounded variable to integer variable \elem {0,1}
	double Y2 = (double)c2i(x[1],0,1);//continuous unbounded variable to integer variable \elem {0,1}
	double Y3 = (double)c2i(x[2],0,1);//continuous unbounded variable to integer variable \elem {0,1}
	double C1 = c2b(x[3],0.,1.);//continuous unbounded variable to box-constrained variable \elem [0,1]
	double B2 = c2b(x[4],0.,1.2);//continuous unbounded variable to box-constrained variable \elem [0,1.2]
	double B3 = c2b(x[5],0.,1.2);//continuous unbounded variable to box-constrained variable \elem [0,1.2]
	double BP = c2b(x[6],0.,1.2);//continuous unbounded variable to box-constrained variable \elem [0,1.2]
	return {Y1,Y2,Y3,C1,B2,B3,BP};
}
  
double SineEnvelope(vector<double> &x, OPTstruct &opt){//Sine Envelope Sine Wave function, Vanaret2020
	//minimum(dim=5) = -5.9659811; minimum(dim=2) = -1.4914953; (on: opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.;)
	double res = 0.;
	for(int d=0;d<opt.D-1;d++){
		double y = x[d+1]*x[d+1]+x[d]*x[d];
		res += 0.5+POW(sin(100.*sqrt(y)-0.5)/(10.*y+1.),2);
	}
	return -res;
}

double Rana(vector<double> &x, OPTstruct &opt){//Rana's function, Vanaret2020
	//minimum(dim=5) = -2046.8320657 (on: opt.SearchSpaceMin = -512.; opt.SearchSpaceMax = 512.;)
	double res = 0.;
	for(int d=0;d<opt.D-1;d++){
		double zp = sqrt(abs(x[d+1]+x[d]+1.)), zm = sqrt(abs(x[d+1]-x[d]+1.));
		res += x[d]*cos(zp)*sin(zm)+(1.+x[d+1])*sin(zp)*cos(zm);
	}
	return res;
}

double UnconstrainedRana(vector<double> &x, OPTstruct &opt){//Rana's function, Vanaret2020
	//perform unconstrained search in x by not imposing constraints explicitly: opt.cma.Constraints = 0;
	//minimum(dim=5) = -2046.8320657 (on: opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;)
	double res = 0.;
	vector<double> y(opt.D);
	for(int d=0;d<opt.D;d++) y[d] = SigmoidX(x[d],-512.,512.);
	for(int d=0;d<opt.D-1;d++){
		double zp = sqrt(abs(y[d+1]+y[d]+1.)), zm = sqrt(abs(y[d+1]-y[d]+1.));
		res += y[d]*cos(zp)*sin(zm)+(1.+y[d+1])*sin(zp)*cos(zm);
	}
	return res;
}

double UnconstrainedEggholder(vector<double> &x, OPTstruct &opt){//Eggholder function, Vanaret2020,
	//perform unconstrained search in x by not imposing constraints explicitly: opt.cma.Constraints = 0; (on: opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;)
	//minimum(dim=2) = -959.6406627 @ (512, 404.231805)
	//minimum(dim=5) = -3719.7248363 @ (485.589834, 436.123707, 451.083199, 466.431218, 421.958519)
	//minimum(dim=10) = -8291.2400675 @ (480.852413, 431.374221, 444.908694, 457.547223, 471.962527, 427.497291, 442.091345, 455.119420, 469.429312, 424.940608)
	double res = 0.;
	vector<double> y(opt.D);
	for(int d=0;d<opt.D;d++) y[d] = SigmoidX(x[d],opt.lowerBound,opt.upperBound);
	int trueD = opt.D/opt.inflate;
	vector<double> z(trueD,0.);
	for(int d=0;d<trueD;d++) for(int i=0;i<opt.inflate;i++) z[d] += y[opt.inflate*d+i];//opt.inflate auxilliary variables y[] yield the actual variable z @ dimension d
	for(int d=0;d<trueD-1;d++){
		double zp = sqrt(abs(0.5*z[d]+(z[d+1]+47.))), zm = sqrt(abs(z[d]-(z[d+1]+47.)));
		res -= (z[d+1]+47.)*sin(zp)+z[d]*sin(zm);
	}
	return res;
}

double NYFunction(vector<double> &x, OPTstruct &opt){//new, since 20250205

	vector<double> a(3),b(3),s(3);

	double wa;
	if(opt.postprocess){
		//Monte Carlo Sampling
		a = SampleFromSphere(opt);
		b = SampleFromSphere(opt);
		//wa = opt.RNpos(opt.MTGEN);//common sense
		wa = 0.5*(1.+cos(3.141592653589793*opt.RNpos(opt.MTGEN)));//Wootter PRD 23, 357 (1981)
		//if(omp_get_thread_num()==0){ opt.AuxMat.push_back(a); opt.AuxMat.push_back(b); }//see PostProcessNYFunction()
	}
	else{
		double theta1 = x[0];
		double theta2 = x[1];
		double phi1 = x[2];
		double phi2 = x[3];
		double alpha = x[4];
		double c1 = cos(theta1);
		double c2 = cos(theta2);
		double s1 = sin(theta1);
		double s2 = sin(theta2);
		a[0] = s1*cos(phi1);
		a[1] = s1*sin(phi1);
		a[2] = c1;
		b[0] = s2*cos(phi2);
		b[1] = s2*sin(phi2);
		b[2] = c2;
		wa = cos(alpha)*cos(alpha);
	}

	if(opt.SubFunc==3){//values for simulated experiment
		wa = 0.63;
		// a = {1./3.,2./3.,2./3.}; b = {-5./sqrt(35.),1./sqrt(35.),-3./sqrt(35.)};//seems to be the wrong order for BGE_20250206
		a = {3./sqrt(35.),-1./sqrt(35.),-5./sqrt(35.)}; b = {-2./3.,-2./3.,1./3.};
	}

	double wb = 1.-wa;
	//if(wb>wa){ double tmp = wb; wb = wa; wa = tmp; }//ensure wa>=wb in each evaluation



	for(int n=0;n<3;n++) s[n] = wa*a[n]+wb*b[n];
	vector<vector<double>> C = vector<vector<double>>(3,vector<double>(3,0.));
	for(int m=0;m<3;m++){
		for(int n=0;n<3;n++){
			C[m][n] = wa*a[m]*a[n]+wb*b[m]*b[n];
		}
	}
	double t = 1./sqrt(3.);
	vector<vector<double>> T = vector<vector<double>>(4,vector<double>(3,t));
	T[1][1] *= -1.; T[1][2] *= -1.;
	T[2][0] *= -1.; T[2][2] *= -1.;
	T[3][0] *= -1.; T[3][1] *= -1.;

	vector<double> prob(0);
	for(int k=0;k<4;k++){
		double tmp1 = 0., tmp2 = 0.;
		for(int n=0;n<3;n++){
			tmp1 += T[k][n]*s[n];
			for(int m=0;m<3;m++){
				tmp2 += T[k][m]*C[m][n]*T[k][n];
			}
		}
		prob.push_back((1.+2.*tmp1+tmp2)/16.);
	}
	for(int j=0;j<4;j++){
		for(int k=j+1;k<4;k++){
			double tmp1 = 0., tmp2 = 0.;
			for(int n=0;n<3;n++){
				tmp1 += (T[j][n]+T[k][n])*s[n];
				for(int m=0;m<3;m++){
					tmp2 += T[j][m]*C[m][n]*T[k][n];
				}
			}
			prob.push_back((1.+tmp1+tmp2)/8.);
		}
	}

	double f = 0.;
	for(int p=0;p<prob.size();p++){
		if(prob[p]<=1.0+1.0e-15){
			if(prob[p]>0.){
				if(opt.SubFunc==0 || opt.SubFunc==3) f -= opt.AuxParams[p] * log(prob[p]);//Loglikelihood (maximum || experiment)
				else if(opt.SubFunc==1) f += POW(opt.AuxParams[p] - prob[p],2);//LeastSquares
				else if(opt.SubFunc==2) f -= sqrt(opt.AuxParams[p] * prob[p]);//Fidelity
			}
		}
		else if(prob[p]>1.0+1.0e-15 || prob[p]<-1.0e-15){
			if(prob[p]<0.){
				#pragma omp critical
				{
					opt.report += "NYFunction: Error !!! prob[" + to_string(p) + "] = " + to_string(prob[p]) + "\n";
				}
			}
			return 1.0e+300;
		}
		//if(IntermediateCalc) cout << "-opt.AuxParams[" << p << "] * log(prob[" << p << "]) = " << -opt.AuxParams[p] * log(prob[p]) << endl;
	}

	//if(IntermediateCalc){ cout << "f = " << f << endl; usleep(10000000); }

	if(opt.finalcalc){
		opt.AuxVec.clear();
		opt.AuxVec.insert(opt.AuxVec.end(), prob.begin(), prob.end());
		for(int k=0;k<4;k++) opt.AuxVec.push_back(ScalarProduct(T[k],a));
		for(int k=0;k<4;k++) opt.AuxVec.push_back(ScalarProduct(T[k],b));
		vector<double> res(3,1.23456789);
		//res = PostProcessNYFunction(f,opt);
		opt.AuxVec.insert(opt.AuxVec.end(), res.begin(), res.end());
	}

	return f;
}

double QuantumCircuitIA(vector<double> &x, OPTstruct &opt){
	double res = 1.0e+300;

	// Execute Python script stored in .../mpScripts/...
    char exePath[PATH_MAX];// Get path to the current executable
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if (count == -1) {
        cerr << "QuantumCircuitIA: Could not determine executable path!" << endl;
        return res;
    }
    exePath[count] = '\0';  // Null-terminate
    string exeDir = dirname(exePath);
    // Construct path to Python script (relative to executable directory)
    //string scriptPath = exeDir + "/mpScripts/Project_ItaiArad_MIT/noisy-DM-PEPS-sim.py";
	//string scriptPath = exeDir + "/mpScripts/Project_ItaiArad_MIT/noisy_mps_vector_sim.py";
	string scriptPath = exeDir + "/mpScripts/Project_ItaiArad_MIT/noisy_mps_vector_sim-Martin.py";
    ostringstream cmd;
    cmd << "python3 " << scriptPath;

    for(double num : x) cmd << " " << num;
    try {// Execute the command, convert its output to double and return
            string output = exec(cmd.str().c_str());
			res = stod(output);
	} catch (const exception &e) {
        #pragma omp critical
        {
            cerr << "QuantumCircuitIA: Error !!! " << ": " << e.what() << " @ x(dim=" << x.size() << ") = " << vec_to_str_with_precision(x,6) << endl;
        }
    }
    return res;
}

vector<double> SampleFromSphere(OPTstruct &opt){
	vector<double> u(3);
	for(int i=0;i<3;i++) u[i] = opt.RNnormal(opt.MTGEN);
	Normalize(u);
	return u;
}

vector<double> PostProcessNYFunction(double f, OPTstruct &opt){
	//BEGIN USER INPUT
	double M = 2.0e+9;//2.0e+6;//
	bool ConsistentAverage = true;//false;//
	//END USER INPUT

	opt.finalcalc = false;
	vector<double> dummy(5);
	//cout << "PostProcessNYFunction: opt.SubFunc = " << opt.SubFunc << endl;
	if(opt.SubFunc==0 || opt.SubFunc==3){
		opt.postprocess = true;
		if(opt.SubFunc==3){
			string InputFileName = findMatchingFile("TabFunc_NYFunctionLikelihood_",".dat");
			vector<vector<double>> Dummy = vector<vector<double>>(ReadMat(InputFileName));
			double fopt = Dummy[opt.AuxIndex][1];//the f-value corresponding to the maximum likelihood
			return {0.,0.,EXP(opt.AuxVal*(fopt-NYFunction(dummy,opt)))};//lambda-value based on experiment parameters
		}
	}
	else{
		opt.SubFunc = 0;//switch from opt.SubFunc>0 to maximum-likelihood calculation
		string InputFileName = findMatchingFile("TabFunc_NYFunctionLikelihood_",".dat");
		vector<vector<double>> Dummy = vector<vector<double>>(ReadMat(InputFileName));
		double fopt = Dummy[opt.AuxIndex][1];//the f-value corresponding to the maximum likelihood
		//cout << "PostProcessNYFunction: " << opt.AuxVal << " " << fopt << endl;
		return {0.,0.,EXP(opt.AuxVal*(fopt-NYFunction(opt.currentBestx,opt)))};//lambda-value based on optimal least-squares measure, optimal fidelity measure, etc. to be compared with critical lambda lcr from maximum-likelihood optimization
	}

	if(M>2147483647.){
		ConsistentAverage = false;
		cout << "PostProcessNYFunction: Warning !!! insufficient memory... set ConsistentAverage -> false" << endl;
	}
	//if(omp_get_thread_num()==0) opt.AuxMat.clear(); opt.AuxMat.resize(0);
	double lcr = 0.,scr = 0., ccr = 0., mp = 2., m = 0.;

	vector<double> arg(0);
	if(ConsistentAverage) arg.resize((int)M);

	while(m<M){
		if(ConsistentAverage){
			arg[(int)m] = opt.AuxVal*(f-NYFunction(dummy,opt));
			lcr += EXP(arg[(int)m]);
		}
		else lcr += EXP(opt.AuxVal*(f-NYFunction(dummy,opt)));
		m += 1.;
		if((int)opt.AuxVal%1000==0 && abs(m-mp)<1.0e-3){ int tmp = opt.printQ; opt.printQ = 2; OPTprint("PostProcessNYFunction(N=" + to_string((int)opt.AuxVal) + "): lcr = " + to_string_with_precision(lcr/m,8) + " @ m = " + to_string_with_precision(m,8) + "\n",opt); opt.printQ = tmp; mp *= 2.; }
	}
	lcr /= M;
	//MatrixToFile(opt.AuxMat,"mpDPFT_AuxMat_M=" + to_string(M) + ".dat",16);

	double loglcr = LOG(lcr);
	m = 0.;
	while(m<M){
		double lambdaOverlcr;
		if(ConsistentAverage) lambdaOverlcr = EXP(arg[(int)m]-loglcr);
		else lambdaOverlcr = EXP(opt.AuxVal*(f-NYFunction(dummy,opt))-loglcr);
		if(lambdaOverlcr>1.){
			scr += 1.;
			ccr += lambdaOverlcr;
		}
		m += 1.;
	}
	scr /= M;
	ccr /= M;

	if(ccr>1.) cout << "PostProcessNYFunction: Warning !!! ccr = " << ccr << " > 1" << endl;

	return {scr,ccr,lcr};
}

// double NYFunction(vector<double> &x, OPTstruct &opt){//old, until 20250205
//
// 	vector<double> prob(9);
//
// 	double prefactor = 1./6.;
// 	double SQRT2 = sqrt(2.);
//     double TwoSQRT2 = 2.*SQRT2;
// 	double PI = 3.141592653589793;
// 	double TwoThirdPi = 2.*PI/3.;
//
// 	//Camilla's code with Pranjal's ordering of input x
// // 	double t0 = x[0], t1 = x[1], f0 = x[2], f1 = x[3], a = x[4];
// // 	double w0 = cos(a)*cos(a);//a is Pi-periodic
// // 	double w1 = 1.-w0;
// // 	double a0 = cos(t0);//t0 is effectively Pi-periodic, because of the combinations in which a0 and b0 appear
// // 	double b0 = sin(t0);
// // 	double a1 = cos(t1);//t1 is effectively Pi-periodic, because of the combinations in which a1 and b1 appear
// // 	double b1 = sin(t1);
// // 	double a02 = a0*a0;
// // 	double a04 = a02*a02;
// // 	double a12 = a1*a1;
// // 	double a14 = a12*a12;
// // 	double b02 = b0*b0;
// // 	double b04 = b02*b02;
// // 	double b12 = b1*b1;
// // 	double b14 = b12*b12;
// // 	double a0b0 = a0*b0;
// // 	double a1b1 = a1*b1;
// // 	prob[0] = prefactor*(w0*b02*(1. + a02 - TwoSQRT2*cos(f0)*a0b0) + w1*b12*(1. + a12 - TwoSQRT2*cos(f1)*a1b1));
// // 	prob[1] = prefactor*(w0*a02*(1. + b02 - TwoSQRT2*cos(f0)*a0b0) + w1*a12*(1. + b12 - TwoSQRT2*cos(f1)*a1b1));
// // 	prob[2] = prefactor*(w0*(a04 + b04 - 2.*cos(2.*f0)*a02*b02) + w1*(a14 + b14 - 2.*cos(2.*f1)*a12*b12));
// // 	prob[3] = prefactor*(w0*b02*(1. + a02 - TwoSQRT2*cos(f0 + TwoThirdPi)*a0b0) + w1*b12*(1. + a12 - TwoSQRT2*cos(f1 + TwoThirdPi)*a1b1));
// // 	prob[4] = prefactor*(w0*a02*(1. + b02 - TwoSQRT2*cos(f0 + TwoThirdPi)*a0b0) + w1*a12*(1. + b12 - TwoSQRT2*cos(f1 + TwoThirdPi)*a1b1));
// // 	prob[5] = prefactor*(w0*(a04 + b04 - 2.*cos(2.*f0 - TwoThirdPi)*a02*b02) + w1*(a14 + b14 - 2.*cos(2.*f1 - TwoThirdPi)*a12*b12));
// // 	prob[6] = prefactor*(w0*b02*(1. + a02 - TwoSQRT2*cos(f0 - TwoThirdPi)*a0b0) + w1*b12*(1. + a12 - TwoSQRT2*cos(f1 - TwoThirdPi)*a1b1));
// // 	prob[7] = prefactor*(w0*a02*(1. + b02 - TwoSQRT2*cos(f0 - TwoThirdPi)*a0b0) + w1*a12*(1. + b12 - TwoSQRT2*cos(f1 - TwoThirdPi)*a1b1));
// // 	prob[8] = prefactor*(w0*(a04 + b04 - 2.*cos(2.*f0 + TwoThirdPi)*a02*b02) + w1*(a14 + b14 - 2.*cos(2.*f1 + TwoThirdPi)*a12*b12));
//
//
// 	//Pranjal:
// 	// bool IntermediateCalc = false;
// 	// if(IntermediateCalc){
// 	// 	cout << "NYFunction: Intermediate calculations @" << endl;
// 	// 	cout << "x = {theta1,theta2,phi1,phi2,alpha} = " << vec_to_CommaSeparatedString_with_precision(x,16) << endl;
// 	// }
// 	double theta1 = x[0];
// 	double theta2 = x[1];
// 	double phi1 = x[2];
// 	double phi2 = x[3];
// 	double alpha = x[4];
//     double c1 = cos(theta1); //if(IntermediateCalc) cout << "c1 = " << c1 << endl;
//     double c2 = cos(theta2); //if(IntermediateCalc) cout << "c2 = " << c2 << endl;
//     double s1 = sin(theta1); //if(IntermediateCalc) cout << "s1 = " << s1 << endl;
//     double s2 = sin(theta2); //if(IntermediateCalc) cout << "s2 = " << s2 << endl;
//     double w1 = cos(alpha)*cos(alpha); //if(IntermediateCalc) cout << "w1 = " << w1 << endl;
//     double w2 = 1.-w1; //if(IntermediateCalc) cout << "w2 = " << w2 << endl;
// 	double c12 = c1*c1;
// 	double c14 = c12*c12;
// 	double c22 = c2*c2;
// 	double c24 = c22*c22;
// 	double s12 = s1*s1;
// 	double s14 = s12*s12;
// 	double s22 = s2*s2;
// 	double s24 = s22*s22;
// 	double s1c1 = s1*c1;
// 	double s2c2 = s2*c2;
// 	double s22c22 = s22*c22;
//     prob[0] = prefactor*(w1*s12*(1.+c12 -TwoSQRT2*cos(phi1)*s1c1)+w2*s22*(1.+c22-TwoSQRT2*cos(phi2)*s2c2));
// 	//if(IntermediateCalc) cout << "prob[0] = " << prob[0] << endl;
// 	prob[1] = prefactor*(w1 * (POW(SQRT2*s1c1*cos(phi1)-c12,2) + POW(SQRT2*s1c1*sin(phi1),2)) +  w2 * (POW(SQRT2*s2c2*cos(phi2)-c22,2) + POW(SQRT2*s2c2*sin(phi2),2)));
// 	//if(IntermediateCalc) cout << "prob[1] = " << prob[1] << endl;
// 	prob[2] = prefactor*(w1*(c14+s14 -2.*cos(2.*phi1)*s12*c12)+ w2*(c24+ s24 -  2.*cos(2.*phi2)*s22c22));
// 	//if(IntermediateCalc) cout << "prob[2] = " << prob[2] << endl;
// 	prob[3] = prefactor*(w1*s12*(1.+c12 -TwoSQRT2*cos(phi1+TwoThirdPi)*s1c1)+ w2*s22*(1.+c22-TwoSQRT2*cos(phi2+TwoThirdPi)*s2c2));
// 	//if(IntermediateCalc) cout << "prob[3] = " << prob[3] << endl;
// 	prob[4] = prefactor*(w1*c12*(1.+s12 -TwoSQRT2*cos(phi1+TwoThirdPi)*c1*s1)+ w2*c22*(1.+s22-TwoSQRT2*cos(phi2+TwoThirdPi)*s2c2));
// 	//if(IntermediateCalc) cout << "prob[4] = " << prob[4] << endl;
// 	prob[5] = prefactor*(w1*(c14+s14-2.*cos(2.*phi1 -  TwoThirdPi)*c12*s12)+ w2*(c24+s24-2.*cos(2.*phi2 -  TwoThirdPi)*s22c22));
// 	//if(IntermediateCalc) cout << "prob[5] = " << prob[5] << endl;
// 	prob[6] = prefactor*(w1*s12*(1.+c12-TwoSQRT2*cos(phi1-TwoThirdPi)*s1c1)+ w2*s22*(1.+c22-TwoSQRT2*cos(phi2-TwoThirdPi)*s2c2));
// 	//if(IntermediateCalc) cout << "prob[6] = " << prob[6] << endl;
// 	prob[7] = prefactor*(w1*c12*(1.+s12-TwoSQRT2*cos(phi1-TwoThirdPi)*c1*s1)+ w2*c22*(1.+s22-TwoSQRT2*cos(phi2-TwoThirdPi)*s2c2));
// 	//if(IntermediateCalc) cout << "prob[7] = " << prob[7] << endl;
// 	prob[8] = prefactor*(w1*(c14+s14-2.*cos(2.*phi1+TwoThirdPi)*c12*s12)+ w2*(c24+s24-2.*cos(2.*phi2+TwoThirdPi)*s22c22));
// 	//if(IntermediateCalc) cout << "prob[8] = " << prob[8] << endl;
//
// 	double f = 0.;
// 	for(int p=0;p<9;p++){
// 		if(prob[p]<=1.0+1.0e-15){
// 			if(prob[p]>0.) f -= opt.AuxParams[p] * log(prob[p]);
// 		}
// 		else if(prob[p]>1.0+1.0e-15 || prob[p]<-1.0e-15){
// 			if(prob[p]<0.){
// 				#pragma omp critical
// 				{
// 					opt.report += "NYFunction: Error !!! prob[" + to_string(p) + "] = " + to_string(prob[p]) + "\n";
// 				}
// 			}
// 			return 1.0e+300;
// 		}
// 		//if(IntermediateCalc) cout << "-opt.AuxParams[" << p << "] * log(prob[" << p << "]) = " << -opt.AuxParams[p] * log(prob[p]) << endl;
// 	}
//
// 	//if(IntermediateCalc){ cout << "f = " << f << endl; usleep(10000000); }
//
// 	if(opt.finalcalc){
// 		opt.AuxVec.clear();
// 		opt.AuxVec.insert(opt.AuxVec.end(), prob.begin(), prob.end());
// 	}
//
//     return f;
// }

  
double ObjFunc(vector<double> &x, OPTstruct &opt){
	double res = 0.;

	//Rastrigin, no shifts or rotations
// 	res = 10.*(double)opt.D;
// 	for(int d=0;d<opt.D;d++){
// 		res += 5.12*x[d]*5.12*x[d]-10.*cos(2.*3.141592653589793*5.12*x[d]);
// 	}
  
	//1pEx-DFT for atoms/ions, no frills
// 	int L = opt.D/2, L2=L*L, L3=L2*L;
// 	double Z = opt.ex.params[1];
// 	vector<double> e1p(L,-0.5*Z*Z), ang(L), phi(L);
// 	for(int l=0;l<L;l++)
// 	{
// 		ang[l] = x[l];
// 		phi[l] = x[L+l];
// 		if(l==0) e1p[l] /= 1.*1.;
// 		else if(l<5) e1p[l] /= 2.*2.;
// 		else if(l<14) e1p[l] /= 3.*3.;
// 		else if(l<30) e1p[l] /= 4.*4.;
// 		else if(l<55) e1p[l] /= 5.*5.;
// 		else if(l<91) e1p[l] /= 6.*6.;
// 	}
// 	vector<double> occ = AnglesToOccNum(ang,opt.ex);//includes NearestProperOccNum()
// 	vector<vector<double>> rhoSeed(L);
// 	for(int i=0;i<L;i++)
// 	{
// 		rhoSeed[i].resize(L);
// 		for(int j=0;j<L;j++){
// 			rhoSeed[i][j] = sqrt(occ[i]*occ[j]);//HF for N=2
// 		}
// 	}
// 	
// 	double E1p = ScalarProduct(occ,e1p);
// 	
// 	double Eint = 0.;
// 	for(int i=0;i<L;i++)
// 	{
// 		for(int j=0;j<L;j++)
// 		{
// 			for(int k=0;k<L;k++)
// 			{
// 				for(int l=0;l<L;l++)
// 				{
// 					Eint += 0.5*Z*rhoSeed[i][j]*rhoSeed[k][l]*(opt.ex.Iabcd[i*L3+j*L2+k*L+l]-0.5*opt.ex.Iabcd[i*L3+l*L2+k*L+j])*cos(phi[i]-phi[j]+phi[k]-phi[l]);
// 				}
// 			} 
// 		}
// 	}
// 	
// 	res = E1p + Eint;

	return res;
}  

double MutuallyUnbiasedBases(vector<double> &x, OPTstruct &opt){//mutually unbiased bases
	double res = 0.;	
	int NB = (int)(opt.AuxParams[1]+0.5), NV = (int)(opt.AuxParams[2]+0.5), dim = NV;
	double PenaltyScale = opt.AuxParams[0];
	double OrthogonalityPenalty = 0.;
	if(PenaltyScale>0.){
		for(int b=1;b<NB;b++){
			for(int v=0;v<NV;v++){
				for(int w=v+1;w<NV;w++){
					double A = 0., B = 0.;
					for(int d=0;d<dim;d++){
						double vR = x[b*NV*dim*2+v*dim*2+d*2];
						double vI = x[b*NV*dim*2+v*dim*2+d*2+1];
						double wR = x[b*NV*dim*2+w*dim*2+d*2];
						double wI = x[b*NV*dim*2+w*dim*2+d*2+1];	    
						A += vR*wR+vI*wI;
						B += vR*wI-vI*wR;
					}
					OrthogonalityPenalty += A*A+B*B;
				}
			}
		}
	}
	for(int a=0;a<NB;a++){
		for(int b=a+1;b<NB;b++){
			double Dab2 = 0.;
			for(int v=0;v<NV;v++){
				for(int w=0;w<NV;w++){
					double A = 0., B = 0.;
					for(int d=0;d<dim;d++){
						double aR = x[a*NV*dim*2+v*dim*2+d*2];
						double aI = x[a*NV*dim*2+v*dim*2+d*2+1];
						double bR = x[b*NV*dim*2+w*dim*2+d*2];
						double bI = x[b*NV*dim*2+w*dim*2+d*2+1];	    
						A += aR*bR+aI*bI;
						B += aR*bI-aI*bR;
					}
					double ab2 = A*A+B*B;
					Dab2 += (ab2-1./((double)dim))*(ab2-1./((double)dim));
				}
			}
			Dab2 = 1.-Dab2/((double)(dim-1));
			res += Dab2;
		}
	}
	return PenaltyScale*OrthogonalityPenalty-2.*res/((double)(NB*(NB-1)));
}  

double DFTe_QPot(vector<double> &x, OPTstruct &opt){//DFTe fit to consumer-resource quasi-potential
	double res = 0.;
	int D = x.size();	
	vector<double> c(6);//driver
	double xStep;
	//OptimizationProblem --- 1000: Dakos2012 --- 1001: Nolting2016
	if(opt.function==1000){
		xStep = 10./100.;
		c = {{0.},{1.5},{2.},{2.35},{2.605},{3.}};
	}
	else if(opt.function==1001){
		xStep = 2.2/100.;
		c = {{0.2},{0.4},{0.6},{0.65},{0.7},{1.}};
	}
	for(int i=0;i<c.size();i++)
	{
		for(int k=1;k<=100;k++)
		{
			double diff, X = (double)k*xStep;//observation
			if(opt.function==1000){
				double f = 0.5*X*X+(X/abs(x[1])-1.)*(X/abs(x[1])-1.);//dispersal and resource energy
				f += x[2]*pow(X,abs(x[3]))+x[4]*pow(X,abs(x[5]));//intrinsic interaction
				f += c[i]*(x[6]*pow(X,abs(x[7]))+x[8]*pow(X,abs(x[9])));//response to harvesting
				diff = c[i]*(X-atan(X))-X*X/2.+X*X*X/30. - f;
			}
			else if(opt.function==1001){
				double f = x[1]+x[2]*X+x[3]*X*X+x[4]*X*X*X+x[5]*X*X*X*X+x[6]*c[i]*X;//(*quartic internal pressure + simple interaction*)
				//double f = x[1]+x[2]*X+x[3]*X*X+x[4]*X*X*X+x[5]*X*X*X*X+x[6]*c[i]*X+x[7]*pow(c[i],abs(x[8]))*pow(X,abs(x[9]));//(*quartic internal pressure + advanced interaction --- not really much better*)
				diff = RelDiff(0.125*(4.*X*(-2.-2.*c[i]+X) + 0.3826834323650898*(-2.*atan(2.414213562373095-2.613125929752753*X) + 2.*atan(2.613125929752753*(0.9238795325112867+X)) - log(1.-0.7653668647301796*X+X*X) + log(1.+0.7653668647301796*X+X*X)) + 0.9238795325112867*(-2.*atan(0.41421356237309503-1.082392200292394*X) + 2.*atan(1.082392200292394*(0.3826834323650898+X)) - log(1.-1.8477590650225735*X+X*X) + log(1.+1.8477590650225735*X+X*X))),f);
			}			
			res += diff*diff;
		}
	}
	double test = sqrt(res/((double)(100*c.size())));
	if(std::isfinite(test)) return min(test,1.0e+3);
	else return 1.0e+3;
}

double DynDFTe_TimeSeries(vector<double> &x, OPTstruct &opt){//fit time series data {c,x} to DE(r,kappa,h) or to DFTe(m,kappa,p1,p2,p3,p4,p5,p6,p7,p8)
	double res = 0.;
	//OptimizationProblem --- 1002: Dakos2012, DEfit --- 1003: Dakos2012, DFTefit --- 1004: Nolting2016, DEfit --- 1005: Nolting2016, DFTefit
	string FileName;
	//BEGIN USER INPUT
	bool inertiaQ = true;//if false, then use only dissipative term in DynDFTe; if true, add Newtonian-type dynamics	
	FileName = "TabFunc_TimeSeries_20231023155736.dat";//Dakos2012
	//FileName = "TabFunc_TimeSeries_20231023155736_BeforeCollapse.dat";//Dakos2012
	//FileName = "TabFunc_TimeSeries_20231023155736_BeforeEWS.dat";//Dakos2012
	//FileName = "TabFunc_TimeSeries_20231103112141.dat";//Nolting2016
	//FileName = "TabFunc_TimeSeries_20231103112141_BeforeCollapse.dat";//Nolting2016
	//FileName = "TabFunc_TimeSeries_20231103112141_BeforeEWS.dat";//Nolting2016
	//END USER INPUT
	bool DFTefit = false;
	if(opt.function==1003 || opt.function==1005) DFTefit = true;	
	vector<vector<double>> TimeSeriesData(ReadMat(FileName));
	double dt = 1.;
	int L = TimeSeriesData.size();
	double pop = TimeSeriesData[0][2];
	if(DFTefit){//fit to DFTe
		double mInverse = 0.; if(inertiaQ) mInverse = abs(x[0]);
		double p = 0., f, kappa=max(0.1,abs(x[1])), threshold = 1.0e+3;
		for(int l=0;l<L;l++){
			if(opt.function>=1002 && opt.function<=1003) f = -(pop + (2.*(-1. + pop/kappa))/kappa + x[2]*pow(pop,-1. + abs(x[3]))*abs(x[3]) + x[4]*pow(pop,-1. + abs(x[5]))*abs(x[5]) + TimeSeriesData[l][1]*(x[6]*pow(pop,-1. + abs(x[7]))*abs(x[7]) + x[8]*pow(pop,-1. + abs(x[9]))*abs(x[9])));
			if(opt.function>=1004 && opt.function<=1005) f = -(x[2]+2.*x[3]*pop+3.*x[4]*pop*pop+4.*x[5]*pop*pop*pop+x[6]*TimeSeriesData[l][1]);
			p += dt*mInverse*f;
			pop += dt*(f+p);
			res += (TimeSeriesData[l][2]-pop)*(TimeSeriesData[l][2]-pop);
			if(!(std::isfinite(res))) return 10.*threshold;
		}
		double test = sqrt(res/(double)L);
		if(std::isfinite(test)) return min(test,threshold + log(test));
		else return 10.*threshold;
	}
	else{//fit to DE
		for(int l=0;l<L;l++){
			if(opt.function>=1002 && opt.function<=1003) pop += dt*(abs(x[0])*pop*(1.-pop/abs(x[1]))-TimeSeriesData[l][1]*pop*pop/(pop*pop + abs(x[2])*abs(x[2])));
			if(opt.function>=1004 && opt.function<=1005) pop += dt*(TimeSeriesData[l][1]-abs(x[0])*pop+abs(x[1])*POW(pop,8)/(POW(x[2],8)+POW(pop,8)));
			res += (TimeSeriesData[l][2]-pop)*(TimeSeriesData[l][2]-pop);
		}
		double test = sqrt(res/(double)L), threshold = 1.0e+3;
		if(!(std::isfinite(test))) return 10.*threshold;
		else return min(test,threshold + log(test));		
	}
}


void CGD(OPTstruct &opt){

  double res; string ParamVec = ""; vector<double> InitParam, Scale; InitParam.resize(opt.D); Scale.resize(opt.D); fill(Scale.begin(),Scale.end(),opt.cgd.scale);
  
  if(opt.function == -11){
    for(int l=0;l<opt.D;l++) InitParam[l] = (opt.SearchSpace[l][1]-opt.SearchSpace[l][0])*opt.RNpos(opt.MTGEN); 
  }
  if(opt.function == -1){
    InitParam = OccNumToAngles(InitializeOccNum(0,opt.ex));
  }
  OPTprint("CGD: Start from... " + vec_to_str(InitParam),opt);
  
  real_1d_array x,s;
  x.setcontent(InitParam.size(), &(InitParam[0]));
  s.setcontent(Scale.size(), &(Scale[0]));
  
  mincgstate state;
  mincgcreatef(x, opt.cgd.diffstep, state);
  mincgsetcond(state, opt.cgd.epsg, opt.cgd.epsf, opt.cgd.epsx, opt.cgd.maxits);
  mincgsetscale(state, s);
  mincgsetprecscale(state);

  mincgreport rep;
  mincgoptimize(state, Objective_Function, NULL, (void*)&opt);
  mincgresults(state, x, rep);
    
  string TerminationType;
  switch(rep.terminationtype){
    case -8: { TerminationType = "internal integrity control detected infinite or NAN values in function/gradient. Abnormal termination signalled."; break; }
    case 1:  { TerminationType = "relative function improvement is no more than EpsF = " + to_string_with_precision(opt.cgd.epsf,16); break; }
    case 2:  { TerminationType = "relative step is no more than EpsX = " + to_string_with_precision(opt.cgd.epsx,16); break; }
    case 4:  { TerminationType = "gradient norm is no more than EpsG = " + to_string_with_precision(opt.cgd.epsg,16); break; }
    case 5:  { string MaxIts; if(opt.cgd.maxits==0) MaxIts = "infinite #"; else MaxIts = to_string(opt.cgd.maxits); TerminationType = MaxIts + " steps were taken"; break; }
    case 7:  { TerminationType = "stopping conditions are too stringent, further improvement is impossible, X contains best point found so far."; break; }
    case 8:  { TerminationType = "terminated by user who called mincgrequesttermination(). X contains point which was current accepted when termination request was submitted."; break; }
    default: { break; }
  }
    
  OPTprint("ConjugateGradientDescent: recompute result from conjugate gradient descent:",opt);
  Objective_Function(x,res,(void*)&opt);
  opt.cgd.bestf = res;
  opt.cgd.bestx = real_1d_arrayToVec(x);
  opt.cgd.CGDgr = (int)rep.nfev;
  if(opt.printQ){
    OPTprint("CDG: Candidate for global minimum " + to_string_with_precision(res,10) + " @ {" + vec_to_str_with_precision(opt.cgd.bestx,8) + "}\n" + "                  [ #Iterations = " + to_string(rep.iterationscount) + " #GradEvals = " + to_string(rep.nfev) + " #TerminationType = " + TerminationType + " from starting point" + "{" + vec_to_str_with_precision(InitParam,8) + "} ]",opt);
  } 
  OPTprint("*** CGD completed ***",opt);
}

void LCO(OPTstruct &opt){//linear constraints (equality & inequality & boundary), with nonlinear conjugate gradient method as underlying optimization algorithm
  int breakQ = 0;
  SetupOPTloopBreakFile(opt);  
  
  bool convergedQ = false;
  int MaxRun = (int)(max(1.,(double)opt.lco.runs/((double)opt.threads)));
  opt.lco.evals.resize(opt.threads); fill(opt.lco.evals.begin(),opt.lco.evals.begin(),0);
  opt.lco.History.resize(MaxRun); fill(opt.lco.History.begin(),opt.lco.History.end(),1.0e+300);
  opt.lco.history.resize(MaxRun); for(int Run=0;Run<MaxRun;Run++) opt.lco.history[Run].resize(opt.D);
    
  for(int Run=0;Run<MaxRun;Run++){
      vector<double> bestf(opt.threads);
      vector<vector<double>> bestx(opt.threads); for(int run=0;run<opt.threads;run++) bestx[run].resize(opt.D);
      int runs = min(opt.threads,MaxRun);
      #pragma omp parallel for schedule(static) if(opt.threads>1)
      for(int run=0;run<runs;run++){
	//for all threads:
	OPTstruct opt2;
	opt2.ex = opt.ex;
	opt2.function = opt.function;
	opt2.D = opt.D;
	opt2.epsf = opt.epsf;
	opt2.threads = 1;
	opt2.printQ = 0;
	opt2.nb_eval = 0;
	opt2.SearchSpaceMin = opt.SearchSpaceMin; opt2.SearchSpaceMax = opt.SearchSpaceMax;
	opt2.SearchSpaceLowerVec = opt.SearchSpaceLowerVec; opt2.SearchSpaceUpperVec = opt.SearchSpaceUpperVec;
	SetDefaultLCOparams(opt2);
	opt2.lco.InitParam.resize(opt2.D);
	//thread-specific:
	if(opt2.function == -1 || opt2.function == -2 || opt2.function == -11){
	  if(opt.PostProcessQ || opt.pso.PostProcessQ){
	    opt2.lco.InitParam = opt.currentBestx;
	    if(opt2.function == -2) for(int d=0;d<opt.ex.L;d++) opt2.lco.InitParam[d] = 1.+cos(opt2.lco.InitParam[d]);
	  }
	  else{
	    vector<double> occnum = InitializeOccNum(false,opt.ex);
	    for(int d=0;d<opt2.D;d++){
	      opt2.lco.InitParam[d] = occnum[d];
	      if(opt2.function == -2 && d>=opt.ex.L) opt2.lco.InitParam[d] = alea( opt.SearchSpace[d][0], opt.SearchSpace[d][1], opt );
	    }
	  }
	}
	SetLCOvariables(opt2);
  
//     cout << "LCO test..." << endl; 
//     opt2.function = -1000;
//     real_1d_array x = "[0,0]";//variables
//     real_1d_array s = "[1,1]";//scales
//     real_1d_array bndl = "[-1,-1]";//-INF for unbounded
//     real_1d_array bndu = "[1,1]";//+INF for unbounded
//     real_2d_array c = "[[1,-1,1],[1,1,0.],[1,2,0.1]]";
//     integer_1d_array ct = "[1,0,-1]";//constraint type -> array c means: 1*x-1*y>=1 && 1*x+1*y==0 && 1*x+2*y<=0.1
    
	minbleicstate state;
	minbleiccreatef(opt2.lco.x, opt2.lco.diffstep, state);
	minbleicsetbc(state, opt2.lco.bndl, opt2.lco.bndu);
	minbleicsetlc(state, opt2.lco.c, opt2.lco.ct);
	minbleicsetscale(state, opt2.lco.s);
	minbleicsetprecscale(state);//preconditioner, optional
	minbleicsetcond(state, opt2.lco.epsg, opt2.lco.epsf, opt2.lco.epsx, opt2.lco.maxits);
	//minbleicoptguardsmoothness(state);//optional
    
	minbleicreport rep;
	minbleicoptimize(state, Objective_Function, NULL, (void*)&opt2);
	minbleicresults(state, opt2.lco.x, rep);
    
	//recompute minimum
	Objective_Function(opt2.lco.x,opt2.lco.bestf,(void*)&opt2);
	opt2.lco.bestx = real_1d_arrayToVec(opt2.lco.x);
	string TerminationType;
	switch(rep.terminationtype){
	  case -8: { TerminationType = "internal integrity control detected infinite or NAN values in function/gradient. Abnormal termination signalled."; break; }
	  case -3: { TerminationType = "inconsistent constraints. Feasible point is either nonexistent or too hard to find. Try to restart optimizer with better initial approximation."; break; }
	  case 1:  { TerminationType = "relative function improvement is no more than EpsF = " + to_string_with_precision(opt2.lco.epsf,16); break; }
	  case 2:  { TerminationType = "relative step is no more than EpsX = " + to_string_with_precision(opt2.lco.epsx,16); break; }
	  case 4:  { TerminationType = "gradient norm is no more than EpsG = " + to_string_with_precision(opt2.lco.epsg,16); break; }
	  case 5:  { string MaxIts; if(opt2.lco.maxits==0) MaxIts = "infinite #"; else MaxIts = to_string(opt2.lco.maxits); TerminationType = MaxIts + " steps were taken"; break; }
	  case 7:  { TerminationType = "stopping conditions are too stringent, further improvement is impossible, X contains best point found so far."; break; }
	  case 8:  { TerminationType = "terminated by user. X contains point which was current accepted when termination request was submitted."; break; }
	  default: { break; }
	}    
	bestf[run] = opt2.lco.bestf;
	bestx[run] = opt2.lco.bestx;
	
	opt.lco.evals[run] += opt2.nb_eval;
	
	//optguardreport ogrep; minbleicoptguardresults(state, ogrep); if(ogrep.nonc0suspected) OPTprint("\n LCO: target function possibly discontinuous\n",opt); if(ogrep.nonc1suspected) OPTprint("\n LCO: target function possibly nonsmooth\n",opt);//optional
      }

      opt.lco.bestf = bestf[0]; opt.lco.bestx = bestx[0]; for(int run=0;run<runs;run++) if(bestf[run]<opt.lco.bestf){ opt.lco.bestf = bestf[run]; opt.lco.bestx = bestx[run]; }

      if(opt.pso.PostProcessQ && opt.function == -2) for(int d=0;d<opt.ex.L;d++) opt.lco.bestx[d] = acos(max(0.,min(2.,opt.lco.bestx[d]))-1.);      
      
      OPTprint("LCO(" + to_string((Run+1)*runs) + ") completed: Minimum = " + to_string(opt.lco.bestf) +" @ x =\n" + vec_to_str(opt.lco.bestx) + "\n",opt);
      opt.lco.History[Run] = opt.lco.bestf;
      opt.lco.history[Run] = opt.lco.bestx;
      opt.currentBestf = opt.lco.bestf;
      opt.currentBestx = opt.lco.bestx;
    
      if(opt.lco.VarianceCheck>0) if(VarianceConvergedQ(opt)) break;
      if(manualOPTbreakQ(opt)) break;
      
  }
  
  opt.lco.bestf = opt.lco.History[0];
  opt.lco.bestx = opt.lco.history[0];
  for(int h=1;h<opt.lco.History.size();h++){
    if(opt.lco.History[h]<opt.lco.bestf){
      opt.lco.bestf = opt.lco.History[h];
      opt.lco.bestx = opt.lco.history[h];
    }
  }

}




void AUL(OPTstruct &opt){
  
  if(opt.printQ) cout << "*** AUL completed ***" << endl;
}

void Objective_Function(const real_1d_array &x, double &func, void *ptr){//Objective_Function(x,res,(void*)&opt);
  OPTstruct *opt = (OPTstruct *)ptr;//cast ptr to a pointer that points to an OPTstruct, name it opt, and address/change the members with ->
  
  if(opt->function == -1000){//test function for LCO
    func = 100*pow(x[0]+3,4) + pow(x[1]-3,4);//expected: {15156.31250, {x -> 0.5000000000, y -> -0.5000000000}}, see 1p-exact-DFT_Misc.nb
  }    
  if(opt->function == -11){
    func = Eint_1pExDFT(real_1d_arrayToVec(x),opt->ex);
  }
  else if(opt->function == -2){
    if(opt->ActiveOptimizer==102){
      vector<double> var = real_1d_arrayToVec(x);
      for(int d=0;d<opt->ex.L;d++) var[d] = acos(max(0.,min(2.,var[d]))-1.);
      func = Etot_1pExDFT_CombinedMin(var,opt->ex);
    }
    else func = Etot_1pExDFT_CombinedMin(real_1d_arrayToVec(x),opt->ex);
  }  
  else if(opt->function == -1){
    if(opt->ex.settings[4]==1) func = Etot_1pExDFT(AnglesToOccNum(real_1d_arrayToVec(x),opt->ex),opt->ex);
    else func = Etot_1pExDFT(real_1d_arrayToVec(x),opt->ex);
  }
  opt->nb_eval++;
  if(opt->printQ && printQ(opt->nb_eval)/* && opt->function==-1*/){
    vector<double> param = real_1d_arrayToVec(x); if(opt->function==-1 && opt->ex.settings[4]==1) AnglesToOccNum(param,opt->ex);
    OPTprint("Objective_Function(" + to_string(opt->nb_eval) + "): func = " + to_string_with_precision(func,16) + " --- OccNum = " + vec_to_str_with_precision(param,16),*opt);
  }
}

void SetupOPTloopBreakFile(OPTstruct &opt){
  if(opt.printQ){//open mpDPFTmanualOPTloopBreakQ.dat during runtime, manually change "0" to "1" and save file in order to break OPT loop
    ofstream file;
    file.open("mpDPFTmanualOPTloopBreakQ.dat");
    file << 0 << "\n";
    file.close();  
  }
}

bool manualOPTbreakQ(OPTstruct &opt){
  if(opt.printQ){
    int breakQ = 0;
    ifstream file;
    file.open("mpDPFTmanualOPTloopBreakQ.dat");
    string line;
    getline(file, line); istringstream issx(line); issx >> breakQ;
    file.close();
    if(breakQ){
      if(breakQ==1){
	OPTprint("--------------------------------------- local OPT loop--Manually--Terminated  ---------------------------------------",opt);
	ofstream file;
	file.open("mpDPFTmanualOPTloopBreakQ.dat");
	file << 0 << "\n";
	file.close();  	
      }
      else if(breakQ==2) OPTprint("--------------------------------------- global OPT loop--Manually--Terminated ---------------------------------------",opt);
      return true;
    }
  }
  return false;
}

bool ManualOPTbreakQ(OPTstruct &opt){
	int breakQ = 0;
    ifstream file;
    file.open("mpDPFTmanualOPTloopBreakQ.dat");
    string line;
    getline(file, line); istringstream issx(line); issx >> breakQ;
    file.close();
    if(breakQ){
		if(breakQ==1){
			OPTprint("--------------------------------------- local OPT loop--Manually--Terminated  ---------------------------------------",opt);
			ofstream file;
			file.open("mpDPFTmanualOPTloopBreakQ.dat");
			file << 0 << "\n";
			file.close();  	
		}
		else if(breakQ==2) OPTprint("--------------------------------------- global OPT loop--Manually--Terminated ---------------------------------------",opt);
		return true;
    }
	return false;
}

void InitRandomNumGenerator(OPTstruct &opt){
    double MaxInt = (double)(std::numeric_limits<int>::max()), Now = ABS((double)chrono::high_resolution_clock::now().time_since_epoch().count() / 1.0e+10), now = (double)((int)Now), seed = 1.0e+10*(Now-now);
    if(seed<1.0e-6) seed = (double)rand();
    while(seed>=MaxInt) seed /= 2.; while(seed<1.0e+06) seed *= 2.;
    mt19937_64 MTGEN((int)seed);
    uniform_real_distribution<double> RNpos(0.,1.0);
    normal_distribution<> RNnormal(0.,1.0);	
    opt.MTGEN = MTGEN;
    opt.RNpos = RNpos;
	opt.RNnormal = RNnormal;
}

void SetDefaultOPTparams(OPTstruct &opt){
  InitRandomNumGenerator(opt);
  opt.timeStamp = YYYYMMDD() + hhmmss();
}

void RandomizeOptLocation(OPTstruct &opt){
	int D = opt.D;
	opt.Rand = true;
	opt.RandShift.clear(); opt.RandShift.resize(D);
	for(int i=0;i<D;i++) opt.RandShift[i] = alea(opt.SearchSpace[i][0]+0.25*opt.searchSpaceExtent[i],opt.SearchSpace[i][1]-0.25*opt.searchSpaceExtent[i],opt);

	// generate a random rotation matrix in D dimensions
	MatrixXd A(D,D);
	for(int i=0;i<D;i++) for(int j=0;j<D;j++) A(i,j) = opt.RNnormal(opt.MTGEN);
	FullPivHouseholderQR<MatrixXd> qr(A);
	opt.RandMat = qr.matrixQ();
	opt.RandMat.conservativeResize(D,D);// In some cases, opt.RandMat may be larger than D x D, so we trim it down
	if (opt.RandMat.determinant() < 0) opt.RandMat.col(0) = -opt.RandMat.col(0);// Ensure the matrix is a proper rotation (determinant +1)

	//for testing
	//opt.RandMat = MatrixXd::Identity(opt.D,opt.D);

	OPTprint("RandomizeOptLocation: RandShift\n" + vec_to_str(opt.RandShift),opt);
	cout << "RandomizeOptLocation: RandMat\n" << static_cast<Eigen::MatrixXd>(opt.RandMat) << endl;

	opt.RandMatInv = opt.RandMat.transpose();

	MatrixXd NullQ = opt.RandMatInv*opt.RandMat - MatrixXd::Identity(opt.D,opt.D);
	if(NullQ.norm()>(double)(D*D)*1.0e-12) cout << "RandomizeOptLocation: Warning !!! Inverse of RandMat not found:" << endl;
	else cout << "RandomizeOptLocation: Success!" << endl;
	cout << static_cast<Eigen::MatrixXd>(NullQ) << endl;
}

void InitializePSOrun(OPTstruct &opt){
  int S = opt.pso.CurrentSwarmSize;
  if(opt.loopCount==0){
    OPTprint("............... initialize PSO run " + to_string(opt.localrun+1),opt);
    opt.pso.besthistory.clear(); opt.pso.besthistory.resize(0);
    opt.pso.loopcounthistory.clear(); opt.pso.loopcounthistory.resize(0);
    opt.pso.besthistoryf.clear(); opt.pso.besthistoryf.resize(0);
    opt.pso.SuccessfulW.clear(); opt.pso.SuccessfulW.resize(0); opt.pso.SuccessfulW.push_back(opt.pso.w);
    opt.pso.SuccessfulC1.clear(); opt.pso.SuccessfulC1.resize(0); opt.pso.SuccessfulC1.push_back(opt.pso.c);
    opt.pso.SuccessfulC2.clear(); opt.pso.SuccessfulC2.resize(0); opt.pso.SuccessfulC2.push_back(opt.pso.c);
    opt.pso.centralW = opt.pso.w;
    opt.pso.centralC1 = opt.pso.c;
    opt.pso.centralC2 = opt.pso.c;
    opt.pso.UpdateParamsQ = false;
    if(opt.localrun>0){//MIT
      double Sd = (double)opt.pso.InitialSwarmSize*pow(EXP(log(opt.pso.increase)/((double)opt.pso.runs)),(double)(opt.localrun+1));
      if((int)Sd==opt.pso.CurrentSwarmSize) opt.pso.NewSwarmSizeQ = false;
      else opt.pso.NewSwarmSizeQ = true;
      S = (int)Sd; if(S>opt.pso.Smax) S=opt.pso.Smax;
      InitializeSwarmSizeDependentVariables(S,opt);
	  opt.pso.InitialSwarmSizeCurrentRun = S;
      OPTprint("............... new swarm size = " + to_string(S),opt);
      if(opt.ResetSearchSpaceQ) opt.SearchSpace = opt.InitSearchSpace;
      if(opt.pso.CoefficientDistribution==1 || opt.pso.CoefficientDistribution==4){
	opt.pso.bestHistory.push_back(opt.pso.best);//store best index of every run
	opt.pso.loopCountHistory.push_back(opt.loopCount);//store loopCount of every run
	if(opt.pso.CoefficientDistribution==1 || opt.localrun<opt.pso.MinimumRandomInitializations){
	  for(int h=0;h<opt.pso.History.size();h++){//for each run in History...
	    if(opt.pso.History[h]<opt.pso.bestThreshold){//check if bestf below threshold...
	      int focalbest = opt.pso.bestHistory[h], HB = h; double checkf = opt.pso.History[h];
	      for(int hb=0;hb<opt.pso.bestHistory.size();hb++){//and select the lowest bestf among those with fixed best index (focalbest)
		if(opt.pso.bestHistory[hb]==focalbest && opt.pso.History[hb]<checkf){ checkf = opt.pso.History[hb]; HB = hb; }
	      }
	      opt.pso.besthistory.push_back(focalbest);
	      opt.pso.besthistoryf.push_back(checkf);
	      opt.pso.loopcounthistory.push_back(opt.pso.loopCountHistory[HB]);
	    }
	  }
	}
      }
    }

    opt.maxEC = vector<vector<double>>(S,vector<double>(opt.NumEC,0.));
	opt.maxEC2 = vector<vector<double>>(S,vector<double>(opt.NumEC,0.));
	opt.maxIC = vector<vector<double>>(S,vector<double>(opt.NumIC,0.));
	opt.maxIC2 = vector<vector<double>>(S,vector<double>(opt.NumIC,0.));
	if((int)opt.PenaltyMethod[0]>0){
		opt.ALmu = vector<vector<double>>(S,vector<double>(1,opt.PenaltyMethod[1]));
		opt.ALmu2 = vector<vector<double>>(S,vector<double>(1,opt.PenaltyMethod[1]));
		opt.ALlambda = vector<vector<vector<double>>>(opt.NumEC+opt.NumIC,vector<vector<double>>(S,vector<double>(1,opt.PenaltyMethod[2])));
		opt.ALlambda2 = vector<vector<vector<double>>>(opt.NumEC+opt.NumIC,vector<vector<double>>(S,vector<double>(1,opt.PenaltyMethod[2])));
	}

  }
  
  if(opt.localrun==0 || opt.pso.NewSwarmSizeQ){
    opt.pso.X.clear(); opt.pso.Xold.clear(); opt.pso.P.clear(); opt.pso.V.clear(); opt.pso.Vmin.clear(); opt.pso.Vmax.clear(); opt.pso.W.clear(); opt.pso.C1.clear(); opt.pso.C2.clear(); opt.pso.I.clear(); opt.pso.R.clear(); opt.pso.delta.clear(); opt.pso.phi.clear();
    opt.pso.X.resize(S); opt.pso.Xold.resize(S); opt.pso.P.resize(S); opt.pso.V.resize(S); opt.pso.Vmin.resize(S); opt.pso.Vmax.resize(S);
    opt.pso.W.resize(S); for(int s=0;s<S;s++) opt.pso.W[s].resize(4);
    opt.pso.C1.resize(S); for(int s=0;s<S;s++) opt.pso.C1[s].resize(4);
    opt.pso.C2.resize(S); for(int s=0;s<S;s++) opt.pso.C2[s].resize(4);
    opt.pso.I.resize(S); opt.pso.R.resize(S); opt.pso.delta.resize(S); opt.pso.phi.resize(S);
  }
  
  if(opt.loopCount==0 || opt.pso.NewSwarmSizeQ){
    for(int s=0;s<S;s++){
      opt.pso.X[s].x.resize(opt.D);
      opt.pso.Xold[s].x.resize(opt.D);
      opt.pso.P[s].x.resize(opt.D);
      opt.pso.V[s].v.resize(opt.D);
      opt.pso.Vmin[s] = opt.pso.vmin;
      opt.pso.Vmax[s] = opt.pso.vmax;
      if(opt.pso.CoefficientDistribution==4 && opt.localrun>=opt.pso.MinimumRandomInitializations) ReinitializePSOparams(s,opt);
      else{
	int which = WhichIntegerElementQ(s,opt.pso.besthistory);
	if( opt.pso.CoefficientDistribution==1 || (opt.pso.CoefficientDistribution==4 && opt.localrun<opt.pso.MinimumRandomInitializations) ){
	  if(which==0) ReinitializePSOparams(s,opt);
	  else OPTprint("keep PSO-parameters of (former) best particle (bestf=" + to_string_with_precision(opt.pso.besthistoryf[which-1],16) + ",FuncEvals=" + to_string(opt.pso.loopcounthistory[which-1]*S) + ") " + to_string(s) + "\n ---  W = " + vec_to_str(opt.pso.W[s]) + "\n --- C1 = " + vec_to_str(opt.pso.C1[s]) + "\n --- C2 = " + vec_to_str(opt.pso.C2[s]) + "\n",opt);
	}
	else{
	  opt.pso.W[s][0] = opt.pso.w;
	  opt.pso.C1[s][0] = opt.pso.c;
	  opt.pso.C2[s][0] = opt.pso.c;
	}
      }
      opt.pso.I[s] = opt.pso.InformerProbability;
      opt.pso.R[s] = (double)opt.pso.reseedEvery;
    }
    opt.pso.LINKS.resize(S); for(int s=0;s<S;s++) opt.pso.LINKS[s].resize(S); // Information links
  }
  
  
  if(opt.loopCount==0){
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for (int s = 0; s < S; s++ ) InitializeSwarmParticle(s,opt);// Positions and velocities

    // First evaluations
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for (int s = 0; s < S; s++ )
    {
      opt.pso.X[s].f = GetFuncVal( s, opt.pso.X[s].x, opt.function, opt );
      opt.pso.P[s] = opt.pso.X[s]; // Best position = current one
    }    
    opt.nb_eval += (double)S;
    opt.pso.nb_eval += S;
    
    opt.pso.OldBestf = opt.pso.X[0].f;
    opt.pso.spread = opt.pso.X[0].f;
    for (int s = 1; s < S; s++ ){
      if(opt.pso.X[s].f<opt.pso.OldBestf) opt.pso.OldBestf = opt.pso.X[s].f;
      else if(opt.pso.X[s].f>opt.pso.spread) opt.pso.spread = opt.pso.X[s].f;
    }
    opt.pso.spread -= opt.pso.OldBestf;
    
    opt.pso.init_links = 1; // So that information links will be initialized 
  }
    
  // Find the best ... and worst
  opt.pso.best = 0;
  opt.pso.worst = 0;
  for (int s = 1; s < S; s++ ){
    if ( opt.pso.P[s].f < opt.pso.P[opt.pso.best].f ) opt.pso.best = s;
    else if ( opt.pso.P[s].f > opt.pso.P[opt.pso.worst].f ) opt.pso.worst = s;
  }
  opt.pso.error =  opt.pso.P[opt.pso.best].f ; // Current min error
  
  if(opt.loopCount==0){
    opt.pso.error_prev=opt.pso.error; // Previous min error
    if(opt.localrun==0){ opt.pso.bestf = opt.pso.error; for (int d = 0; d < opt.D; d++ ) opt.pso.bestx[d] = opt.pso.P[opt.pso.best].x[d]; }
    opt.pso.History.push_back(opt.pso.P[opt.pso.best].f);
    opt.pso.HistoryX.push_back(opt.pso.P[opt.pso.best].x);
  }
}

bool ConvergedQ(int n_exec, OPTstruct &opt){
  int H = opt.pso.History.size(), MinEncounters = 0;
  double min = *min_element(opt.pso.History.begin(),opt.pso.History.end());
  if(H>1) for(int h=0;h<H;h++) if(opt.pso.History[h]-min<sqrt(opt.pso.epsf)) MinEncounters++;
  vector<double> SortedHistory = opt.pso.History;
  sort(SortedHistory.begin(),SortedHistory.end());
  if(MinEncounters>=opt.pso.TargetMinEncounters){
    OPTprint("ConvergedQ: PSO converged @ run=" + to_string(n_exec) + "(H=" + to_string(H) + ") --- History (sorted):\n" + vec_to_str(SortedHistory) + " ...for MinEncounters (deviation<sqrt(opt.pso.epsf)) = " + to_string(MinEncounters) + "/" + to_string(opt.pso.TargetMinEncounters),opt);
    return true; 
  }
  else{
	  string his = vec_to_str_with_precision(SortedHistory,12);
	  if(SortedHistory.size()>8) his = partial_vec_to_str_with_precision(SortedHistory,0,3,12) + " .......... " +  partial_vec_to_str_with_precision(SortedHistory,SortedHistory.size()-4,SortedHistory.size()-1,12);	  
    OPTprint("ConvergedQ: Minimum[run=" + to_string(n_exec) + "/" + to_string(opt.pso.runs) + "(H=" + to_string(H) + ")] = " + to_string(min) + " for MinEncounters (deviation<sqrt(opt.pso.epsf)) = " + to_string(MinEncounters) + "/" + to_string(opt.pso.TargetMinEncounters) + ") --- History (sorted):\n" + his + "\n Best minimum so far: " + to_string_with_precision(min,16),opt);
    return false;
  }
}

bool PostProcess(OPTstruct &opt){
  int ActiveOptimizer_original = opt.ActiveOptimizer;
  double bf = opt.currentBestf;
  vector<double> bx = opt.currentBestx;
  SetDefaultLCOparams(opt);
  opt.lco.runs = 1;
  opt.lco.epsf = 1.0e+2*opt.epsf;
  opt.lco.VarianceCheck = 0;
  opt.lco.diffstep = 1.0e-6;
  //opt.lco.maxits = 10*opt.D;
  LCO(opt);//change currentBest
  opt.ActiveOptimizer = ActiveOptimizer_original;
  if(bf-opt.lco.bestf>1.0e-12){
    OPTprint("\n  ### PostProcessing successful (" + to_string(accumulate(opt.lco.evals.begin(),opt.lco.evals.end(),0)) + " accumulated LCO function evaluations): " + to_string_with_precision(bf,16) + " ---> " + to_string_with_precision(opt.currentBestf,16) + " @ new currentBestx = \n" + vec_to_str_with_precision(opt.currentBestx,8),opt);
    return true;
  }
  else{//reset currentBest
    opt.currentBestf = bf;
    opt.currentBestx = bx;
    return false;
  }
}

bool VarianceConvergedQ(OPTstruct &opt){
	opt.varianceUpdatedQ = false;
	int N, n = opt.Historyf.size();
	if(opt.ActiveOptimizer==101){
		N = opt.pso.VarianceCheck;
		opt.VarianceThreshold = sqrt((double)N)*opt.pso.epsf;
	}
	else if(opt.ActiveOptimizer==102){
		N = opt.lco.VarianceCheck;
		opt.VarianceThreshold = sqrt((double)N)*opt.lco.epsf;    
	}
	else if(opt.ActiveOptimizer==104){
		N = opt.gao.VarianceCheck;
		opt.VarianceThreshold = sqrt((double)N)*opt.epsf;    
	}  
	else if(opt.ActiveOptimizer==105){
		N = opt.csa.VarianceCheck;
		opt.VarianceThreshold = sqrt((double)N)*opt.csa.epsf;   
	}  
	else if(opt.ActiveOptimizer==106){
		N = opt.cma.VarianceCheck;
		opt.VarianceThreshold = sqrt((double)N)*opt.epsf;    
	}     
	if(N==0) return true;
  
	if(n<N){ opt.Historyf.push_back(opt.currentBestf); return false; }
	sort(opt.Historyf.begin(),opt.Historyf.end());
	if(opt.currentBestf<opt.Historyf[N-1]){
		opt.varianceUpdatedQ = true;
		opt.Historyf[N-1] = opt.currentBestf;
	}
  
	double mean = accumulate(opt.Historyf.begin(),opt.Historyf.end(),0.)/((double)N);
	opt.CurrentVariance = 0.;
	for(int h=0;h<N;h++) opt.CurrentVariance += (opt.Historyf[h]-mean)*(opt.Historyf[h]-mean);
	opt.CurrentVariance /= ((double)N);
  
	opt.CurrentVariance = sqrt(opt.CurrentVariance);
  
	if(opt.CurrentVariance<opt.VarianceThreshold){
		opt.printQ = 1;
		string his = vec_to_str_with_precision(opt.Historyf,12);
		if(opt.Historyf.size()>8) his = partial_vec_to_str_with_precision(opt.Historyf,0,3,12) + " .......... " +  partial_vec_to_str_with_precision(opt.Historyf,opt.Historyf.size()-4,opt.Historyf.size()-1,12);
		OPTprint("VarianceConvergedQ(" + to_string(n) + ") check " + to_string_with_precision(opt.currentBestf,8) + " --> variance = " + to_string_with_precision(opt.CurrentVariance,8) + " converged (threshold=" + to_string_with_precision(opt.VarianceThreshold,8) + ")\n opt.Historyf = " + his,opt);
		sort(opt.Historyf.begin(),opt.Historyf.end());
		if(opt.ActiveOptimizer==101){
			vector<double> SortedHistory = opt.pso.History;
			sort(SortedHistory.begin(),SortedHistory.end());
			his = vec_to_str_with_precision(SortedHistory,12);
			if(SortedHistory.size()>8) his = partial_vec_to_str_with_precision(SortedHistory,0,3,12) + " .......... " +  partial_vec_to_str_with_precision(SortedHistory,SortedHistory.size()-4,SortedHistory.size()-1,12);	   
			OPTprint("History across runs (sorted): " + his + "\n",opt);
			if(SortedHistory.size()>=opt.pso.bestToKeep) opt.pso.bestThreshold = SortedHistory[opt.pso.bestToKeep-1]; else opt.pso.bestThreshold = SortedHistory[SortedHistory.size()-1];
		}
		opt.reportQ = 2;
		return true;
	}
	else{
		string his = vec_to_str_with_precision(opt.Historyf,12);
		if(opt.Historyf.size()>8) his = partial_vec_to_str_with_precision(opt.Historyf,0,3,12) + " .......... " +  partial_vec_to_str_with_precision(opt.Historyf,opt.Historyf.size()-4,opt.Historyf.size()-1,12);
		if(opt.printQ && opt.reportQ==2) OPTprint("VarianceConvergedQ(" + to_string(n) + ") check " + to_string_with_precision(opt.currentBestf,8) + " --> variance = " + to_string_with_precision(opt.CurrentVariance,8) + " NOT converged (threshold=" + to_string_with_precision(opt.VarianceThreshold,8) + ")\n opt.Historyf = " + his,opt);
		return false;
	}
}

bool UpdateSearchSpace(OPTstruct &opt){
	bool UpdatedQ = false;
	if(opt.UpdateSearchSpaceQ>0){
		if(opt.printQ && opt.reportQ>1) OPTprint("UpdateSearchSpace:",opt);
		double threshold = 0.05, expandFactor = 0.2, shrinkFactor = 0.1;
		vector<int> update(opt.D); fill(update.begin(),update.end(),0);
			
		vector<double> bestx(opt.D);
		if(opt.ActiveOptimizer==101) bestx = opt.pso.P[opt.pso.best].x;
		else if(opt.ActiveOptimizer==104) bestx = opt.gao.bestx;
		else if(opt.ActiveOptimizer==106) bestx = opt.currentBestx;
			
		for(int i=0;i<opt.D;i++){//shift search space
			double extent = opt.searchSpaceExtent[i];
			if(extent>1.0e-12){
				if((opt.VariableLowerBoundaryQ[i] || opt.UpdateSearchSpaceQ==2) && bestx[i]<opt.SearchSpace[i][0]+threshold*extent ){
					opt.SearchSpace[i][0] -= expandFactor*extent;
					opt.SearchSpace[i][1] -= expandFactor*extent;
					update[i] = 1; UpdatedQ = true;
				}
				else if( (opt.VariableUpperBoundaryQ[i] || opt.UpdateSearchSpaceQ==2) && bestx[i]>opt.SearchSpace[i][1]-threshold*extent ){
					opt.SearchSpace[i][1] += expandFactor*extent;
					opt.SearchSpace[i][0] += expandFactor*extent;
					update[i] = 1; UpdatedQ = true;
				}
				if(update[i]==1 && opt.UpdateSearchSpaceQ==2){
					opt.SearchSpace[i][0] = max(opt.SearchSpace[i][0],opt.InitSearchSpace[i][0]);
					opt.SearchSpace[i][1] = min(opt.SearchSpace[i][1],opt.InitSearchSpace[i][1]);
				}	  
			}	
		}
		for(int i=0;i<opt.D;i++){//shrink search space
			double extent = opt.searchSpaceExtent[i], centre = opt.SearchSpace[i][0]+0.5*extent;
			if(extent>1.0e-12){
				if(ABS(centre-bestx[i])<threshold*extent && ABS(centre-bestx[i])<threshold*extent){
					if(opt.VariableLowerBoundaryQ[i] || opt.UpdateSearchSpaceQ==2){
						opt.SearchSpace[i][0] += shrinkFactor*extent; update[i] = -1;
					}
					if(opt.VariableUpperBoundaryQ[i] || opt.UpdateSearchSpaceQ==2){
						opt.SearchSpace[i][1] -= shrinkFactor*extent; update[i] = -1;
					}
				}
				else if(ABS(opt.SearchSpace[i][0]-bestx[i])<threshold*extent && ABS(opt.SearchSpace[i][0]-bestx[i])<threshold*extent){ 
					if(opt.VariableUpperBoundaryQ[i] || opt.UpdateSearchSpaceQ==2){
						opt.SearchSpace[i][1] -= shrinkFactor*extent; update[i] = -1;
					}	    
				}
				else if(ABS(opt.SearchSpace[i][1]-bestx[i])<threshold*extent && ABS(opt.SearchSpace[i][1]-bestx[i])<threshold*extent){
					if(opt.VariableLowerBoundaryQ[i] || opt.UpdateSearchSpaceQ==2){
						opt.SearchSpace[i][0] += shrinkFactor*extent; update[i] = -1;
					}	    
				}	  
			}
		}
			
		for(int i=0;i<opt.D;i++){
			opt.SearchSpaceLowerVec[i] = opt.SearchSpace[i][0];
			opt.SearchSpaceUpperVec[i] = opt.SearchSpace[i][1];
			if(opt.printQ && opt.reportQ>1){
				string param = ": [" + to_string_with_precision(opt.SearchSpace[i][0],8) + "," + to_string_with_precision(opt.SearchSpace[i][1],8) + "]";
				if(update[i]==1) OPTprint(      " search space shifted    >>>> @ param " + to_string(i) + param,opt);
				else if(update[i]==0) OPTprint( " search space unchanged  .... @ param " + to_string(i) + param,opt);
				else if(update[i]==-1) OPTprint(" search space contracted ---- @ param " + to_string(i) + param,opt);
			}
		}
     
		opt.SearchSpaceExtent = 0.;
		for(int i=0;i<opt.D;i++){
			opt.searchSpaceExtent[i] = opt.SearchSpace[i][1]-opt.SearchSpace[i][0];
			opt.SearchSpaceExtent += opt.searchSpaceExtent[i]/((double)opt.D);//average search space extent
		}
      
	}
	return UpdatedQ;
}

void GetFreeIndices(OPTstruct &opt){
	cout << "GetFreeIndices... requires OccNum & Phases !!!!" << endl;
	int L = opt.ex.settings[0];
	opt.ex.FreeIndices.clear(); opt.ex.FreeIndices.resize(0);
	//for atomic systems
    if(opt.ex.settings[19]>100){
		//opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		//opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		int OccupiedCore = opt.ex.settings[19] % 100;
		cout << "OccupiedCore " << OccupiedCore << endl;
		for(int i=0;i<L;i++){
			if(i<OccupiedCore){//freeze the first few (OccupiedCore) levels as fully occupied
				opt.SearchSpaceLowerVec[i] = 0.;
				opt.SearchSpaceUpperVec[i] = 0.;
				opt.SearchSpaceLowerVec[L+i] = 0.;
				opt.SearchSpaceUpperVec[L+i] = 0.;
			}
			else{//use only S (opt.ex.settings[19]<200), S&P (opt.ex.settings[19]<300), S&P&D (opt.ex.settings[19]<400), etc. levels -> freeze other levels as unoccupied
				bool SQ = false, PQ = false, DQ = false, FQ = false;
				if(i==0 || i==1 || i==5 || i==14 || i==30 || i==55) SQ = true;
				else if( (i>=2 && i<=4) || (i>=6 && i<=8) || (i>=15 && i<=17) || (i>=31 && i<=33) || (i>=56 && i<=58) ) PQ = true;
				else if( (i>=9 && i<=13) || (i>=18 && i<=22) || (i>=34 && i<=38) || (i>=59 && i<=63) ) DQ = true;
				else if( (i>=23 && i<=29) || (i>=39 && i<=45) || (i>=64 && i<=70) ) FQ = true;
				if( (opt.ex.settings[19]<200 && !SQ) || (opt.ex.settings[19]<300 && !SQ && !PQ) || (opt.ex.settings[19]<400 && !SQ && !PQ && !DQ) || (opt.ex.settings[19]<500 && !SQ && !PQ && !DQ && !FQ) ){
					opt.SearchSpaceLowerVec[i] = 3.141592653589793;
					opt.SearchSpaceUpperVec[i] = 3.141592653589793;
					opt.SearchSpaceLowerVec[L+i] = 0.;
					opt.SearchSpaceUpperVec[L+i] = 0.;
				}
				else opt.ex.FreeIndices.push_back(i);
				cout << "GetFreeIndices " << i << endl;
			}
		}	
		cout << "SetDefaultSearchSpace: opt.ex.FreeIndices.size() " << opt.ex.FreeIndices.size() << endl;
	}
	else for(int i=0;i<L;i++) opt.ex.FreeIndices.push_back(i);
	cout << "FreeIndices determined." << endl;
}

void SetDefaultSearchSpace(OPTstruct &opt){
  opt.InitSearchSpace.resize(opt.D);
  opt.SearchSpace.resize(opt.D);
  opt.searchSpaceExtent.resize(opt.D);
  opt.SearchSpaceCentre.resize(opt.D);
  
  if(opt.ex.System>=100) GetFreeIndices(opt);	  
  
  for(int i=0;i<opt.D;i++){
    opt.InitSearchSpace[i].resize(2);
    opt.SearchSpace[i].resize(2);
    if(opt.SearchSpaceMin==0. && opt.SearchSpaceMax==0.){
      opt.SearchSpace[i][0] = opt.SearchSpaceLowerVec[i];
      opt.SearchSpace[i][1] = opt.SearchSpaceUpperVec[i];      
    }   
    else{
      opt.SearchSpace[i][0] = opt.SearchSpaceMin;
      opt.SearchSpace[i][1] = opt.SearchSpaceMax;
      opt.SearchSpaceLowerVec[i] = opt.SearchSpaceMin;
      opt.SearchSpaceUpperVec[i] = opt.SearchSpaceMax;      
    }
  }
  opt.InitSearchSpace = opt.SearchSpace;
  
  opt.SearchSpaceExtent = 0.;
  for(int i=0;i<opt.D;i++){
    opt.searchSpaceExtent[i] = opt.SearchSpace[i][1]-opt.SearchSpace[i][0];
    opt.SearchSpaceExtent += opt.searchSpaceExtent[i]/((double)opt.D);
	opt.SearchSpaceCentre[i] = opt.SearchSpace[i][0]+0.5*opt.searchSpaceExtent[i];
  }
  
  opt.currentBestx.resize(opt.D);
  
  if(opt.UpdateSearchSpaceQ==2){
    opt.VariableLowerBoundaryQ.clear(); opt.VariableLowerBoundaryQ.resize(opt.D);
    opt.VariableUpperBoundaryQ.clear(); opt.VariableUpperBoundaryQ.resize(opt.D);
    for(int d=0;d<opt.D;d++){
      opt.VariableLowerBoundaryQ[d] = false;
      opt.VariableUpperBoundaryQ[d] = false;
    }
  }
  
  if(opt.PCARBF) SetDefaultPCIparams(opt);
  //cout << "Default SearchSpace set." << endl;



}

vector<vector<double>> PCA(OPTstruct &opt){
  int NumVecs = opt.pci.xList.size();// opt.D*NumVecs=10^9 data values -> 40 seconds with pcaType==1
  
  vector<vector<double>> pca(opt.D); for(int eigvec=0;eigvec<opt.D;eigvec++) pca[eigvec].resize(opt.D);
  opt.pci.PCAeigValues.resize(opt.pci.dim);
  real_2d_array inputMatrix; inputMatrix.setlength(NumVecs,opt.D);
  alglib::ae_int_t info;// integer status code?
  alglib::real_1d_array eigValues;// scalar values that describe variances along each eigenvector
  alglib::real_2d_array eigVectors;// targeted orthogonal basis as unit eigenvectors 
  #pragma omp parallel for schedule(static) if(opt.threads>1)
  for(int i=0;i<NumVecs;i++) for(int d=0;d<opt.D;d++) inputMatrix[i][d] = opt.pci.xList[i][d];

  if(opt.pci.pcaType==0) pcabuildbasis(inputMatrix, NumVecs, opt.D, eigValues, eigVectors);//full pca
  else if(opt.pci.pcaType==1) pcatruncatedsubspace(inputMatrix, NumVecs, opt.D, opt.pci.dim, opt.pci.eps, opt.pci.maxits, eigValues, eigVectors);//truncated pca
  for(int eigvec=0;eigvec<opt.pci.dim;eigvec++){
    vector<double> EigVec(opt.D);
    for(int coor=0;coor<opt.D;coor++){ EigVec[coor] = eigVectors[coor][eigvec]; opt.pci.PCAeigValues[eigvec] = eigValues[eigvec]; }
    double normEigVec = Norm(EigVec), sign = 1.;
    if(abs(*min_element(EigVec.begin(),EigVec.end()))>abs(*max_element(EigVec.begin(),EigVec.end()))) sign = -1.;
    for(int coor=0;coor<opt.D;coor++) pca[eigvec][coor] = sign*eigVectors[coor][eigvec]/normEigVec;
  }

  //for(int eigvec=0;eigvec<opt.pci.dim;eigvec++) OPTprint("PCA unit eigenvector" + to_string(eigvec) + " (truncated PCA) = {" + vec_to_str_with_precision(pca[eigvec],6) + "}\n",opt);
  OPTprint("PCA eigenValues = " + vec_to_str(opt.pci.PCAeigValues) + "\n",opt);
  return pca;
}

void UpdatePCARBF(OPTstruct &opt){
  if(opt.pci.enabled){
    int productionLayerIndex = opt.pci.RBFinterpolant.size();
    
    //initialize containers
    if(productionLayerIndex==0 && opt.pci.fList.size()==0){
      //if(opt.pci.UseSuccessfulPredictions){ opt.pci.SuccessfulPredictions.clear(); opt.pci.SuccessfulPredictions.resize(0); }
      opt.pci.fList.resize(1);
      opt.pci.yList.resize(1);
      opt.pci.MoveAcceptanceRate.resize(opt.threads);
    }

    //keep collecting data pairs {x,f(x)}
    if(opt.pci.xList.size()<opt.pci.NumPoints){
      if(opt.ActiveOptimizer==101){
        for(int s=0;s<opt.pso.X.size();s++){
	  opt.pci.xList.push_back(opt.pso.X[s].x);
	  opt.pci.fList[productionLayerIndex].push_back(opt.pso.X[s].f); 
        }
      }
    }
    else{
      startTimer("\n UpdatePCARBF",opt);
      PreparePCARBF(opt);
      //(un)comment if needed; export raw data {xVec,f} to file
      MatrixToFile(opt.pci.xList,"mpDPFT_OPT_RawData_x_" + to_string(productionLayerIndex) + ".dat",16);
      MatrixToFile(opt.pci.fList,"mpDPFT_OPT_RawData_f.dat",16);
      //get principal components
      opt.pci.W.push_back(PCA(opt));
      //fill yList
      int NumVecs = opt.pci.xList.size(); opt.pci.yList[productionLayerIndex].resize(NumVecs);
      #pragma omp parallel for schedule(static) if(opt.threads>1)
      for(int i=0;i<NumVecs;i++){
	opt.pci.yList[productionLayerIndex][i].resize(opt.pci.dim);
	for(int d=0;d<opt.pci.dim;d++) opt.pci.yList[productionLayerIndex][i][d] = ScalarProduct(opt.pci.xList[i],opt.pci.W[productionLayerIndex][d]);
      }
      //build interpolant and calculate required parameters
      if(opt.ActiveOptimizer==101) opt.pci.fshift = opt.pso.bestf;
      BuildRBFinterpolant(opt.pci.yList[productionLayerIndex],opt.pci.fList[productionLayerIndex],opt);
      //preparations for next PCARBF-update
      vector<vector<double>> nextyList(0); vector<double> nextfList(0); opt.pci.yList.push_back(nextyList); opt.pci.fList.push_back(nextfList);
      opt.pci.xList.clear(); opt.pci.xList.resize(0); 
      if(opt.pci.UseSuccessfulPredictions && productionLayerIndex>0){
	for(int i=0;i<opt.pci.SuccessfulPredictions.size();i++){
	  opt.pci.xList.push_back(opt.pci.SuccessfulPredictions[i].x);
	  opt.pci.fList[productionLayerIndex+1].push_back(opt.pci.SuccessfulPredictions[i].f); 
	}
	opt.pci.NumPoints += opt.pci.SuccessfulPredictions.size();
      }
      endTimer("UpdatePCARBF",opt);
    }
    
    int S = 1; if(opt.ActiveOptimizer==101) S = opt.pso.CurrentSwarmSize;
    opt.pci.successRate.clear(); opt.pci.successRate.resize(S);
    opt.pci.successfulMoveRate.clear(); opt.pci.successfulMoveRate.resize(S);
    opt.pci.successfulJumpRate.clear(); opt.pci.successfulJumpRate.resize(S);
    opt.pci.JumpAt.clear(); opt.pci.JumpAt.resize(S);
    opt.pci.moveCount.clear(); opt.pci.moveCount.resize(S);
    opt.pci.report.clear(); opt.pci.report.resize(S);
    fill(opt.pci.JumpAt.begin(),opt.pci.JumpAt.end(),0);
    fill(opt.pci.MoveAcceptanceRate.begin(),opt.pci.MoveAcceptanceRate.end(),0.);
    fill(opt.pci.moveCount.begin(),opt.pci.moveCount.end(),0);    
    fill(opt.pci.successRate.begin(),opt.pci.successRate.end(),0.);  
    fill(opt.pci.successfulMoveRate.begin(),opt.pci.successfulMoveRate.end(),0.);  
    fill(opt.pci.successfulJumpRate.begin(),opt.pci.successfulJumpRate.end(),0.);
  }
  
}

void BuildRBFinterpolant(vector<vector<double>> &ypList, vector<double> &fxList, OPTstruct &opt){
  //RBFinterpolant is not thread-safe -> distribute copies of 'model' to threads

  //create interpolant
  rbfmodel model;
  rbfcreate(opt.pci.dim, 1, model);//opt.pci.funcDIM==1
  int productionLayerIndex = opt.pci.RBFinterpolant.size();
  OPTprint("BuildRBFinterpolant (productionLayerIndex = " + to_string(productionLayerIndex) +  "):",opt);
  
  //fetch unscaled raw data
  int NumVecs = ypList.size();
  vector<vector<double>> ypCoor(opt.pci.dim);
  for(int d=0;d<opt.pci.dim;d++) ypCoor[d].resize(NumVecs);
  vector<vector<double>> rawdataset(NumVecs);
  #pragma omp parallel for schedule(static) if(opt.threads>1)
  for(int i=0;i<NumVecs;i++){
    rawdataset[i].resize(opt.pci.dim+1);
    for(int d=0;d<opt.pci.dim;d++){
      rawdataset[i][d] = ypList[i][d];
      ypCoor[d][i] = ypList[i][d];
    }
    rawdataset[i][opt.pci.dim] = fxList[i];
  }
  string STR = "";
  while(NumVecs>99999){
    rawdataset.erase(rawdataset.begin()+alea_integer(0,NumVecs-1,opt));
    NumVecs = rawdataset.size();
    STR = " ... and shrunk to below 100000 data points";
  }
  OPTprint("                     raw data fetched" + STR,opt);
  
  //remove coordinate duplicates
  vector<vector<double>> cleandataset(0);
  vector<vector<double>> cleanxList(0);
  vector<double> cleanfList(0);
  for(auto i: sort_indices(ypCoor[0])){
    vector<double> pt(0); for(int d=0;d<=opt.pci.dim;d++) pt.push_back(rawdataset[i][d]);
    bool AlreadyAddedQ = false;
    if(cleandataset.size()>0){
      int previousIndex = cleandataset.size()-1;
      while(ABS(ypCoor[0][i]-cleandataset[previousIndex][0])<1.0e-12){
	int sameQ = 0;
	for(int d=1;d<opt.pci.dim;d++) if(ABS(pt[d]-cleandataset[previousIndex][d])<1.0e-12) sameQ++;
	if(sameQ==opt.pci.dim-1) AlreadyAddedQ = true;
	if(previousIndex==0 || AlreadyAddedQ) break;
	else previousIndex--;
      }
    }
    if(!AlreadyAddedQ){
      if(opt.pci.exportQ){
	cleanxList.push_back(opt.pci.xList[i]);//unscaled Coordinates
	cleanfList.push_back(fxList[i]);//unscaled function values
      }
      cleandataset.push_back(pt);//unscaled dataset pt={yp,fx}
    }
  }
  int NewNumVecs = cleandataset.size();
  OPTprint("                     duplicates removed: NumVecs " + to_string(NumVecs) + " -> " + to_string(NewNumVecs),opt);
  
  //set data for ALGLIB
  real_2d_array dataForALGLIB; dataForALGLIB.setlength(NewNumVecs,opt.pci.dim+1);
  vector<vector<double>> cleanypCoor(opt.pci.dim); for(int d=0;d<opt.pci.dim;d++) cleanypCoor[d].resize(NewNumVecs);
  #pragma omp parallel for schedule(static) if(opt.threads>1)
  for(int i=0;i<NewNumVecs;i++) for(int d=0;d<opt.pci.dim;d++) cleanypCoor[d][i] = cleandataset[i][d];
  vector<vector<double>> layerhyperbox(opt.pci.dim);
  vector<double> ypVariances(opt.pci.dim);
  for(int d=0;d<opt.pci.dim;d++){
    layerhyperbox[d].resize(2);
    layerhyperbox[d][0] = *min_element(cleanypCoor[d].begin(),cleanypCoor[d].end());
    layerhyperbox[d][1] = *max_element(cleanypCoor[d].begin(),cleanypCoor[d].end());
    ypVariances[d] = VecVariance(cleanypCoor[d]);
  }
  opt.pci.hyperbox.push_back(layerhyperbox);
  vector<double> extentVec(opt.pci.dim);
  for(int d=0;d<opt.pci.dim;d++) extentVec[d] = opt.pci.hyperbox[productionLayerIndex][d][1] - opt.pci.hyperbox[productionLayerIndex][d][0];
  opt.pci.ExtentVec.push_back(extentVec);
  double normalizeExtentVec = 1./(*max_element(opt.pci.ExtentVec[productionLayerIndex].begin(),opt.pci.ExtentVec[productionLayerIndex].end()));
  for(int d=0;d<opt.pci.dim;d++) opt.pci.ExtentVec[productionLayerIndex][d] *= normalizeExtentVec;
  double fshift = opt.pci.fshift;//*min_element(cleanfList.begin(),cleanfList.end());
  opt.pci.fShift.push_back(fshift); 
  double fspread = *max_element(cleanfList.begin(),cleanfList.end()) - fshift;
  opt.pci.fSpread.push_back(fspread);   
  #pragma omp parallel for schedule(static) if(opt.threads>1)
  for(int i=0;i<NewNumVecs;i++){
    for(int d=0;d<opt.pci.dim;d++) cleandataset[i][d] /= opt.pci.ExtentVec[productionLayerIndex][d];//all PCA coordinates rescaled to UnitInterval
    cleandataset[i][opt.pci.dim] -= opt.pci.fshift;//function values shifted such that their minimum is zero 
    for(int d=0;d<=opt.pci.dim;d++) dataForALGLIB[i][d] = cleandataset[i][d];
  }
  
  //determine main RBF parameters
  double UnitInterval = 1., averageDistance = UnitInterval/pow((double)NewNumVecs,1./((double)opt.pci.dim));
  double baseRadius = opt.pci.RBFbaseRadiusPrefactor*averageDistance;//opt.pci.RBFbaseRadiusPrefactor * (*max_element(opt.pci.InverseExtentVec.begin(),opt.pci.InverseExtentVec.end()));
  int NumLayers = opt.pci.RBFnumLayersPrefactor*(int)(log(baseRadius/1.0e-12)/log(2.)+1.); 
  OPTprint("                     averageDistance = " + to_string_with_precision(averageDistance,12) + ", baseRadius = " + to_string_with_precision(baseRadius,12) + ", NumInternalRBFLayers = " + to_string(NumLayers) + ", NumNodes = " + to_string(NewNumVecs),opt); 
  
  //set nodes
  rbfsetpoints(model, dataForALGLIB);
  OPTprint("                     nodes and function values set",opt);
  OPTprint("                     yp-variances      = " + vec_to_str_with_precision(ypVariances,8),opt);
  OPTprint("                     PCA-ExtentVec     = " + vec_to_str_with_precision(opt.pci.ExtentVec[productionLayerIndex],8),opt);
  OPTprint("                     fShift            = " + to_string_with_precision(opt.pci.fShift[opt.pci.fShift.size()-1],16),opt);
  OPTprint("                     fSpread           = " + to_string_with_precision(opt.pci.fSpread[opt.pci.fSpread.size()-1],16),opt);

  //build model for shifted function values on rescaled PCA coordinates
  rbfreport rep;
  rbfsetalgohierarchical(model, baseRadius, NumLayers, opt.pci.smoothing);
  if(opt.pci.RBFasymptote==0) rbfsetzeroterm(model);
  else if(opt.pci.RBFasymptote==1) rbfsetconstterm(model);
  else if(opt.pci.RBFasymptote==2){ /*alglib-default: linear term*/ }
  rbfbuildmodel(model, rep); 
  if(rep.terminationtype>0) OPTprint("                     model built successfully",opt);
  
  //distribute interpolant copies to threads
  vector<rbfmodel> modelVec(opt.threads);
  #pragma omp parallel for schedule(static) if(opt.threads>1)
  for(int m=0;m<opt.threads;m++) modelVec[m] = model;
  opt.pci.RBFinterpolant.push_back(modelVec);
  OPTprint("                     new interpolants distributed",opt);
        
    
  //optionally, export input data and interpolation (of first two dimensions) to file
  if(opt.pci.exportQ){
    OPTprint("                     export RBF-interpolation data",opt);
    string str;
    double vicinity = 1.0e-1;//vicinity scale for choosing random points
    
    //Check quality of production layer
    str = "mpDPFT_RBF_" + opt.timeStamp + "_CheckInterpolQuality" + to_string(productionLayerIndex) + ".dat";
    ofstream RBF_CheckInterpolQuality; RBF_CheckInterpolQuality.open(str);
    RBF_CheckInterpolQuality << std::setprecision(16);
    vector<vector<double>> PCAcoor(NewNumVecs);
    vector<double> funcVal(NewNumVecs);
    vector<double> interpolVal(NewNumVecs);
    //fetch unscaled Coordinates
    vector<vector<double>> xCoor(opt.D); for(int d=0;d<opt.D;d++) xCoor[d].resize(NewNumVecs);
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for(int i=0;i<NewNumVecs;i++) for(int d=0;d<opt.D;d++) xCoor[d][i] = cleanxList[i][d];//unscaled Coordinates
    vector<double> cleanxListMinVec(opt.D);
    vector<double> cleanxListMaxVec(opt.D);
    vector<double> cleanxListExtentVec(opt.D);
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for(int d=0;d<opt.D;d++){//get HyperBox that contains cleanxList
      cleanxListMinVec[d] = *min_element(xCoor[d].begin(),xCoor[d].end());//minimum of unscaled Coordinates
      cleanxListMaxVec[d] = *max_element(xCoor[d].begin(),xCoor[d].end());//maximum of unscaled Coordinates
      cleanxListExtentVec[d] = cleanxListMaxVec[d] - cleanxListMinVec[d]; // extent of unscaled Coordinates
    }
    //check interpol vs function @testCoordinates in vicinity of nodes
    vector<double> tmpVec1(opt.D), tmpVec2(opt.D);
    if(opt.ActiveOptimizer==101){ tmpVec1 = opt.pso.X[0].x; tmpVec2 = opt.pso.V[0].v; }
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for(int i=0;i<NewNumVecs;i++){
      vector<double> Coor(opt.D);
      PCAcoor[i].resize(opt.pci.dim);
      for(int d=0;d<opt.D;d++) Coor[d] = cleanxList[i][d] + alea(-vicinity,vicinity,opt)*cleanxListExtentVec[d];//unscaled Coordinates
      if(opt.ActiveOptimizer==101){ opt.pso.X[0].x = Coor; ConstrainPSO(0,opt); Coor = opt.pso.X[0].x; }//constrain random Coor
      for(int d=0;d<opt.pci.dim;d++) PCAcoor[i][d] = ScalarProduct(Coor,opt.pci.W[productionLayerIndex][d])/opt.pci.ExtentVec[productionLayerIndex][d];//PCA coordinates rescaled into UnitInterval
      funcVal[i] = GetFuncVal( 0, Coor, opt.function, opt );//function values
      interpolVal[i] = EvalRBFonLayer(productionLayerIndex,omp_get_thread_num(),Coor,opt);//interpol values    
    }
    if(opt.ActiveOptimizer==101){ opt.pso.X[0].x = tmpVec1; opt.pso.V[0].v = tmpVec2; }
    for(int i=0;i<NewNumVecs;i++){
      for(int d=0;d<opt.pci.dim;d++) RBF_CheckInterpolQuality << PCAcoor[i][d] << " ";
      RBF_CheckInterpolQuality << funcVal[i] << " " << interpolVal[i] << "\n";
    }
    RBF_CheckInterpolQuality.close();     
    
    //export cleandataset and interpol at cleanxList
    str = "mpDPFT_RBF_" + opt.timeStamp + "_dataset" + to_string(productionLayerIndex) + ".dat";
    ofstream RBF_dataset; RBF_dataset.open(str);
    RBF_dataset << std::setprecision(16);
    for(int i=0;i<NewNumVecs;i++){
      for(int d=0;d<opt.pci.dim;d++) RBF_dataset << cleandataset[i][d] << " ";
      RBF_dataset << cleandataset[i][opt.pci.dim]+opt.pci.fshift << " " << EvalRBFonLayer(productionLayerIndex,omp_get_thread_num(),cleanxList[i],opt) << "\n";
    }
    RBF_dataset.close();
    
    //evaluate current cleanxList on previous RBF-Layer
    if(productionLayerIndex>0){
      str = "mpDPFT_RBF_" + opt.timeStamp + "_cleanxListOnPrevInterpol" + to_string(productionLayerIndex) + ".dat";
      ofstream RBF_cleanxListOnPrevInterpol; RBF_cleanxListOnPrevInterpol.open(str);
      RBF_cleanxListOnPrevInterpol << std::setprecision(16);
      vector<double> InterpolVal(NewNumVecs);
      int prevLayerIndex = productionLayerIndex-1;
      #pragma omp parallel for schedule(static) if(opt.threads>1)
      for(int i=0;i<NewNumVecs;i++) InterpolVal[i] = EvalRBFonLayer(prevLayerIndex,omp_get_thread_num(),cleanxList[i],opt);
      for(int i=0;i<NewNumVecs;i++){
	for(int d=0;d<opt.pci.dim;d++) RBF_cleanxListOnPrevInterpol << cleandataset[i][d] << " ";
	RBF_cleanxListOnPrevInterpol << cleandataset[i][opt.pci.dim]+opt.pci.fshift << " " << InterpolVal[i] << "\n";
      }
      RBF_cleanxListOnPrevInterpol.close();
    }    

    //store function values and interpolation for random points drawn from the HyperBox that contains cleanxList
    str = "mpDPFT_RBF_" + opt.timeStamp + "_interpol" + to_string(productionLayerIndex) + ".dat";
    ofstream RBF_interpol; RBF_interpol.open(str);
    RBF_interpol << std::setprecision(16);
    int NumHyperboxPoints = 30000;
    vector<double> x(NumHyperboxPoints);
    vector<double> y(NumHyperboxPoints);
    vector<double> FuncVal(NumHyperboxPoints);
    vector<double> InterpolVal(NumHyperboxPoints);
    if(opt.ActiveOptimizer==101){ tmpVec1 = opt.pso.X[0].x; tmpVec2 = opt.pso.V[0].v; }
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for(int i=0;i<NumHyperboxPoints;i++){
      vector<double> Coor(opt.D);
      //option1: unscaled random Coordinates from within HyperBox
      //for(int d=0;d<opt.D;d++) Coor[d] = cleanxListMinVec[d]+alea(0.,1.,opt)*cleanxListExtentVec[d];
      //option2: random displacements from cleanxList
      for(int d=0;d<opt.D;d++) Coor[d] = cleanxList[alea_integer(0,cleanxList.size()-1,opt)][d] + alea(-vicinity,vicinity,opt)*cleanxListExtentVec[d];
      if(opt.ActiveOptimizer==101){ opt.pso.X[0].x = Coor; ConstrainPSO(0,opt); Coor = opt.pso.X[0].x; }//constrain Coor
      x[i] = ScalarProduct(Coor,opt.pci.W[productionLayerIndex][0]);///opt.pci.ExtentVec[productionLayerIndex][0];
      y[i] = ScalarProduct(Coor,opt.pci.W[productionLayerIndex][1]);///opt.pci.ExtentVec[productionLayerIndex][1];
      FuncVal[i] = GetFuncVal( 0, Coor, opt.function, opt );//function values
      InterpolVal[i] = EvalRBFonLayer(productionLayerIndex,omp_get_thread_num(),Coor,opt);//interpol values	
    }
    if(opt.ActiveOptimizer==101){ opt.pso.X[0].x = tmpVec1; opt.pso.V[0].v = tmpVec2; }
    for(int i=0;i<NumHyperboxPoints;i++) RBF_interpol << x[i] << " " << y[i] << " " << FuncVal[i] << " " << InterpolVal[i] << "\n";
    RBF_interpol.close();
    OPTprint("                     interpolation exported to file " + str,opt);
  }
}

double EvalRBFonLayer(int layerIndex, int thread, vector<double> &xp, OPTstruct &opt){
  double F;
  vector<double> coor(opt.pci.dim);
  for(int d=0;d<opt.pci.dim;d++) coor[d] = ScalarProduct(xp,opt.pci.W[layerIndex][d])/opt.pci.ExtentVec[layerIndex][d];
  if(opt.pci.dim==2) F = rbfcalc2(opt.pci.RBFinterpolant[layerIndex][thread], coor[0], coor[1]);
  else if(opt.pci.dim==3) F = rbfcalc3(opt.pci.RBFinterpolant[layerIndex][thread], coor[0], coor[1], coor[2]);
  else{
    real_1d_array tmpF; tmpF.setlength(1);
    rbfcalc(opt.pci.RBFinterpolant[layerIndex][thread], VecTOreal_1d_array(coor), tmpF);
    F = tmpF[0];
  }
  return F + opt.pci.fshift;
}

void PreparePCARBF(OPTstruct &opt){
  int productionLayerIndex = opt.pci.RBFinterpolant.size();
  
  if(productionLayerIndex>0){
    vector<double> F(opt.pci.fList[productionLayerIndex].size());
    
    //for all current x, store interpolations F(x), based on latest RBF interpolant:
    #pragma omp parallel for schedule(static) if(opt.threads>1)
    for(int i=0;i<opt.pci.xList.size();i++) F[i] = EvalRBFonLayer(productionLayerIndex-1, omp_get_thread_num(), opt.pci.xList[i], opt);
    OPTprint("PreparePCARBF:",opt);
    
    //compare current fList with F(x)
    vector<double> AbsDiff = VecAbsDiff(F,opt.pci.fList[productionLayerIndex]);
    vector<double> RelDiff = VecRelDiff(F,opt.pci.fList[productionLayerIndex]);
    double MeanAbsDiff = VecAv(AbsDiff);
    double MeanRelDiff = VecAv(RelDiff);
    OPTprint(" MeanAbsDiff = " + to_string_with_precision(MeanAbsDiff,12) + " --- StdDevAbsDiff = " + to_string_with_precision(sqrt(VecVariance(AbsDiff)),12) + " --- Meanf = " + to_string_with_precision(VecAv(opt.pci.fList[productionLayerIndex]),12) + " --- MeanInterpol = " + to_string_with_precision(VecAv(F),12) + " --- MeanRelDiff = " + to_string_with_precision(MeanRelDiff,12),opt);
    string str = "";
    if(MeanRelDiff<opt.pci.RelDiffCriterion){ str = " interpolation activated"; opt.pci.active = true; }
    else{ str = " interpolation deactivated"; opt.pci.active = false; }
    
    //store data points of the current xList that are (very) accurately predicted by the latest RBF interpolant, i.e., the layer preceeding the productionLayer
    if(opt.pci.UseSuccessfulPredictions){
      opt.pci.SuccessfulPredictions.clear(); opt.pci.SuccessfulPredictions.resize(0);
      for(int i=0;i<RelDiff.size();i++){
	if(RelDiff[i]<0.1*/*opt.pci.RelDiffCriterion*/MeanRelDiff){
	  position pt; pt.x = opt.pci.xList[i]; pt.f = opt.pci.fList[productionLayerIndex][i];
	  opt.pci.SuccessfulPredictions.push_back(pt);
	}
      }
      OPTprint(str + "\n" + " #SuccessfulPredictions(of current xList by preceeding interpolant) = " + to_string(opt.pci.SuccessfulPredictions.size()) + "/" + to_string(opt.pci.xList.size()) + " [to be used for productionLayerIndex " + to_string(productionLayerIndex+1) + "] \n",opt);
    }
    else OPTprint(str + "\n",opt);
  }
  
}

bool AcceptMoveQ(int thread, int s, vector<double> &xp, vector<double> &vp, double fbenchmark, OPTstruct &opt){//called from within parallel region
  if(opt.pci.active){
    int NumExistingLayers = opt.pci.RBFinterpolant.size();
    if(opt.pci.moveCount[s]==opt.pci.MaxMoveCount){
      //select best encounter of unsuccessful trajectory
      if(NumExistingLayers>0){
	for(auto i: sort_indices(opt.pci.report[s].F)){
	  if(opt.ActiveOptimizer==101){
	    opt.pso.X[s].x = opt.pci.report[s].xVec[i];
	    opt.pso.V[s].v = opt.pci.report[s].vVec[i];
	  }
	  break;
	}
      }  
      return true;
    }
    else if(NumExistingLayers>0){
      int layerIndex = NumExistingLayers-1;//default, most recent layer
      int WithinHyperBoxCount=0;
      for(int l=NumExistingLayers-1;l>0;l--){//search for RBF-layer that best contains xp in PCA space, starting from latest RBFinterpolant
	vector<double> coor(opt.pci.dim);
	int coorWithinHyperBox = 0;
	for(int d=0;d<opt.pci.dim;d++){
	  coor[d] = ScalarProduct(xp,opt.pci.W[l][d])/opt.pci.ExtentVec[l][d];
	  if(coor[d]>opt.pci.hyperbox[l][d][0] && coor[d]<opt.pci.hyperbox[l][d][1]) coorWithinHyperBox++;
	}
	if(coorWithinHyperBox==opt.pci.dim){ layerIndex = l; break; }//all coordinates of coor are contained in hyperbox of layer l
	else if(coorWithinHyperBox>WithinHyperBoxCount){ WithinHyperBoxCount = coorWithinHyperBox; layerIndex = l; }//a maximum of coordinates of (PCA-)coor are contained in hyperbox of layer l
      }
      double F = EvalRBFonLayer(layerIndex, omp_get_thread_num(), xp, opt);
      opt.pci.report[s].F.push_back(F);
      opt.pci.report[s].xVec.push_back(xp);
      opt.pci.report[s].vVec.push_back(vp);
      if(F<fbenchmark-opt.pci.SuccessfulMoveCriterion){
	double f = GetFuncVal(s,xp,opt.function,opt);
	if(f<F) cout << "AcceptMoveQ: Good Move in layer " << layerIndex << " --- func = " << to_string_with_precision(f,16) << " interpol = " << to_string_with_precision(F,16) << " --- RelDiff = " << RelDiff(f,F) << endl;
	opt.pci.MoveAcceptanceRate[thread] += 1.;
	return true;
      }
      else return false;
    }
    else return true;
  } 
  else return true; 
}

void AdaptPCARBF(OPTstruct &opt){
  if(opt.pci.active && opt.pci.RBFinterpolant.size()>0){
    double AcceptanceRateThreshold = 0.8, SuccessRateThreshold = 0.1;
    
    //collect info
    int S = 1; 
    if(opt.ActiveOptimizer==101){ S = opt.pso.CurrentSwarmSize; SuccessRateThreshold = 1./((double)S); }
    for(int s=0;s<S;s++){
      double testf; if(opt.ActiveOptimizer==101) testf = opt.pso.P[s].f;
      if(opt.pci.moveCount[s]<opt.pci.MaxMoveCount && opt.pso.X[s].f<(1.-opt.pci.SuccessfulMoveCriterion)*testf){
	opt.pci.successRate[s] = 1.;//if significant improvement (w.r.t. SuccessfulMoveCriterion) before MaxMoveCount is reached
	if(opt.pci.moveCount[s]>1){
	  if(opt.pci.JumpAt[s]==opt.pci.moveCount[s]) opt.pci.successfulJumpRate[s] = 1.;//if an PCARBF-informed jump yielded that improvement
	  else opt.pci.successfulMoveRate[s] = 1.;//if the PCARBF-informed move yielded that improvement
	}
      }
    }
    opt.pci.SuccessRate = VecAv(opt.pci.successRate);//fraction of significant improvements (w.r.t. SuccessfulMoveCriterion) among all S
    opt.pci.SuccessfulMoveRate = VecAv(opt.pci.successfulMoveRate);//for monitoring
    opt.pci.SuccessfulJumpRate = VecAv(opt.pci.successfulJumpRate);//for monitoring
    int TotalNumberOfMoves = accumulate(opt.pci.moveCount.begin(),opt.pci.moveCount.end(),0);
    opt.pci.AverageMoveCount = (int)((double)TotalNumberOfMoves/((double)S));      
    opt.pci.AcceptanceRate = accumulate(opt.pci.MoveAcceptanceRate.begin(),opt.pci.MoveAcceptanceRate.end(),0.0) / ((double)S);
    
    if(opt.pci.SuccessRate<SuccessRateThreshold){//not a single improvement was found ...
      opt.pci.MovePrefactor *= 1.1;
      
//       if(opt.pci.AcceptanceRate>AcceptanceRateThreshold){// ... at high acceptance rate because ...
// 	//(I) too many bad moves are accepted (because SuccessfulMoveCriterion is too small) -> implicitly increase AverageMoveCount
// 	opt.pci.SuccessfulMoveCriterion *= 2.;
// 	//(II) too many bad points exist in the neighborhood (possibly due to wide-range random search -> refocus on best positions)
// 	opt.pci.MovePrefactor /= 1.1; 
//       }
//       else if(opt.pci.AcceptanceRate<1.-AcceptanceRateThreshold){// ... at low acceptance rate because ...
// 	//(I) too few good proposals are accepted ...
// 	//(i) because acceptance probability is too low (accompanied by maximum number of moves for finding good proposals ->increase acceptance probability if AverageMoveCount is large)
// 	opt.pci.SuccessfulMoveCriterion *= 0.5;
// 	//(ii) because too many proposals (which are accepted) are actually bad proposals (-> improve PCA)
// 	//if(!opt.pci.NumPointsUpdatedQ && opt.pci.NumPoints<45000){ opt.pci.NumPoints *= 2; opt.pci.NumPointsUpdatedQ = true; }
// 	//(II) because too few better points exist in the neighborhood (possibly due to local minimum -> move out more aggressively)
// 	opt.pci.MovePrefactor *= 1.1;
//       }
    }
    else opt.pci.MovePrefactor /= 1.1;
      
    if(opt.pci.MovePrefactor<=opt.pci.MovePrefactorMin) opt.pci.MovePrefactor = opt.pci.MovePrefactorMin; else opt.pci.MovePrefactor = opt.pci.MovePrefactorMax;
    if(opt.pci.SuccessfulMoveCriterion>=1.0e-1) opt.pci.SuccessfulMoveCriterion = 1.0e-1; else if(opt.pci.SuccessfulMoveCriterion<1.0e-16) opt.pci.SuccessfulMoveCriterion = 1.0e-16;      

    OPTprint("SuccessRate             = " + to_string(opt.pci.SuccessRate) + " [SuccessRateThreshold = " + to_string(SuccessRateThreshold) + "]",opt);
    OPTprint("SuccessfulMoveRate      = " + to_string(opt.pci.SuccessfulMoveRate),opt);
    OPTprint("SuccessfulJumpRate      = " + to_string(opt.pci.SuccessfulJumpRate),opt);	
    OPTprint("AcceptanceRate          = " + to_string(opt.pci.AcceptanceRate) + " [AcceptanceRateThreshold = " + to_string(AcceptanceRateThreshold) + "]",opt);
    OPTprint("AverageMoveCount        = " + to_string(opt.pci.AverageMoveCount) + "/" + to_string(opt.pci.MaxMoveCount),opt);
    OPTprint("SuccessfulMoveCriterion = " + to_string_with_precision(opt.pci.SuccessfulMoveCriterion,16),opt);
    OPTprint("opt.pci.MovePrefactor   = " + to_string_with_precision(opt.pci.MovePrefactor,16),opt);      
  }
}

void SetDefaultPCIparams(OPTstruct &opt){
  opt.pci.enabled = true;
  opt.pci.exportQ = true;
  opt.pci.UseSuccessfulPredictions = false;
  opt.pci.dim = /*opt.ex.settings[14];*//*1+(int)pow(opt.D,0.37);*/2+(int)((double)(6*opt.D)/200.);
  opt.pci.RelDiffCriterion = 0.02;
  opt.pci.eps = 1.0e-12;
  opt.pci.SuccessfulMoveCriterion = 1.0e-4;
  opt.pci.NumPoints = min(99999,/*500**//*100**/25*opt.D);
  opt.pci.MaxMoveCount = 10*opt.D;
  opt.pci.MovePrefactorMin = 1.0e-8;
  opt.pci.MovePrefactorMax = 0.1;//1./((double)opt.pci.MaxMoveCount);
  opt.pci.MovePrefactor = opt.pci.MovePrefactorMin;
}

void SetDefaultCGDparams(OPTstruct &opt){
  opt.ActiveOptimizer = 100;
  SetDefaultSearchSpace(opt);
  opt.cgd.maxits = 0;
  opt.cgd.scale = 1.;
  opt.cgd.epsf = opt.epsf;
  opt.cgd.epsg = 0;
  opt.cgd.epsx = 0;
  opt.cgd.diffstep = 1.0e-1;
}



void SetDefaultCSAparams(OPTstruct &opt){
  opt.ActiveOptimizer = 105;
  SetDefaultSearchSpace(opt);
  opt.csa.epsf = opt.epsf;
  opt.csa.max_iterations = (int)1.0e+9;//The maximum number of iterations/steps.
  opt.csa.max_funcEvals = (double)opt.D*1.0e+5;//The maximum number of evaluations of the objective function
  opt.csa.tgen_initial = 1.;/*0.01;*/// The initial value of the generation temperature.
  opt.csa.tgen_schedule = 0.999;/*0.99999;*/// Determines the factor that `tgen` is multiplied by during each update.
  opt.csa.LargeStepProb = 0.05;/*0.1;*//*1./((double)opt.D);*/ //probability of making a large step (up to tgen_initial) in each of the optimization variables, set to zero for original cSA
  opt.csa.minimal_step = 1.0e-6;//if LargeStepProb>0, take at least a step of size minimal_step in void step()
  opt.csa.maxStallCount = 100;//double annealing temperature if Currentbestf is not improved after maxStallCount trials of moving uphill (in the respective thread)
  opt.csa.AcceptProb = -0.1;/*-0.1;*///constant acceptance probability (corresponding to opt.threads uncoupled annealers in parallel with same starting variables; disabled if negative)
  opt.csa.minimal_AcceptProb = 1.0e-4;//lower limit probabilty for accepting uphill move; default: 0
  opt.csa.tacc_initial = 0.9;/*0.9;*/// The initial value of the acceptance temperature.
  opt.csa.tacc_schedule = 1.0e-4;/*0.01;*/// Determines the percentage by which `tacc` is increased or decreased during each update.
  double m = (double)opt.threads; opt.csa.desired_variance = 0.99*(m-1.)/(m*m);/*0.99;*/// The desired variance of the acceptance probabilities.  
  opt.csa.X.resize(opt.threads); opt.csa.nb_eval.resize(opt.threads);
  opt.currentBestx.resize(opt.D); 
  for(int t=0;t<opt.threads;t++){ opt.csa.X[t].x.resize(opt.D); opt.csa.nb_eval[t] = 0; }
  opt.csa.VarianceCheck = opt.D;
  opt.csa.averageStepSize.resize(opt.threads);
}

void SetDefaultPSOparams(OPTstruct &opt){
  opt.ActiveOptimizer = 101;
  if(opt.threads>1 && opt.function+2014>=1 && opt.function+2014<=30){ opt.threads = 1; cout << "opt.threads -> 1 !!!!!!!!!!!!!!!!!!!!!" << endl; }
  SetDefaultSearchSpace(opt);
  opt.pso.Smax = 100000;
  opt.pso.epsf = opt.epsf;
  opt.pso.runs = 1;
  opt.pso.elitism = 0.;
  opt.pso.avPartQ = 0;
  opt.pso.reseed = 0.;  
  opt.pso.reseedFactor = 0.1;
  opt.pso.InitialSwarmSize = 40;//10+(int)(2.*sqrt((double)opt.D));
  InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
  opt.pso.InitialSwarmSizeCurrentRun = opt.pso.InitialSwarmSize;
  opt.pso.loopMax = 2000*opt.D;
  opt.pso.reseedEvery = max(1,(int)(0.25*opt.pso.reseedFactor*(double)opt.pso.loopMax));
  opt.pso.increase = 1.;
  opt.pso.VarianceCheck = opt.D;/*10;*///number of PSO loops to consider for variance check
  opt.pso.TargetMinEncounters = 10;//2;
  opt.pso.PostProcessQ = 0;
  opt.pso.AlwaysUpdateSearchSpace = false;
}

void InitializeSwarmSizeDependentVariables(int &S, OPTstruct &opt){
  	if(opt.threads>1){
    	if(S<=opt.threads) S = opt.threads;
    	else if(S%opt.threads>0) S = max(opt.threads,S-(S%opt.threads)+opt.threads);
  	}
  	opt.pso.CurrentSwarmSize = S;
  	opt.pso.MaxLinks = 1+(int)((double)S/4.);
  	opt.pso.eval_max_init = (int)2.0e+9;
  	opt.pso.bestToKeep = (int)(0.2*(double)S);

}

void ReinitializePSOparams(int s, OPTstruct &opt){
  if( opt.pso.CoefficientDistribution==1 || (opt.pso.CoefficientDistribution==4 && opt.localrun<opt.pso.MinimumRandomInitializations) ){//random initializations
    opt.pso.W[s][1] = alea(opt.pso.ab[1],opt.pso.ab[2],opt);//start
    opt.pso.C1[s][1] = alea(opt.pso.ab[3],opt.pso.ab[4],opt);
    opt.pso.C2[s][1] = alea(opt.pso.ab[5],opt.pso.ab[6],opt);
    opt.pso.W[s][0] = opt.pso.W[s][1];//current
    opt.pso.C1[s][0] = opt.pso.C1[s][1];
    opt.pso.C2[s][0] = opt.pso.C2[s][1];
    opt.pso.W[s][2] = alea(opt.pso.ab[1],opt.pso.ab[2],opt);//target
    opt.pso.C1[s][2] = alea(opt.pso.ab[3],opt.pso.ab[4],opt);
    opt.pso.C2[s][2] = alea(opt.pso.ab[5],opt.pso.ab[6],opt);
    opt.pso.W[s][3] = alea(0.,opt.pso.ab[0],opt);//rate alpha
    opt.pso.C1[s][3] = alea(0.,opt.pso.ab[0],opt);
    opt.pso.C2[s][3] = alea(0.,opt.pso.ab[0],opt);
  }
  else if(opt.pso.CoefficientDistribution==4){
    int H = opt.pso.SortedHistory.size();
    //cross-breed with higher probability those PSO-parameters that yielded better function values
    int h1 = (int)((double)(H-1)*exp(-alea(0.,(double)H,opt)/(0.3*(double)H)));
    int h2 = (int)((double)(H-1)*exp(-alea(0.,(double)H,opt)/(0.3*(double)H)));
    //int h1 = alea_integer(0,(int)(0.1*(double)(H-1)),opt);
    //int h2 = alea_integer(0,(int)(0.1*(double)(H-1)),opt);
    opt.pso.W[s][1] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][0][1]+opt.pso.SortedPSOparamsHistory[h2][0][1]);//start
    opt.pso.C1[s][1] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][1][1]+opt.pso.SortedPSOparamsHistory[h2][1][1]);
    opt.pso.C2[s][1] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][2][1]+opt.pso.SortedPSOparamsHistory[h2][2][1]);
    opt.pso.W[s][0] = opt.pso.W[s][1];//current
    opt.pso.C1[s][0] = opt.pso.C1[s][1];
    opt.pso.C2[s][0] = opt.pso.C2[s][1];
    opt.pso.W[s][2] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][0][2]+opt.pso.SortedPSOparamsHistory[h2][0][2]);//target
    opt.pso.C1[s][2] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][1][2]+opt.pso.SortedPSOparamsHistory[h2][1][2]);
    opt.pso.C2[s][2] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][2][2]+opt.pso.SortedPSOparamsHistory[h2][2][2]);
    opt.pso.W[s][3] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][0][3]+opt.pso.SortedPSOparamsHistory[h2][0][3]);//rate alpha
    opt.pso.C1[s][3] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][1][3]+opt.pso.SortedPSOparamsHistory[h2][1][3]);
    opt.pso.C2[s][3] = 0.5*(opt.pso.SortedPSOparamsHistory[h1][2][3]+opt.pso.SortedPSOparamsHistory[h2][2][3]);
  }
  else{
    opt.pso.W[s][0] = opt.pso.w;
    opt.pso.C1[s][0] = opt.pso.c;
    opt.pso.C2[s][0] = opt.pso.c;    
  }
}

void UpdatePSOparams(OPTstruct &opt){
  if(opt.pso.CoefficientDistribution==1 || opt.pso.CoefficientDistribution==4){
    for(int s=0;s<opt.pso.CurrentSwarmSize;s++){
      opt.pso.W[s][0] = max(opt.pso.ab[1],min(opt.pso.ab[2], opt.pso.W[s][2] + (opt.pso.W[s][1] - opt.pso.W[s][2])*EXP(-tan(opt.pso.W[s][3])*(double)opt.loopCount/((double)opt.pso.loopMax))));
      opt.pso.C1[s][0] = max(opt.pso.ab[3],min(opt.pso.ab[4], opt.pso.C1[s][2] + (opt.pso.C1[s][1] - opt.pso.C1[s][2])*EXP(-tan(opt.pso.C1[s][3])*(double)opt.loopCount/((double)opt.pso.loopMax))));
      opt.pso.C2[s][0] = max(opt.pso.ab[5],min(opt.pso.ab[6], opt.pso.C2[s][2] + (opt.pso.C2[s][1] - opt.pso.C2[s][2])*EXP(-tan(opt.pso.C2[s][3])*(double)opt.loopCount/((double)opt.pso.loopMax))));
    }
  }
  else if(opt.pso.CoefficientDistribution==5 && opt.pso.UpdateParamsQ){
    double admixture = 0.2, width = 2.*0.05*0.05;
    int activeHistory = 10;/*100;*/
    opt.pso.centralW = (1.-admixture)*opt.pso.centralW+admixture*PartialVecAv(opt.pso.SuccessfulW,activeHistory);
    opt.pso.centralC1 = (1.-admixture)*opt.pso.centralC1+admixture*PartialVecAv(opt.pso.SuccessfulC1,activeHistory);
    opt.pso.centralC2 = (1.-admixture)*opt.pso.centralC2+admixture*PartialVecAv(opt.pso.SuccessfulC2,activeHistory);
    //cout << "   *** UpdatePSOparams(" << opt.pso.SuccessfulW.size() << "): centers = " << opt.pso.centralW << " " << opt.pso.centralC1 << " " << opt.pso.centralC2 << endl;
    for(int s=0;s<opt.pso.CurrentSwarmSize;s++){//draw from Gaussian centered at centralW, etc.
      opt.pso.W[s][0] = max(opt.pso.ab[1],min(opt.pso.ab[2],opt.pso.centralW+randSign(opt)*sqrt(-width*log(alea(1.0e-16,1.,opt)))));
      opt.pso.C1[s][0] = max(opt.pso.ab[3],min(opt.pso.ab[4],opt.pso.centralC1+randSign(opt)*sqrt(-width*log(alea(1.0e-16,1.,opt)))));
      opt.pso.C2[s][0] = max(opt.pso.ab[5],min(opt.pso.ab[6],opt.pso.centralC2+randSign(opt)*sqrt(-width*log(alea(1.0e-16,1.,opt)))));
    }
  }
}

void SetDefaultLCOparams(OPTstruct &opt){
  opt.ActiveOptimizer = 102; 
  SetDefaultSearchSpace(opt);
  opt.lco.runs = 1000;
  opt.lco.VarianceCheck = 10;
  opt.lco.VarianceThreshold = 1.0e-6;
  opt.lco.epsg = 0;
  opt.lco.maxits = 0;
  opt.lco.epsf = opt.epsf;
  opt.lco.epsx = 0;
  opt.lco.diffstep = 1.0e-1;
}

void SetLCOvariables(OPTstruct &opt){
  opt.lco.x.setlength(opt.D);
  opt.lco.x.setcontent(opt.D,&(opt.lco.InitParam[0]));//initial values of variables
  opt.lco.bndl.setlength(opt.D); for(int i=0;i<opt.D;i++) opt.lco.bndl[i] = opt.SearchSpace[i][0];//-INF for unbounded
  opt.lco.bndu.setlength(opt.D); for(int i=0;i<opt.D;i++) opt.lco.bndu[i] = opt.SearchSpace[i][1];//+INF for unbounded
  opt.lco.s.setlength(opt.D); for(int i=0;i<opt.D;i++) opt.lco.s[i] = opt.lco.bndu[i]-opt.lco.bndl[i];//scales
  cout << "SetLCOvariables: " << vec_to_str(real_1d_arrayToVec(opt.lco.s)) << endl;
  if(opt.function == -1 || opt.function == -2 || opt.function == -11){
      opt.lco.ct.setlength(1); opt.lco.ct[0] = 0;//constraint types
      opt.lco.c.setlength(1,opt.D+1);//constraints:
      for(int i=0;i<opt.D;i++){
	opt.lco.c[0][i] = 1.;
	if(opt.ex.settings[8]==1 && i>=opt.ex.L) opt.lco.c[0][i] = 0.;
      }
      opt.lco.c[0][opt.D] = opt.ex.Abundances[0];
    }
}

void setupLCO(int runs, int dim, double acc, OPTstruct &opt){
  int L = opt.ex.L;
  opt.D = dim;
  opt.ex.settings[4] = 0;
  opt.epsf = 0.;//MP;
  opt.printQ = 1;
  opt.SearchSpaceMin = 0.+acc; opt.SearchSpaceMax = 2.-acc;
  if(opt.ex.settings[8]==1){
    opt.function = -2;
    opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
    opt.SearchSpaceLowerVec.resize(opt.D); opt.SearchSpaceUpperVec.resize(opt.D);
    for(int d=0;d<L;d++){
      opt.SearchSpaceLowerVec[d] = 0.+acc;
      opt.SearchSpaceUpperVec[d] = 2.-acc;
      opt.SearchSpaceLowerVec[L+d] = 0.+acc;
      opt.SearchSpaceUpperVec[L+d] = 2.*3.141592653589793-acc;
    }
  }
  SetDefaultLCOparams(opt);
  opt.lco.runs = runs;
}

void reportLCO(OPTstruct &opt){
  int L = opt.ex.L;
  opt.lco.rep.FuncEvals = accumulate(opt.lco.evals.begin(),opt.lco.evals.end(),0);
  if(opt.lco.rep.FuncEvals<0){
    opt.lco.rep.FuncEvals = std::numeric_limits<int>::max();
    OPTprint(" Warning !!! FuncEvals overflow)\n",opt);
  }
  opt.lco.rep.bestf = opt.lco.bestf;
  opt.lco.rep.OccNum.resize(opt.ex.L);
  opt.lco.rep.Phases.resize(opt.ex.L);
  for(int l=0;l<L;l++){
    opt.lco.rep.OccNum[l] = opt.lco.bestx[l];
    if(opt.ex.settings[8]==1 && opt.ex.settings[10]==1) opt.lco.rep.Phases[l] = opt.lco.bestx[L+l];
  }
}

void SetDefaultAULparams(OPTstruct &opt){
  opt.ActiveOptimizer = 103;
  SetDefaultSearchSpace(opt);
  ae_int_t maxits = 0;
  opt.aul.scale = 1.;
  opt.aul.epsf = opt.epsf;
  opt.aul.epsx = 1.0e-6;
  opt.aul.diffstep = 1.0e-1;
}

void LoadOPTparams(OPTstruct &opt){
  opt.reportQOriginal = opt.reportQ;
  
  ifstream infile;
  if(opt.ActiveOptimizer==101){//load PSO parameters from file
    infile.open("TabFunc_PSOparams.dat");
    string line;
    int i = 0;
    double val;
    vector<double> params;
    bool loadQ = false;
  
    while(getline(infile,line)){
      istringstream iss(line);
      if(line != ""){
	iss.str(line);
	iss >> val;
	if(i==0 && (int)val==0) break;
	else if(i>0){
	  loadQ = true;
	  params.push_back(val);
	}
	i++;
      }
    }
    
    if(loadQ){
      opt.pso.Smax = (int)params[0];  	// Smax; // maximum swarm size
      opt.pso.InitialSwarmSize = (int)params[1];	  	// InitialSwarmSize; // Initial number of swarm particles (capped at Smax)  
      opt.pso.MaxLinks = (int)params[2];	  	// MaxLinks; // Max number of particles informed by a given one
      opt.pso.w = params[3];	  	// w; // First confidence coefficient
      opt.pso.c = params[4];	  	// c; // Second confidence coefficient  
      opt.pso.epsf = params[5];	  	// epsf; // //absolute tolerance for convergence of objective function
      opt.pso.VarianceCheck = (int)params[6];	  	// VarianceCheck; //number of PSO loops to take into account for variance check  
      opt.pso.elitism = params[7];	  	// elitism; // elitism from none (0.) to full (1.)
      opt.pso.avPartQ = (int)params[8];	  	// avPartQ; // 1: replace worst by average particle --- 0: don't
      opt.pso.PostProcessQ = (int)params[9];	  	// PostProcessQ; // 1: post-process CurrentBestx with LCO --- 0: don't
      opt.pso.eval_max_init = (int)params[10];	// eval_max_init; // maximum number of evaluations for first run
      opt.pso.increase = params[11];	// increase; // increase swarm size and eval_max in consecutive runs
      opt.pso.TargetMinEncounters = (int)params[12];	// TargetMinEncounters; // minimum number of runs that encounter the current optimum within epsf
      opt.pso.runs = (int)params[13];	// runs; // Numbers of runs
      OPTprint("LoadOPTparams " + vec_to_str(params),opt);
    }
    
  }
  infile.close();    
}

void ShrinkSwarm(OPTstruct &opt){
  	double initS = (double)opt.pso.InitialSwarmSizeCurrentRun, minS = max(10.,(double)opt.threads), c = log10(opt.CurrentVariance), t = log10(opt.VarianceThreshold);
  	int NewSwarmSize = (int)min(initS,max(minS,(double)opt.threads + initS * (1.-opt.pso.SwarmDecayRate*c/t)));
  	if(opt.loopCount%opt.pso.VarianceCheck==0 && NewSwarmSize<=opt.pso.CurrentSwarmSize-opt.threads){
    	//cout << c << " " << t << " " << initS * (1.-opt.pso.SwarmDecayRate*c/t) << " " << NewSwarmSize << endl;
    	vector<double> flist(opt.pso.CurrentSwarmSize); for(int s=0;s<opt.pso.CurrentSwarmSize;s++) flist[s] = opt.pso.P[s].f;//collect personal bests of swarm...
    	opt.pso.NewSwarmSizeQ = true;
    	InitializeSwarmSizeDependentVariables(NewSwarmSize,opt);
		OPTprint("ShrinkSwarm @ PSO-run " + to_string(opt.localrun+1) + " loop " + to_string(opt.loopCount) + ": NewSwarmSize = " + to_string(NewSwarmSize),opt);
    	vector<position> XToKeep(0),XoldToKeep(0),PToKeep(0);
    	vector<velocity> VToKeep(0);
    	vector<vector<double>> WToKeep(0),C1ToKeep(0),C2ToKeep(0);
    	int count = 0;
    	for(auto s: sort_indices(flist)){//...sort them (best first)...
      		if(count<NewSwarmSize){
				XToKeep.push_back(opt.pso.X[s]);
				XoldToKeep.push_back(opt.pso.Xold[s]);
				PToKeep.push_back(opt.pso.P[s]);
				VToKeep.push_back(opt.pso.V[s]);
				WToKeep.push_back(opt.pso.W[s]);
				C1ToKeep.push_back(opt.pso.C1[s]);
				C2ToKeep.push_back(opt.pso.C2[s]);
				opt.maxEC2[count] = opt.maxEC[s];
				opt.maxIC2[count] = opt.maxIC[s];
				if((int)opt.PenaltyMethod[0]>0){
					for(int i=0;i<opt.NumEC+opt.NumIC;i++){
						opt.ALmu2[i][count] = opt.ALmu[i][s];
						opt.ALlambda2[i][count] = opt.ALlambda[i][s];
					}
				}
				count++;
      		}
    	}
    	InitializePSOrun(opt);
    	//cout << NewSwarmSize << " " << VToKeep.size() << " " << opt.pso.V.size() << endl;
    	for(int i=0;i<NewSwarmSize;i++){
       		opt.pso.X[i].x = XToKeep[i].x;
       		opt.pso.X[i].f = XToKeep[i].f;
       		opt.pso.Xold[i].x = XoldToKeep[i].x;
       		opt.pso.Xold[i].f = XoldToKeep[i].f;
       		opt.pso.P[i].x = PToKeep[i].x;
       		opt.pso.P[i].f = PToKeep[i].f;
       		opt.pso.V[i].v = VToKeep[i].v;
       		opt.pso.W[i] = WToKeep[i];
       		opt.pso.C1[i] = C1ToKeep[i];
       		opt.pso.C2[i] = C2ToKeep[i];
    	}
    	opt.maxEC.resize(NewSwarmSize);
		opt.maxEC2.resize(NewSwarmSize);
		opt.maxIC.resize(NewSwarmSize);
		opt.maxIC2.resize(NewSwarmSize);
		if((int)opt.PenaltyMethod[0]>0){
			opt.ALmu.resize(NewSwarmSize);
			opt.ALmu2.resize(NewSwarmSize);
			for(int i=0;i<opt.NumEC+opt.NumIC;i++){
				opt.ALlambda[i].resize(NewSwarmSize);
				opt.ALlambda2[i].resize(NewSwarmSize);
			}

		}
  }
  else opt.pso.NewSwarmSizeQ = false;
}

double Anneal(double f, int s, OPTstruct &opt){
	double spread = 0., t = 1, T = 1;
	vector<double> x(opt.D);
	if(opt.ActiveOptimizer==101 && opt.loopCount>1){
		t = (double)opt.loopCount;
		T = (double)opt.pso.loopMax;
		x = opt.pso.X[s].x;
		if(opt.homotopy>0 || opt.AnnealType==-1) spread = opt.pso.X[opt.pso.worst].f-opt.currentBestf;
		else if(opt.AnnealType==0) spread = opt.SearchSpaceExtent;
	}
	else if(opt.ActiveOptimizer==106 && opt.cma.generation>1){
		t = (double)opt.cma.generation;
		T = (double)opt.cma.generationMax;
		int c = s%opt.cma.populationSize;
		int p = (s-c)/opt.cma.populationSize;
		x = opt.cma.pop[p][c];
		if(opt.homotopy>0 || opt.AnnealType==-1) spread = opt.cma.spread[p];
		else if(opt.AnnealType==0) spread = opt.SearchSpaceExtent;
	}
	double admix = exp(-40.*t/T);
	if(opt.homotopy>0) return -admix*f + admix*spread*Norm2(x)/(opt.D*opt.SearchSpaceExtent*opt.SearchSpaceExtent);
	return opt.anneal*admix*spread;
}

