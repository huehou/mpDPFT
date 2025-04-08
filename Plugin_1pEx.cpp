//1-particle-exact DFT

#include "MPDPFT_HEADER_ONEPEXDFT.h"
#include "Plugin_1pEx_Rho1p.h"
#include "Plugin_1pEx.h"
#include "Plugin_OPT.h"
#include "mpDPFT.h"
#include <cstdio>
#include <cmath>
#include <complex>
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
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_laguerre.h>
//#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/legendre.hpp>
// #include <Eigen/Dense>
// #include <Eigen/QR>
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

using namespace std;
using namespace Eigen;
using namespace std::literals;

vector<double> Hint(0);
vector<vector<vector<vector<double>>>> Hint4DArray(0);
vector<vector<int>> HintIndexVec(0);
double subdivRiemannEst = 10000.;//number of intervals for the inital estimate via Riemann sum
double tailMultiplier = 10.;//how often to increase 'infinity'
double MinDelta = 1.0e-14;//minimal delta x for sudivisions.

double Orbital(int lambda, vector<double> position, exDFTstruct &ex){
  if(ex.System==100){//V_ext == 1D harmonic oscillator
    double x = position[0]/ex.UNITlength;
    double Herm = alglib::hermitecalculate((ae_int_t)lambda, x, alglib::xdefault);
    double AbsHerm = ABS(Herm), SignHerm = Sign(Herm);
    if(AbsHerm>0.) return SignHerm*EXP(0.5*(-x*x-(double)gsl_sf_lnfact((unsigned int)lambda)-(double)lambda*log(2.)+2.*log(AbsHerm)))/pow(3.141592653589793*ex.UNITlength,0.25);
    else if(AbsHerm!=AbsHerm){ cout << "OrbitalSquared error: AbsHerm(l=" << lambda << ",x=" << x << ") = " << AbsHerm << endl; usleep(10*1000000); }
  }
  else if(ex.System==101 || ex.System==102){
	//V_ext == 3D Coulomb-ion of charge Z; be aware: part of dependence on magnetic quantum number m is shifted to void GetDensity(...)
	//System==101: nonrelativistic, System==102: relativistic
	//if(ex.System==102) EXprint(1, "Orbital: Warning!!! relativistic spatial wavefunctions not yet implemented.", ex);
    vector<double> sc = SphericalCoor(position);
    double r = sc[0], theta = sc[1], rho = 2.*ex.params[1]*r;
    vector<int> nlm = ex.QuantumNumbers[lambda];
    int n = nlm[0], l = nlm[1], m = nlm[2];
    double dn = (double)n, dl = (double)l, x = rho/dn;
    double R = sqrt((2.*ex.params[1]/dn)*(2.*ex.params[1]/dn)*(2.*ex.params[1]/dn)*EXP((double)gsl_sf_lnfact((unsigned int)(n-l-1))-(double)gsl_sf_lnfact((unsigned int)(n+l))-log(2.*dn)-x)) * pow(x,dl) * boost::math::laguerre((unsigned)(n-l-1),(unsigned)(2*l+1),x);
    double Ytilde = EXP(0.5*((double)gsl_sf_lnfact((unsigned int)(l-m))+(double)gsl_sf_lnfact((unsigned int)(2*l+1))-(double)gsl_sf_lnfact((unsigned int)(l+m))-log(4.*3.141592653589793))) * boost::math::legendre_p(l,m,cos(theta));
    return R*Ytilde;
  }
  return 0.;
}

void GetQuantumNumbers(exDFTstruct &ex){
  int L = ex.settings[0];
  ex.QuantNumRestrictIndexVectors.clear();
  ex.QuantumNumbers.clear(); ex.QuantumNumbers.resize(L);
  ex.LevelNames.clear(); ex.LevelNames.resize(L);
  if(ex.settings[2]==23 || ex.settings[2]==24){
    for(int lambda=0;lambda<L;lambda++) ex.QuantumNumbers[lambda].resize(3);
    int lambda = 0, n=1;
    while(lambda<L){
      vector<double> ShellIndices(0);
      for(int l=0;l<n;l++){
	vector<double> mIndices(0);
	for(int m=-l;m<=l;m++){
	  if(lambda<L){
	    ex.QuantumNumbers[lambda] = {{n,l,m}};
	    string lstring; if(l==0) lstring = "S"; if(l==1) lstring = "P"; if(l==2) lstring = "D"; if(l==3) lstring = "F"; if(l==4) lstring = "G"; if(l==5) lstring = "H";
	    string mstring = to_string(m); if(m==0){ if(l==0) mstring = "  "; else mstring = " " + to_string(abs(m)); } else if(m>0) mstring = "+" + to_string(abs(m));
	    ex.LevelNames[lambda] = to_string(n) + lstring + mstring;
	    cout << vec_to_str(ex.QuantumNumbers[lambda]) << endl;
	    mIndices.push_back(lambda);
	    ShellIndices.push_back(lambda);
	    lambda++;
	  }
	}
	if(ex.settings[17]==1) ex.QuantNumRestrictIndexVectors.push_back(mIndices);
      }
      if(ex.settings[17]==2) ex.QuantNumRestrictIndexVectors.push_back(ShellIndices);
      n++;
    }
  }
}

double Energy1pEx(int lambda, exDFTstruct &ex){
  if(ex.System==100){//V_ext == 1D harmonic oscillator
    return ex.UNITenergy*(lambda+0.5);
  }
  else if(ex.System==101){//V_ext == 3D Coulomb-ion of charge Z, nonrelativistic
//     int OrbitalNumber = lambda+1, Shell = 1;
//     while(Shell*(1+Shell)*(1+2*Shell)<6*OrbitalNumber) Shell++;
//     return -ex.UNITenergy*ex.params[1]*ex.params[1]/(2.*Shell*Shell);
    return -ex.UNITenergy*ex.params[1]*ex.params[1]/(2.*ex.QuantumNumbers[lambda][0]*ex.QuantumNumbers[lambda][0]);
  }
  else if(ex.System==102){//V_ext == 3D Coulomb-ion of charge Z, relativistic
    return ex.E1pLoaded[lambda];
  }  
  else return 0.; 
}

void LoadE1p_X2C(exDFTstruct &ex){
	cout << "LoadE1p_X2C" << endl;
	ex.E1pLoaded.clear(); ex.E1pLoaded.resize(0);
	int Z = ex.params[1];
	ReadVec("TabFunc_X2C_E_Z=" + to_string(Z) + ".dat", ex.E1pLoaded);
	vecFact(ex.E1pLoaded,ex.UNITenergy);
	if(ex.E1pLoaded.size()==0) cout << "LoadE1p_X2C Error !!!" << endl;
	else for(int mu=0;mu<ex.E1pLoaded.size();mu++) cout << "ex.E1pLoaded[" << mu << "] = " << ex.E1pLoaded[mu] << endl;
}

void LoadCMatrix_X2C(exDFTstruct &ex){
	cout << "LoadCMatrix_X2C" << endl;
	vector<double> cMatrix(0);
	int Z = ex.params[1], a = 0;
	ReadVec("TabFunc_X2C_C_Z=" + to_string(Z) + ".dat", cMatrix);
	int LL = (int)(sqrt((double)cMatrix.size())+0.5);
	ex.CMatrixLoaded.clear(); ex.CMatrixLoaded.resize(LL);
	for(int mu=0;mu<LL;mu++){
		ex.CMatrixLoaded[mu].resize(LL);
		for(int k=0;k<LL;k++){
			if(abs(cMatrix[a])>1.0e-12) ex.CMatrixLoaded[mu][k] = cMatrix[a];
			else ex.CMatrixLoaded[mu][k] = 0.;
			a++;
		}
		cout << "ex.CMatrixLoaded[" << mu << "][0] = " << ex.CMatrixLoaded[mu][0] << endl;
	}
}

vector<double> InitializeOccNum(bool noninteractingQ, exDFTstruct &ex){//Alex? Better way of initialization, such that we get a good representation of the total search space for any demanded number (e.g., from 10 to 1000) of occupation-number-vectors
  int L = ex.settings[0];
  double N = ex.Abundances[0], Lmax = (double)L, ran;
  vector<double> occnum(L);
  
  if(noninteractingQ){
    fill(occnum.begin(), occnum.end(), 0.);
    double accumulated = 0.;
    for(int l=0;l<L;l++){
      if(accumulated+2.<N) occnum[l] = 2.;
      else{
	occnum[l] = N-accumulated;
	break;
      }
      accumulated += occnum[l];
    }
  }
  else if(ex.settings[3]==1){//linear decrease
    if(Lmax>=N){
      double delta = 2.*N/(Lmax*Lmax-Lmax);
      for(int l=0;l<L;l++) occnum[l] = min(2.,max(0.,delta*(Lmax-(double)(l+1)))); 
    }
    else{
      double delta = (4.*Lmax-2.*N)/(Lmax*Lmax-Lmax);
      for(int l=0;l<L;l++) occnum[l] = min(2.,max(0.,2.-delta*(double)l));
    }
  }
  else if(ex.settings[3]==2){//random(version1)
    double n = N/Lmax;
    fill(occnum.begin(), occnum.end(), n);
    for(int l=0;l<(int)(0.5*Lmax+0.1);l++){ ran = min(n,2.-n)*ex.RNpos(ex.MTGEN); occnum[l] += ran; occnum[L-1-l] -= ran; } 
  }
  else if(ex.settings[3]==3){//random(version2)
    for(int l=0;l<L;l++) occnum[l] = 2.*ex.RNpos(ex.MTGEN);
    //cout << "InitializeOccNum:              occnum = " << vec_to_str(occnum) << endl;
    
    //cout << "InitializeOccNum: NearestProperOccNum = " << vec_to_str(occnum) << endl;
  }  
  else if(ex.settings[3]==4){//manual input
vector<double> input = {{0.17975156,1.5752083,1.5874309,2.9266328,3.0444839,1.5723627,1.6459477,2.7464853,3.1415922,3.0735794,3.0695853,3.1197172,3.0936468,3.0675258}};
    for(int l=0;l<L;l++) occnum[l] = input[l];
  }
  else if(ex.settings[3]==5){//load from mpDPFT_IntermediateOPTresults.dat (phases)
    vector<double> input(LoadIntermediateOPTresults());
    for(int l=0;l<L;l++) occnum[l] = 1.+cos(input[l]);
  }   
  //cout << "InitializeOccNum done" << endl;
  if(!noninteractingQ) NearestProperOccNum(occnum,ex);//That is, InitializeOccNum delivers proper occnum
    
  return occnum;
}

void NearestProperOccNum(vector<double> &occ, exDFTstruct &ex){
  int L = ex.settings[0];
  
  //pre-process
  for(int l=0;l<L;l++){
    if(occ[l]<0.)occ[l] = 0.;
    if(occ[l]>2.)occ[l] = 2.;
  }  
  //cout << "NearestProperOccNum: pre-process done" << endl;
  
	if(ex.settings[5]==0){
		//Do nothing
	}
	else if(ex.settings[5]==1){//MIT
		double N = ex.Abundances[0], Nocc = accumulate(occ.begin(),occ.end(),0.0);
		if(ABS(Nocc)<1.0e-16){ fill(occ.begin(),occ.end(),N/((double)L)); cout << "NearestProperOccNum: W A R N I N G --- invalid request of occupation numbers !!!" << endl; }
		else{
			vecFact(occ,N/Nocc);//rescale occ
			for(int l=0;l<L-1;l++){
				if(occ[l]>2.){
					occ[l+1] += occ[l]-2.;//move excess to next occnum
					occ[l] = 2.;
				}
				else if(occ[l]<0.){
					occ[l+1] -= occ[l];//move excess to next occnum
					occ[l] = 0.;
				}
			}
		}
	}
  else if(ex.settings[5]>=2 && ex.settings[5]<=4){//Alex
    /*
	double N = ex.Abundances[0];
	int Ind = 0;
	double Nocc = accumulate(occ.begin(), occ.end(), 0.0), Excess = 0., delta, Occind;
	
	if(ex.settings[5]==2 || (ex.settings[5]==4 && ex.RNpos(ex.MTGEN)<0.5) ){// Rescale version: Rescale the vector onto the constrained plane
	  if (Nocc != N){
		for (int l = 0; l < L; l++){ occ[l] *= N/Nocc; }
	  }
	}
	else if(ex.settings[5]==3 || ex.settings[5]==4 ){//Projection version: Perpendicular projection of the vector onto the constrained plane
	  if (Nocc != N){
		for(int l=0; l<L; l++){ occ[l] = occ[l] - Nocc/((double)L) + N/((double)L); }
	  }
	}

	// Find the largest excess of the vector exceeding the box
	for(int loop = 0; loop < L; loop++){
		if( occ[loop] > 2. ){
			delta = occ[loop] - 2.;
		}
		else if( occ[loop] < 0. ){
			delta = -occ[loop];
		}
		else{
			delta = 0.;
		}
		if (delta > Excess){
			Excess = delta;
			Ind = loop;
		}
	}

	// Rescale the vector so that it goes to the nearest boundary of the box
	if ( Excess != 0. ){
		Occind = occ[Ind];
		for( int l=0; l<L; l++ ){ occ[l] = ABS( (ABS(Occind - N/((double)L)) - Excess)/( Occind - N/((double)L)) ) * (occ[l] - N/((double)L)) + N/((double)L); }
	}    */
	
	//parsimonious version:
	int frozen = 0, Lf = ex.FreeIndices.size();
	if(ex.settings[19]>100) frozen = (ex.settings[19] % 100)*2;
	double N = ex.Abundances[0]-(double)frozen;
	if(ABS(accumulate(occ.begin(),occ.end(),0.0))<1.0e-16){ fill(occ.begin(),occ.end(),N/((double)L)); cout << "NearestProperOccNum: W A R N I N G --- invalid request of occupation numbers !!!" << endl; }
	vector<double> Occ(ex.FreeIndices.size());
	//cout << "ex.FreeIndices.size() " << ex.FreeIndices.size() << endl;
		int j = 0;
		for(int i=0;i<L;i++){
			if(i==ex.FreeIndices[j]){
				Occ[j] = occ[i];
				j++;
			}
		}	
	double Ntest = accumulate(Occ.begin(), Occ.end(), 0.0), ratio = N/Ntest, Excess = 0., avn = N/((double)Lf), add = avn - Ntest/((double)Lf);
	
	int Ind = 0;
	if( ex.settings[5]==2 || (ex.settings[5]==4 && ex.RNpos(ex.MTGEN)<0.5) ) vecFact(Occ,ratio);// for rescaling onto constraining hyper-plane
	else if(ex.settings[5]==3 || ex.settings[5]==4 ) for(int l=0;l<Lf;l++) Occ[l] += add;// for projection onto constraining hyper-plane
	for(int l=0;l<Lf;l++){// Find the largest excess of the vector exceeding the box
		double delta = 0.;
		if(Occ[l]>2.) delta = Occ[l] - 2.;
		else if(Occ[l]<0.) delta = -Occ[l];
		if(delta>Excess){
			Excess = delta;
			Ind = l;
		}
	}

	double Delta = ABS(Occ[Ind] - avn);
	if(Excess!=0. && Delta!=0.) for(int l=0;l<Lf;l++) Occ[l] = avn + ABS(1.-Excess/Delta) * (Occ[l] - avn); // Rescale the vector so that it goes to the nearest boundary of the box	

	//for atomic systems
    if(ex.settings[19]>100){
		int j = 0;
		for(int i=0;i<L;i++){
			if(i<ex.settings[19] % 100){
				occ[i] = 2.;
			}
			else if(!IntegerElementQ(i,ex.FreeIndices)){
				occ[i] = 0.;					
			}
			else{ occ[i] = Occ[j]; j++; }
		}	
	}
	else occ = Occ;

	
// 	int Ind = 0;
// 	if( ex.settings[5]==2 || (ex.settings[5]==4 && ex.RNpos(ex.MTGEN)<0.5) ) vecFact(occ,ratio);// for rescaling onto constraining hyper-plane
// 	else if(ex.settings[5]==3 || ex.settings[5]==4 ) for(int l=0;l<L;l++) occ[l] += add;// for projection onto constraining hyper-plane
// 	for(int l=0;l<L;l++){// Find the largest excess of the vector exceeding the box
// 		double delta = 0.;
// 		if(occ[l]>2.) delta = occ[l] - 2.;
// 		else if(occ[l]<0.) delta = -occ[l];
// 		if(delta>Excess){
// 			Excess = delta;
// 			Ind = l;
// 		}
// 	}
// 	double Delta = ABS(occ[Ind] - avn);
// 	if(Excess!=0. && Delta!=0.) for(int l=0;l<L;l++) occ[l] = avn + ABS(1.-Excess/Delta) * (occ[l] - avn); // Rescale the vector so that it goes to the nearest boundary of the box

  }
  else if(ex.settings[5]==5){//MIT, not really working properly... why?
    double N = ex.Abundances[0], Ntest = accumulate(occ.begin(), occ.end(), 0.0), rescale;

      while(Ntest!=0. && ABS(Ntest/N-1.)>1.0e-14){
	rescale = N/Ntest; if(ABS(rescale-1.)>1.0e-14) vecFact(occ,rescale);//rescale
	for(int l=0;l<L;l++) occ[l] = min(2.,max(0.,occ[l]));//trim the occupation numbers
	Ntest = accumulate(occ.begin(), occ.end(), 0.0);
      }
      if(ABS(Ntest/N-1.)>1.0e-14){
	EXprint(1, "NearestProperOccNum: rescaling unsuccessful --- Ntest = " + to_string_with_precision(Ntest,16), ex);
	occ = InitializeOccNum(false,ex);
      }


  }  
  
  //cout << "NearestProperOccNum: initialized" << endl;
  
  //clean up
  bool repairedQ = false;
  for(int l=0;l<L;l++){
    if(occ[l]<0.){ if(occ[l]<-1.0e-15) EXprint(1, "NearestProperOccNum: negative occupation number " + to_string_with_precision(occ[l],20) + " ... repaired", ex); occ[l] = 0.; repairedQ = true; }
    else if(occ[l]>2.){ if(occ[l]>2.+1.0e-15) EXprint(1, "NearestProperOccNum: excessive occupation number " + to_string_with_precision(occ[l],20) + " ... repaired", ex); occ[l] = 2.; repairedQ = true; }
  }
  if(repairedQ) NearestProperOccNum(occ,ex);
  //cout << "NearestProperOccNum: repaired" << endl;
  
  double TotalOcc = accumulate(occ.begin(), occ.end(), 0.0);
  if(ABS(TotalOcc-ex.Abundances[0])/ex.Abundances[0]>1.0e-14) EXprint(1, "NearestProperOccNum: Warning!!! particle count " + to_string_with_precision(TotalOcc,20) + " != " + to_string_with_precision(ex.Abundances[0],20), ex);
  for(int l=0;l<L;l++){
    if(occ[l]<0.) EXprint(1, "NearestProperOccNum: Warning!!! negative occupation number " + to_string_with_precision(occ[l],20) + " ...the code should not even come here at all!", ex);
    else if(occ[l]>2.) EXprint(1, "NearestProperOccNum: Warning!!! excessive occupation number " + to_string_with_precision(occ[l],20) + " ...the code should not even come here at all!", ex);
  }
  //cout << "NearestProperOccNum: tested" << endl;
  
  if(ex.settings[17]>0){
    for(int q=0;q<ex.QuantNumRestrictIndexVectors.size();q++){
      double av = 0.;
      for(int p=0;p<ex.QuantNumRestrictIndexVectors[q].size();p++) av += occ[ex.QuantNumRestrictIndexVectors[q][p]];
      for(int p=0;p<ex.QuantNumRestrictIndexVectors[q].size();p++) occ[ex.QuantNumRestrictIndexVectors[q][p]] = av/((double)ex.QuantNumRestrictIndexVectors[q].size());
    }
  }
  //cout << "NearestProperOccNum: finished" << endl;
  //At this point, and if no Warnings were issued, the set of occupation numbers is proper: each is within [0,2] (strictly!) and they sum up to N (to within 1.0e-14 relative accuracy)
}

bool MonitorEx(int ErrorCode, vector<double> &occ, vector<vector<double>> &rho1p, exDFTstruct &ex){
  //export {ErrorCode,occ[0,...L-1],rho1p[0,...,L^2-1]} in one line
  int L = ex.settings[0];
  vector<double> tmp(1+L+L*L);
  tmp[0] = (double)ErrorCode;
  for(int l=0;l<L;l++) tmp[1+l] = occ[l];
  for(int l=0;l<L;l++) for(int k=0;k<L;k++) tmp[1+L+l*L+k] = rho1p[l][k];
  ex.MonitorMatrix.push_back(tmp);
  //cout << "Plugin_1pEx: MonitorEx: MonitorMatrix, size = " << ex.MonitorMatrix.size() << endl;
  return false;
}

bool rho1pConsistencyCheck(vector<vector<double>> &rho1p, vector<double> &occ, exDFTstruct &ex){
  double num = 0., AverageRhoSquared = 0., threshold = 1.0e-7;
  int L = ex.settings[0];
  vector<vector<double>> Dirac(L);
  vector<double> occnum(L);
  for(int l=0;l<L;l++){
    Dirac[l].resize(L);
    occnum[l] = rho1p[l][l]*ex.Abundances[0];
    if(occnum[l]<-1.0e-14 || occnum[l]>2.+1.0e-14){
      EXprint(1, "rho1pConsistencyCheck: Warning!!! occupation numbers =  " + vec_to_CommaSeparatedString_with_precision(occnum,20), ex);
      EXprint(1, "...out of bounds", ex);
      return MonitorEx(1,occ,rho1p,ex);
    }    
    if(abs(occnum[l]-occ[l])>threshold){
      EXprint(1, "rho1pConsistencyCheck: Warning!!! rho-diagonal[" + to_string(l) + "] =  " + to_string_with_precision(occnum[l],20), ex);
      EXprint(1, "                       but targeted occupation number =  " + to_string_with_precision(occ[l],20), ex);
      EXprint(1, "...(accumulated=" + to_string_with_precision(accumulate(occ.begin(),occ.end(),0.),20) + ") missed!?", ex);
      return MonitorEx(2,occ,rho1p,ex);
    }
    num += occnum[l];
    for(int k=0;k<L;k++){
      Dirac[l][k] = -2.*ex.Abundances[0]*rho1p[l][k];
      for(int m=0;m<L;m++){
	Dirac[l][k] += ex.Abundances[0]*rho1p[l][m]*ex.Abundances[0]*rho1p[m][k];
      }
      AverageRhoSquared += abs(Dirac[l][k]);
    }
  }
  AverageRhoSquared /= double(L*L);
  if(ABS(num-ex.Abundances[0])>threshold){
    EXprint(1, "rho1pConsistencyCheck: Warning!!! particle number (target=" + to_string_with_precision(ex.Abundances[0],20) + ") = " + to_string_with_precision(num,20),ex);
    EXprint(1, "...targeted particle number missed", ex);
    return MonitorEx(3,occ,rho1p,ex);
  }
  //(un)comment option 1
  if(AverageRhoSquared>threshold){
    EXprint(1, "rho1pConsistencyCheck: Warning!!! average absolute deviation of RDM-elements from Dirac criterion = " + to_string_with_precision(AverageRhoSquared,20), ex);
    EXprint(1, "...more than threshold", ex);
    return MonitorEx(4,occ,rho1p,ex);
  }
  return true;
}

void LoadOptOccNum(exDFTstruct &ex){
    ifstream infile;
    infile.open("mpDPFT_1pExDFT_OptOccNum.dat");
    string line;
    int a;
    while(getline(infile,line)){
      istringstream iss(line);
      if(line != ""){
	iss.str(line);
 	iss >> a; iss >> ex.OccNum[a]; iss >> ex.Phases[a];
      }
    }
    infile.close();
    EXprint(1, "LoadOptOccNum: \n Occnum = " + vec_to_str(ex.OccNum) + "\n Phases = "  + vec_to_str(ex.Phases), ex);
}

void GetRho1p(vector<vector<double>> &rho1p, vector<double> occ, exDFTstruct &ex){//ensure that Num = even (but type double), and that occ is proper before passing to GetRho1p
  int error = 1;
  int L = ex.settings[0], count = 0;
  double Num = ex.Abundances[0];
  
  //test ground:
//   vector<double> occTest = {{0.2311563478821805,0.2358322385381099,0.09765426840120053,0.003193780905960676,0.1438842647888048,0.02637246374379599,0.220606360521806,0.06863534763873903,0.149989715234786,0.1034576235600996,0.08892622092157955,0.1577261359908214,1.110223024625157e-16,0.2469205557028926,0.2256446761692233}};
//   error = GenerateRho1pMixer(rho1p, L, occTest, Num);
//   cout << "errorCode = " << error << endl;
//   MatrixToFile(rho1p,"mpDPFT_1pExDFT_rho1p_error1.dat",16);
//   occTest = {{0.2311563478821805,0.2358322385381099,0.09765426840120053,0.003193780905960676,0.1438842647888048,0.02637246374379599,0.220606360521806,0.06863534763873903,0.149989715234786,0.1034576235600996,0.08892622092157955,0.1577261359908214,1.110223024625157e-12,0.2469205557028926,0.2256446761692233}};
//   error = GenerateRho1pMixer(rho1p, L, occTest, Num);
//   cout << "errorCode = " << error << endl;
//   MatrixToFile(rho1p,"mpDPFT_1pExDFT_rho1p_error2.dat",16);
//   usleep(100*1000000);
  
  if(ex.settings[1]==100){//diagonal RDM
    for(int i=0;i<rho1p.size();i++) fill(rho1p[i].begin(),rho1p[i].end(),0.);
    for(int i=0;i<L;i++) rho1p[i][i] = occ[i]/Num;
  }
  else if(ex.settings[1]==101 && L%2==0 && Num>=0.){//for special sequence {n1,n2,...,2-n2,2-n1} of occupation numbers, probably not useful
    for(int i=0;i<L;i++){
      fill(rho1p[i].begin(),rho1p[i].end(),0.);
      for(int j=0;j<L;j++){
	if(i==j) rho1p[i][j] = occ[i]/Num;
	else if(j==L-i) rho1p[i][j] = sqrt(2.*occ[i]-occ[i]*occ[i])/Num;
      }
    }
  }
  else if(ex.settings[1]==102){//Berge's algorithm, implemented by Mikolaj
    //error = GenerateRho1pMixer(rho1p, L, occ, Num, ex.settings[18]);
    error = GenerateRho1pMixer(rho1p, L, occ, Num);
    if(ex.settings[9]>0 && error!=0){
      EXprint(0,"GetRho1p: Error = " + to_string(error) + " !!!", ex);
      EXprint(0,"          occ = " + vec_to_CommaSeparatedString_with_precision(occ,20) + "\n" + "occ accumulated = " + to_string_with_precision(accumulate(occ.begin(),occ.end(),0.),20) + "==" + to_string_with_precision(ex.Abundances[0],20) + " ?" + "\n", ex);
      EXprint(0,"          export result of GenerateRho1pMixer = ", ex);
      for(int l=0;l<L;l++) EXprint(0,"          " + vec_to_CommaSeparatedString_with_precision(rho1p[l],20) + ",\n", ex);
      if(ex.settings[9]>1) MatrixToFile(rho1p,"mpDPFT_1pExDFT_rho1p_error.dat",16);
    }
  }
  else if(ex.settings[1]==103){//Berge's TF-analog, cot-version
    double threshold = 1.0e-12;
    //double threshold = 1.0e-6;
    double PiHalf = 0.5*3.141592653589793, alpha;
    for(int i=0;i<L;i++) rho1p[i][i] = occ[i];// /Num;
    for(int i=0;i<L;i++){
      for(int j=i+1;j<L;j++){//j>i
	if(rho1p[i][i]<threshold || rho1p[j][j]<threshold || rho1p[i][i]>2.-threshold || rho1p[j][j]>2.-threshold) rho1p[i][j] = 0.;
	else{ alpha = atan( 2./(1./tan(PiHalf*rho1p[i][i])+1./tan(PiHalf*rho1p[j][j])) ); rho1p[i][j] = sin((double)(i-j)*alpha)/(PiHalf*(double)(i-j)); }
	rho1p[j][i] = rho1p[i][j];
      }
    }
    for(int i=0;i<L;i++) for(int j=0;j<L;j++) rho1p[i][j] /= Num;
    //for(int i=0;i<L;i++){ for(int j=0;j<L;j++) cout << rho1p[i][j] << " "; cout << endl; } usleep(100000000);
  }
  else if(ex.settings[1]==104){//manual input, with specified dimension, be aware of difference between rho_1p and rho_mp
    int inputdimension = 16;
    //double prefactor = 1.;//for input of type rho_1p
    double prefactor = 1./Num;//for input of type rho_mp
    vector<vector<double>> input(inputdimension); for(int i=0;i<inputdimension;i++) input.resize(inputdimension);
    input = {{1.53571,0.,0.373569,0.,-0.0407253,0.,0.00632582,0.,-0.000253579,0.,-0.000692242,0.,0.000621729,0.,-0.000424226,0.},{0.,0.270784,0.,-0.0642743,0.,0.0243077,0.,-0.0105979,0.,0.00497179,0.,-0.0024444,0.,0.00124294,0.,-0.000649332},{0.373569,0.,0.150737,0.,-0.0307442,0.,0.0114348,0.,-0.00515167,0.,0.00255475,0.,-0.00134337,0.,0.000732477,0.},{0.,-0.0642743,0.,0.0183735,0.,-0.00831101,0.,0.00429989,0.,-0.00237409,0.,0.00136218,0.,-0.000801623,0.,0.000480421},{-0.0407253,0.,-0.0307442,0.,0.00927116,0.,-0.00455726,0.,0.00254969,0.,-0.00150972,0.,0.00092206,0.,-0.000573632,0.},{0.,0.0243077,0.,-0.00831101,0.,0.0043782,0.,-0.00258547,0.,0.00160433,0.,-0.00102193,0.,0.000661085,0.,-0.000431917},{0.00632582,0.,0.0114348,0.,-0.00455726,0.,0.00265318,0.,-0.00168525,0.,0.00110782,0.,-0.00074052,0.,0.000499399,0.},{0.,-0.0105979,0.,0.00429989,0.,-0.00258547,0.,0.0017021,0.,-0.00115916,0.,0.000801352,0.,-0.000557924,0.,0.000389727},{-0.000253579,0.,-0.00515167,0.,0.00254969,0.,-0.00168525,0.,0.00117942,0.,-0.000840978,0.,0.000603675,0.,-0.0004341,0.},{0.,0.00497179,0.,-0.00237409,0.,0.00160433,0.,-0.00115916,0.,0.000853966,0.,-0.000632451,0.,0.000468381,0.,-0.000346119},{-0.000692242,0.,0.00255475,0.,-0.00150972,0.,0.00110782,0.,-0.000840978,0.,0.000642463,0.,-0.00049011,0.,0.000372369,0.},{0.,-0.0024444,0.,0.00136218,0.,-0.00102193,0.,0.000801352,0.,-0.000632451,0.,0.000497502,0.,-0.00038893,0.,0.000301964},{0.000621729,0.,-0.00134337,0.,0.00092206,0.,-0.00074052,0.,0.000603675,0.,-0.00049011,0.,0.000394603,0.,-0.000314823,0.},{0.,0.00124294,0.,-0.000801623,0.,0.000661085,0.,-0.000557924,0.,0.000468381,0.,-0.00038893,0.,0.000319214,0.,-0.000259113},{-0.000424226,0.,0.000732477,0.,-0.000573632,0.,0.000499399,0.,-0.0004341,0.,0.000372369,0.,-0.000314823,0.,0.00026257,0.},{0.,-0.000649332,0.,0.000480421,0.,-0.000431917,0.,0.000389727,0.,-0.000346119,0.,0.000301964,0.,-0.000259113,0.,0.000219075}};//for ex.params[0]=20
    for(int i=0;i<L;i++) for(int j=0;j<L;j++) rho1p[i][j] = prefactor*input[i][j];
  }
  else if(ex.settings[1]==105) for(int i=0;i<L;i++) for(int j=0;j<L;j++) rho1p[i][j] = sqrt(occ[i]*occ[j])/Num;
  else if(ex.settings[1]==106){//Schur-Horn algorithm, see 1p-exact-DFT_MIT_Schur-Horn_Algorithm.nb
    //cout << "Start SH algorithm" << endl;
    //begin USER input
    int testlength = 10;//choose even
    int StartStepAdaptationAt = (int)(0.1*(double)ex.SHmaxCount);
    bool testStageQ = false;
    //end USER input
    
    bool randomSeedQ = ex.SHrandomSeedQ;
    int CheckEvery = max(1,(int)((double)ex.SHmaxCount/100.));    
    
    vector<double> a; for(int i=0;i<L;i++) a.push_back(occ[i]-1.);
    double trU = accumulate(a.begin(),a.end(),0.0), sgntrU = Sign(trU), abstrU = abs(trU);
    int l = (int)(abs(trU)+0.1), k = L-l;
    if(k%2==1){ cout << "GetRho1p: Error k !!! Do something..." << endl; usleep(10000000); }
    vector<double> lambda; for(int i=0;i<l;i++) lambda.push_back(sgntrU); for(int i=0;i<k/2;i++){ lambda.push_back(1.); lambda.push_back(-1.); }
    sort(lambda.begin(),lambda.end());
    if(lambda.size()!=L){ cout << "GetRho1p: Error lambda.size() !!! Do something..." << endl; usleep(10000000); }
    if( (accumulate(lambda.begin(),lambda.end(),0.0)-trU)>1.0e-12 ){ cout << "GetRho1p: Error lambda content !!! Do something..." << endl; usleep(10000000); }
    
    MatrixXd Lambda = MatrixXd::Zero(L,L), IdentityMatrix = MatrixXd::Zero(L,L);
    for(int i=0;i<L;i++){ Lambda(i,i) = lambda[i]; IdentityMatrix(i,i) = 1.; }
    
    //cout << "initialize U" << endl;
    MatrixXd U(L,L);
    if(randomSeedQ){
      MatrixXd Orth = orthogonal_matrix(randomSeedQ,ex);
      U = Orth.transpose()*Lambda*Orth;
      if(testStageQ){ double test = (Orth.transpose()*Orth-IdentityMatrix).norm(); if(test>ex.SHConvergenceCrit){ cout << "GetRho1p: Warning !!! Orth not orthogonal: " << test << endl; } }
    }
    else U = ex.RandOrthMat.transpose()*Lambda*ex.RandOrthMat;
    
    //cout << "Start SH loop" << endl;
    double Convergence = 1., dt = ex.SHdt;
    vector<double> CH;//ConvergenceHistory
    vector<double> ev(L);
    int count = 0;
    MatrixXd xdot(L,L), tmp(L,L), alpha = MatrixXd::Zero(L,L);
    MatrixXcd diagonalizedU(L,L);
    while(Convergence>ex.SHConvergenceCrit && count<ex.SHmaxCount){
      count++; //cout << count << endl;
      for(int i=0;i<L;i++) alpha(i,i) = U(i,i) - a[i];
      tmp = alpha*U-U*alpha;
      xdot = U*tmp-tmp*U;
      U += dt*xdot;
      if(count>CheckEvery && count%CheckEvery==0){
	EigenSolver<MatrixXd> es(U);
	
	//option 1
// 	MatrixXcd P = (es.eigenvectors()).transpose();
// 	MatrixXcd POrt = P*((P.transpose()*P).cwiseSqrt()).inverse();
// 	Convergence = ((POrt.transpose()*U*POrt).real()-Lambda).norm();
	//option 2
	for(int i=0;i<L;i++) ev[i] = es.eigenvalues()[i].real();
	sort(ev.begin(),ev.end());
	Convergence = Norm(VecDiff(ev,lambda));
	//option 3
// 	diagonalizedU = es.eigenvalues().asDiagonal();
// 	Convergence = (diagonalizedU.real()-Lambda).norm();
	
	//cout << "GetRho1p: Convergence = " << Convergence << " U(0,0) = " << U(0,0) << endl;
	CH.push_back(Convergence);
	if(varianceConvergedQ(CH, testlength, ex.SHConvergenceCrit)) break;
	//if(count>StartStepAdaptationAt){ int H = CH.size(); if(H>1){ if(CH[H-1]>CH[H-2]) dt *= 0.8; else dt *= 1.2; } }
      }
    }
    
    if(testStageQ && count==ex.SHmaxCount) cout << "GetRho1p: Warning !!! SHmaxCount reached --- Convergence = " << Convergence << "(target=" << ex.SHConvergenceCrit << ") dt = " << dt << endl;
    //else if(testStageQ) cout << "GetRho1p: SH count = " << count << "/" << ex.SHmaxCount << " --- dt = " << dt << endl;
    if(testStageQ && (U*U-IdentityMatrix).norm()/((double)L)>0.01){ cout << "GetRho1p: Warning !!! U^2 deviation from identity " << (U*U-IdentityMatrix).norm() << endl; }
    
    vector<double> Udiag(L);
    for(int i=0;i<L;i++){//clean up U
      Udiag[i] = U(i,i);
      U(i,i) = a[i];
      for(int j=i+1;j<L;j++) U(i,j) = U(j,i);
    }
    if(testStageQ && Norm(VecDiff(Udiag,a))/Norm(a)>0.01){
      cout << "GetRho1p: Warning !!! U-diagonal deviation from target " << Norm(VecDiff(Udiag,a)) << endl;
      cout << vec_to_str_with_precision(a,6) << endl;
      cout << vec_to_str_with_precision(Udiag,6) << endl;      
    }
    for(int i=0;i<L;i++) for(int j=0;j<L;j++) rho1p[i][j] = (U(i,j)+IdentityMatrix(i,j))/Num;
    
    if(testStageQ) bool rhoOk = rho1pConsistencyCheck(rho1p,occ,ex);
  }
  else if(ex.settings[1]==107){//Berge's TF-analog, cos-version
    double threshold = 1.0e-12;
    double PiHalf = 0.5*3.141592653589793, alpha;
    for(int i=0;i<L;i++) rho1p[i][i] = occ[i];// /Num;
    for(int i=0;i<L;i++){
      for(int j=i+1;j<L;j++){//j>i
// 	if(rho1p[i][i]<threshold || rho1p[j][j]<threshold || rho1p[i][i]>2.-threshold || rho1p[j][j]>2.-threshold) rho1p[i][j] = 0.;
// 	else{ alpha = acos( 0.5*(cos(PiHalf*rho1p[i][i])+cos(PiHalf*rho1p[j][j])) ); rho1p[i][j] = sin((double)(i-j)*alpha)/(PiHalf*(double)(i-j)); }
	alpha = acos( 0.5*(cos(PiHalf*rho1p[i][i])+cos(PiHalf*rho1p[j][j])) ); rho1p[i][j] = sin((double)(i-j)*alpha)/(PiHalf*(double)(i-j));
	rho1p[j][i] = rho1p[i][j];
      }
    }
    for(int i=0;i<L;i++) for(int j=0;j<L;j++) rho1p[i][j] /= Num;
    //for(int i=0;i<L;i++){ for(int j=0;j<L;j++) cout << rho1p[i][j] << " "; cout << endl; } usleep(100000000);
  }
  

  
  //ad-hoc clean-up
  for(int i=0;i<L;i++){
    double na = Num*rho1p[i][i];
    if(na<0. || na>2.){ /*cout << "GetRho1p: na = " << na << " out of bounds --> clamp row and column" << endl;*/ for(int j=0;j<L;j++){ rho1p[i][j] = 0.; rho1p[j][i] = 0.; } }
    if(na>2.) rho1p[i][i] = 2./Num;
  }
  
  bool ResetToDiagonalQ = false;
  if(ex.settings[9]>0) if(!rho1pConsistencyCheck(rho1p,occ,ex)){ GetRho1p(rho1p,occ,ex); if(!rho1pConsistencyCheck(rho1p,occ,ex)) ResetToDiagonalQ = true; }
  if(ex.settings[1]==102 && error!=0) ResetToDiagonalQ = true; 
  if(ResetToDiagonalQ){ /*EXprint(1, "GetRho1p: Reset rho1p to diagonal", ex);*/ for(int i=0;i<L;i++){ fill(rho1p[i].begin(),rho1p[i].end(),0.); rho1p[i][i] = occ[i]/Num; } }
}

void TransformRho1p(vector<vector<double>> &rho1p, vector<vector<double>> &rho1pIm, vector<double> &unitaries, vector<double> &extraphases, exDFTstruct &ex){
	int L = ex.settings[0];
	vector<double> occ1p(L);
	
	MatrixXcd M(L,L);
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			M(i,j) = rho1p[i][j];
		}
		occ1p[i] = rho1p[i][i];
	}
	
	bool basic = false;
	
	int u = 0;
	for(int a=0;a<L-1;a++){
		for(int b=a+1;b<L;b++){
			MatrixXcd U = MatrixXcd::Identity(L,L);
			std::complex<double> Maa(rho1p[a][a],0.0);
			std::complex<double> Mab(M(a,b).real(),M(a,b).imag());
			std::complex<double> Mba(M(b,a).real(),M(b,a).imag());
			std::complex<double> Mbb(rho1p[b][b],0.0);
			double phi = unitaries[u], xi_a = 0., xi_b = 0., A = Maa.real()-Mbb.real(), B = 2.*(Mab.real()*cos(phi)+Mab.imag()*sin(phi)), A2B2 = A*A+B*B;
			if(ex.settings[20]==2){ xi_a = extraphases[u]; xi_b = extraphases[L*(L-1)/2+u]; }
			if( abs(A)>1.0e-12 && (basic || (abs(Mab)>1.0e-12 && A2B2>1.0e-16 && std::isfinite(A2B2))) ){
				if(basic){ double co = cos(phi), si = sin(phi); U(a,a) = co - si * 1.0i; U(a,b) = 0.; U(b,a) = 0.; U(b,b) = co + si * 1.0i; }
				else{
					std::complex<double> prefactor(1.0/sqrt(A2B2),0.0);
					U(a,a) = prefactor*A*exp(xi_a*1.0i); U(a,b) = prefactor*(Mba*exp(phi*2.0i)+Mab)*exp(xi_b*1.0i); U(b,a) = prefactor*(Mab*exp(-phi*2.0i)+Mba)*exp(xi_a*1.0i); U(b,b) = prefactor*(-A)*exp(xi_b*1.0i);
				}
				if(std::isfinite(abs(U(a,a))) && std::isfinite(abs(U(a,b))) && std::isfinite(abs(U(b,a))) && std::isfinite(abs(U(b,b)))){
					M = U.adjoint()*M*U;
					if(ABS(M(a,a).real()-occ1p[a])>1.0e-12 || ABS(M(b,b).real()-occ1p[b])>1.0e-12){ cout << endl << "TransformRho1p: " << M(a,a).real()-occ1p[a] << " " << M(a,a).imag() << " " << M(b,b).real()-occ1p[b] << " " << M(b,b).imag() << endl << endl; }
				}
				else ex.UnitariesFlag[u] = 1.;
			}
			u++;
		}
	}
	
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			rho1p[i][j] = M(i,j).real();
			rho1pIm[i][j] = M(i,j).imag();
		}
	}	
}

bool TestN(vector<double> &occ, exDFTstruct &ex){
	int L = ex.settings[0];
	double MachinePrecision = 1.0e-14, N = ex.Abundances[0];
	double Nocc = accumulate(occ.begin(), occ.end(), 0.0);
	if (ABS(Nocc - N) > MachinePrecision){
		return false;
	}

	for (int l = 0; l < L; l++){
		if (occ[l] > 2. + MachinePrecision || occ[l] < - MachinePrecision){
			return false;
		}
	}

	return true;
}

double H_int(int i, int j, int k, int l){
	if((i+j+k+l)%2 == 0){
                double H = 0.;
                int l0=1;//initialise
                int l1=1;
                int l2=1;
                int l3=1;
                double sgn=1.;
                int max1=max(i,j);//order among matrix element labels
                int min1=min(i,j);
                int max2=max(k,l);
                int min2=min(k,l);
                if(max1>max2){//if l2 comes from second pair
                    l2=max1;
                    l3=min1;
                    l1=max2;
                    l0=min2;
                }
                else{//otherwise order is correct
                    l2=max2;
                    l3=min2;
                    l1=max1;
                    l0=min1;
                }
                double fact_l0 = EXPnonzero(gsl_sf_lnfact((unsigned int)l0));//factorials
                double fact_l1 = EXPnonzero(gsl_sf_lnfact((unsigned int)l1));
                double fact_l2 = EXPnonzero(gsl_sf_lnfact((unsigned int)l2));
                double fact_l3 = EXPnonzero(gsl_sf_lnfact((unsigned int)l3));
		for(int r=0;r<l1+1;r++){
			for(int b=0;b<r+1;b++){
				for(int a=0;a<l2-b+1;a++){
					if(r+a-b <= l3){
						int G = i+j+k+l-2*r-2*a;
						double fact_G = EXPnonzero(gsl_sf_lnfact((unsigned int)G));//G factorial
						double halffact_G = EXPnonzero(gsl_sf_lnfact((unsigned int)G/2));//G/2 factorial
						double G_only = (fact_G/halffact_G)/pow(-4.,(double)G/2.);//term with only G dependency
						double fact_a = EXPnonzero(gsl_sf_lnfact((unsigned int)a));//a factorial
						double fact_b = EXPnonzero(gsl_sf_lnfact((unsigned int)b));//b factorial
						double fact_rb = EXPnonzero(gsl_sf_lnfact((unsigned int)(r-b)));//r-b factorial
						double fact_l1r = EXPnonzero(gsl_sf_lnfact((unsigned int)(l1-r)));//l1-r factorial
						double fact_l2ab = EXPnonzero(gsl_sf_lnfact((unsigned int)(l2-a-b)));//l2-a-b factorial
						double fact_l3rab = EXPnonzero(gsl_sf_lnfact((unsigned int)(l3-r-a+b)));//l3-r-a+b factorial
						H = H + G_only/(fact_l1r*fact_l2ab*fact_a*fact_b*fact_l3rab*fact_rb);//add to sum
					}
				}
			}
		}
		if(l0%2==1){sgn=-1.;}//-1^lambda
		else{sgn=1.;}
		if(i==j && i==k && i==l) cout << std::setprecision(10) << "H_int = " << H*sgn*sqrt(3.141592653589793*fact_l1*fact_l2*fact_l3/fact_l0) << " --- H = " << H << " --- factorials = " << sgn*sqrt(3.141592653589793*fact_l1*fact_l2*fact_l3/fact_l0) << endl;//MIT
		return(H*sgn*sqrt(3.141592653589793*fact_l1*fact_l2*fact_l3/fact_l0));
	}
	else{return 0.;}
}

long double T(int l0,int alpha,int alpha1,int l3,long double s00){//l0,l1,l2,l3 must be ordered
long double H=0.;
long double sj0=s00;
long double sjj=s00;
int Gj0 = (alpha+alpha1)/2;//twice of G
int Gjj = (alpha+alpha1)/2;
for(int j=0;j<l0+1;j++){
    sjj=sj0;//reset j1=0 at j
    Gjj=Gj0;//reset Gjj to Gj0
    for(int j1=0;j1<l3+1;j1++){
        H=H+sjj;//update H
        sjj=sjj*S(j1,Gjj,l3,alpha1);//update summand s(j+1)
        Gjj=Gjj+1;//increase Gjj for next iteration
    }
    sj0=sj0*S(j,Gj0,l0,alpha);//update sj0 for j1=0, j=j+1
    Gj0=Gj0+1;//increase Gj0 for next iteration
}return(H*0.5*sqrt(2./3.141592653589793));
}

double fact_ratio(int b, int a, int N){//ratio of b! to a! for b>=a, use stirling for integers greater than N
  if(a>N) return EXPnonzero(gsl_sf_lngamma(b+1)-gsl_sf_lngamma(a+1));//both large
  if(a<=N && b>N) return sqrt(2.*3.141592653589793*(double)b)*EXPnonzero((double)b*log((double)b)-(double)b)/(double)EXPnonzero(gsl_sf_lnfact((double)a));//for large b, stirling.
  else if(b>10) return (double)EXPnonzero(gsl_sf_lnfact((double)b)-gsl_sf_lnfact((double)a));
  int x=1;
  for(int i=b;i>a;i--) x *= i;
  return (double)x;
}

double AsymL(int n,int alpha,double x){//asymptotic expression, note x=K^2
  double theta = 2.*sqrt((double)n)*x-(((double)alpha/2.)+0.25)*3.141592653589793;
  double b1 = ((x*x*x)/12.-0.25*(double)(alpha*alpha)/(double)x-0.5*((double)alpha*x)-0.5*(double)x+0.0625/(double)x);
  double prefactor = EXPnonzero((((double)alpha/2.)-0.25)*log((double)n))/sqrt(3.141592653589793)/sqrt(x);
  return prefactor*(cos(theta));//+(b1/sqrt((double)n))*sin(theta));
}

double alph_up(double lambda, double alpha, double alpha1, double x){
return -x*(1.+alpha1/(alpha+1.))*sqrt((alpha+2.+lambda)*(alpha+1.+lambda))/(alpha*2.+4.);}

double integral_div2(int l0, int l1, int l2, int l3,double a,double b,double subdiv,int Ns,int Na){//integrates for range a<x<b,with N subdivisions
  //Ns is large integer to use stirling
  //Na is large integer to use asymptotic laguerre polynomial
  //if(j<i){ int dummy = j; j=i; i=dummy; }
  //if(j<i){swap(i,j);}//j>=i
  //if(k<l){swap(k,l);}//k>=l
  //if(i>l){swap(i,l); swap(j,k);}//arrange couples such that l>i
  int i=l0;
  int j=l1;
  int k=l2;
  int l=l3;
  if(l0>l3){
  	i=l3;
  	j=l2;
  	k=l1;
  	l=l0;
  }
  double sgn=-1.;
  if((j-i+l-k)%4==0){sgn=1.;}
  double Integral=0.;
  double dx=(b-a)/subdiv;
  //cout << "i=" << i << " j=" << j << " k=" << k << " l=" << l << endl;
  double prefactor=sgn/sqrt(fact_ratio(j,i,Ns))/sqrt(fact_ratio(k,l,Ns));
  //for(double x=a;x<b;x=x+dx){//cannot start from 0
  for(int d=0;d<(int)(subdiv+0.5);d++){//cannot start from 0
    double x = a+0.5*dx+(double)d*dx;
    if(l<=Na || x<1.){//will not use asymptotic if both i and l are not large, or if x is too small
      Integral += dx/sqrt(x)*pow(x,(double)((j+k-i-l)/2))*EXPnonzero(-x)*gsl_sf_laguerre_n(i,(double)(j-i),x)*gsl_sf_laguerre_n(l,(double)(k-l),x);
    }
    else{//asymptotic approximation for k,l couple
      if(i<=Na && l>Na){
	Integral += dx/sqrt(x)*pow(x,(double)((j-i)/2))*EXPnonzero(-x/2.)*gsl_sf_laguerre_n(i,(double)(j-i),x)*AsymL(l,k-l,x);
      }
      else{//asymptotic approximation for both
	Integral=Integral+dx/sqrt(x)*AsymL(i,j-i,x)*AsymL(l,k-l,x);
      }
    }
  //cout<<"Integral="<<Integral<<endl;
  }
  return prefactor*Integral/(sqrt(2.)*3.141592653589793);
}

double integral2(int i,int j,int k,int l,double a,double b,double crit,double subdiv,double small,int Ns,int Na,double T0,double Est){
  //T=integral_div(i,j,k,l,a,b,subdiv,Ns,Na) is a guess for the value of the integral
  double midpoint = (a+b)/2.;
  //what if T is very small, like 1.0e-12
  if(abs(T0)<small){return T0;}//accept if T is small enough
  //cout << "T=" << T0 << endl;
  double T1=integral_div2(i,j,k,l,a,midpoint,subdiv,Ns,Na);
  double T2=integral_div2(i,j,k,l,midpoint,b,subdiv,Ns,Na);
  if((b-a)/subdiv<MinDelta) return T1+T2;
  if(abs(T0-(T1+T2))<=crit*abs(T0)){
        //cout<<T0<<endl;
        return T1+T2;
  }
  else {
    return integral2(i,j,k,l,a,midpoint,crit,subdiv,small/2.,Ns,Na,T1,Est)+integral2(i,j,k,l,midpoint,b,crit,subdiv,small/2.,Ns,Na,T2,Est);}
}

double inf_integral2(int i,int j,int k,int l,double b,double tail,double crit,double subdiv,double small,int Ns,int Na,double crit1,double small1,double T0,double Est){
  //double T=JonahIntegral(i,j,k,l,a,b,crit,lambda,subdiv,small,Ns,Na);
  if(abs(T0)<small1){return T0;}//accept if T is small enough
  //cout << "T=" << T0 << endl;
  double T1=integral2(i,j,k,l,b,b+tail,crit,subdiv,small,Ns,Na,T0,Est);
  if(b+tail>tailMultiplier*tail) return T0+T1;
  if(abs(T1)<=crit1*abs(T0)) {
        //cout<<T0<<endl;
        return T0+T1;
  }
  else {//cout<<T0<<endl;
    return inf_integral2(i,j,k,l,b+tail,tail,crit,subdiv,small,Ns,Na,crit1,small1,T0+T1,Est);}
}

double JonahIntegral(int i,int j,int k,int l,double a,double b,double crit,double subdiv,double small,int Ns,int Na,double T0,double Est){//amended(MIT)
  double midpoint = 0.5*(a+b);
  if(ABS(T0)<small && log((b-a)/(tailMultiplier*10.))/log(0.5)>10.) return T0;
  double T1=integral_div(i,j,k,l,a,midpoint,subdiv,Ns,Na);
  double T2=integral_div(i,j,k,l,midpoint,b,subdiv,Ns,Na);
  if((b-a)/subdiv<MinDelta || ABS(T0-(T1+T2))<0.1*crit*min(ABS(T0),ABS(T1+T2))/* || ABS(T0-(T1+T2))<crit*Est*/) return T1+T2;
  else return JonahIntegral(i,j,k,l,a,midpoint,crit,subdiv,0.5*small,Ns,Na,T1,Est)+JonahIntegral(i,j,k,l,midpoint,b,crit,subdiv,0.5*small,Ns,Na,T2,Est);
}

double inf_integral(int i,int j,int k,int l,double b,double tail,double crit,double subdiv,double small,int Ns,int Na,double crit1,double small1,double T0,double Est){//amended(MIT)
  //if(ABS(T0)<small1) return T0;
  double T1=JonahIntegral(i,j,k,l,b,b+tail,crit,subdiv,small,Ns,Na,T0,Est);
  if(b+tail>tailMultiplier*tail || ABS(T1)<small/tailMultiplier || ABS(T1)<crit*min(ABS(T0),ABS(T0+T1))/tailMultiplier) return T0+T1;
  else return inf_integral(i,j,k,l,b+tail,tail,crit,subdiv,small,Ns,Na,crit1,small1,T0+T1,Est);
}

double integral_div(int l0, int l1, int l2, int l3,double a,double b,double subdiv,int Ns,int Na){//called with l1>=l0 & l2>=l3
  //amended(MIT)
  //integrates for range a<x<b, with (int)subdiv intervals of size dx; Ns is large integer to use stirling, Na is large integer to use asymptotic laguerre polynomial
  int i=l0, j=l1, k=l2, l=l3;//i.e. j>=i & k>=l
  if(l0>l3){ i=l3; j=l2; k=l1; l=l0; }//swap i<->l & j<->k ==> k>=l & j>=i 
  double sgn=-1.; if((j-i+l-k)%4==0 || (j-i+l-k)==0) sgn=1.;
  int bp = max(1,(int)(0.25*subdiv)), Lf = 4*bp+1;
  double dx=(b-a)/((double)(Lf-1)), factRatiosPow = -0.5*(gsl_sf_lnfact((double)j)-gsl_sf_lnfact((double)i)+gsl_sf_lnfact((double)k)-gsl_sf_lnfact((double)l)), pow1 = (double)(j+k-i-l), pow2 = (double)(j-i), arg = (double)(k-l);
  vector<double> f(Lf); fill(f.begin(),f.end(),0.);
  for(int d=0;d<Lf;d++){
    double x = a+(double)d*dx, x2 = x*x;
    if(x>1.0e-300){
      if(l<=Na || x<1.){
	double pexp = EXP(pow1*LOG(x)-x2+factRatiosPow);
	if(pexp>0.) f[d] = pexp*(boost::math::laguerre((unsigned)i,(unsigned)(j-i),x2))*(boost::math::laguerre((unsigned)l,(unsigned)(k-l),x2));
      }
      else if(i<=Na && l>Na){
	double pexp = EXP(pow2*LOG(x)-0.5*x2+factRatiosPow);
	if(pexp>0.) f[d] = pexp*(boost::math::laguerre((unsigned)i,(unsigned)(j-i),x2))*AsymL(l,k-l,x);
      }
      else f[d] = EXP(factRatiosPow)*AsymL(i,j-i,x)*AsymL(l,k-l,x);
    }
  }
  return sgn*BooleRule1D(1,f,Lf,a,b,bp)/(sqrt(2.)*3.141592653589793);
}

double GetPartialIntegral(int i,int j,int k,int l,double a, double b,int MaxIter, int iter, double prevInt, double relCrit,double absCrit,double NumSubDivisions,int Ns,int Na, exDFTstruct &ex){
  //begin USERINPUT
  double prefactor = 2.*ex.params[0], minDelta = 1.0e-16, distinguishabilityThreshold = 1.0e-14, maxIterationQuantifier = 5., minIterationQuantifier = 0.3;
  //end USERINPUT
  int maxIter = max(2,(int)(maxIterationQuantifier*(double)MaxIter)), minIter = max(2,(int)(minIterationQuantifier*(double)maxIter));
  double midpoint = 0.5*(a+b);
  //calculate bisection values
  double A = prefactor*integral_div(i,j,k,l,a,midpoint,NumSubDivisions,Ns,Na);
  double B = prefactor*integral_div(i,j,k,l,midpoint,b,NumSubDivisions,Ns,Na);
  double C = A+B;
  double absDiff = ABS(prevInt-C);
  double relDiff = RelDiff(prevInt,C);
  //set flags for controlled termination
  bool resolutionLimit = false, iterLimit = false, TargetRelAccuracyReached = false, TargetAbsAccuracyReached = false, distinguishabilityLimit = false, alert = false;
  if(b-a<minDelta) resolutionLimit = true;
  if(iter==maxIter) iterLimit = true;
  if(absDiff<relCrit*min(ABS(prevInt),ABS(C))) TargetRelAccuracyReached = true;
  if(absDiff<absCrit) TargetAbsAccuracyReached = true;
  if(relDiff<distinguishabilityThreshold) distinguishabilityLimit = true;
  if(relDiff>max(0.01,sqrt(relCrit))) alert = true;
  //process bisection result
  if(!std::isfinite(A) || !std::isfinite(B)){ cout << "GetPartialIntegral: Integral not finite!" << endl; return prevInt; }
  if( iterLimit || (iter>minIter && (resolutionLimit || TargetRelAccuracyReached || TargetAbsAccuracyReached)) || distinguishabilityLimit){
    if(alert){
      if(resolutionLimit) cout << "GetPartialIntegral: minDelta (b-a<" << minDelta << ") reached!" << endl;
      if(iterLimit) cout << "GetPartialIntegral: maxIter=" << maxIter << " reached --- final successive section values " << prevInt << " --> " << C << endl;
    }
    //if(distinguishabilityLimit) cout << "GetPartialIntegral: successive section values numerically indistinguishable " << prevInt << " --> " << C << endl;
    return C;
  }
  double refinedA = GetPartialIntegral(i,j,k,l,a,midpoint,MaxIter,iter+1,A,relCrit,0.5*absCrit,NumSubDivisions,Ns,Na,ex);
  double refinedB = GetPartialIntegral(i,j,k,l,midpoint,b,MaxIter,iter+1,B,relCrit,0.5*absCrit,NumSubDivisions,Ns,Na,ex);
  return refinedA+refinedB;
}

double GetTotalIntegral(int i,int j,int k,int l,double a,double b,int MaxIter,int Iter,double relCrit,double absCrit,double NumSubDivisions,int Ns,int Na,exDFTstruct &ex){
  //begin USERINPUT
  double GaranteedIntervalCoverage = 0.2;
  //end USERINPUT
  double BaseLength = (b-a)/((double)MaxIter), aa = a, bb = a+BaseLength, Result = 0., prevResult = 0.;
  while(Iter<MaxIter){
    Result += GetPartialIntegral(i,j,k,l,aa,bb,MaxIter,0,1.0e+300,relCrit,absCrit,NumSubDivisions,Ns,Na,ex);
    if( Iter>(int)(GaranteedIntervalCoverage*(double)MaxIter) && (RelDiff(Result,prevResult)<relCrit || ABS(Result-prevResult)<absCrit/(double)MaxIter) ) break;
    else{ prevResult = Result; aa += BaseLength; bb += BaseLength; Iter++; if(Iter==MaxIter) cout << "GetTotalIntegral: MaxIter reached!" << endl; }
  }
  return Result;
}

void testHintIntegral2(exDFTstruct &ex){
  
  int orig = ex.settings[7];
  
//   ex.settings[7] = ...; GetHint(ex);
//   exportHint("mpDPFT_1pExDFT_Hint_loaded.dat", ex);
  ex.settings[7] = 1; GetHint(ex);
  exportHint("mpDPFT_1pExDFT_Hint_summation.dat", ex);
  ex.settings[7] = 4; GetHint(ex);
  exportHint("mpDPFT_1pExDFT_Hint_integration.dat", ex);
  
  ex.settings[7] = orig;
  
  
//   int MaxL = 10, Ns = 150, Na = 150;
//   double BaseLength = 10., relCrit = 3.0e-6, NumSubDivisions = 50., absCrit = 1.0e-10;
//   
//   ofstream testHintIntegralFile2;
//   testHintIntegralFile2.open("mpDPFT_testHintIntegral2.dat");
//   testHintIntegralFile2 << std::setprecision(16);  
//   
//   vector<vector<double>> IndexList(MaxL*MaxL*MaxL*MaxL), Output(MaxL*MaxL*MaxL*MaxL);
//   for(int i=0;i<Output.size();i++){ IndexList[i].resize(4); Output[i].resize(4); } //4 indices 
//   
//   #pragma omp parallel for schedule(dynamic)
//   for(int i=0;i<MaxL;i++){
//     for(int j=0;j<MaxL;j++){
//       for(int k=0;k<MaxL;k++){
// 	for(int l=0;l<MaxL;l++){
// 	  vector<int> indices = {{i,j,k,l}};
// 	  int sum = accumulate(indices.begin(),indices.end(),0);
// 	  double TotalIntegral = 0.;
// 	  if(sum%2==0 || sum==0){
// 	    int I = indices[0]; if(I>indices[1]){ indices[0] = indices[1]; indices[1] = I; }
// 	    I = indices[3]; if(I>indices[2]){ indices[3] = indices[2]; indices[2] = I; }
// 	    //TotalIntegral = GetTotalIntegral(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,UpperIntegrationLimit,MaxIter,0,PARAMS[p][0],PARAMS[p][1],PARAMS[p][2],Ns,Na,ex);
// 	  }
// 	}
//       }
//     }
//   }
//   
//   testHintIntegralFile2.close();
  
  cout << "testHintIntegral2() completed." << endl;
}

void testHintIntegral(exDFTstruct &ex){
  //generate test cases (random indices across level range and around maximum level (L~100) and for benchmarking against summation formula (L~20))
  //-> produce Hint for selection of crit, small, etc., track timing for each case
  //sort&printToFile increasing relerror
  //sort&printToFile increasing abserror
  //sort&printToFile increasing timing
  //report parameters for situations where ALL test cases have relerror<0.01 --> we target (on 1 thread) AverageTiming(TESTCASES) < 80threads*2weeks/(100^4matrixElements)=1sec/matrixElement

  //begin USERINPUT
  int L = ex.settings[0], L2 = L*L, L3 = L2*L, L4 = L2*L2;
  int MaxL = 100;
  int NumTestCases = /*10000,*/4*L4, NumParamChoices = 9;
  int ParamProgression = 4;// --- 0: random --- 1: linear --- >1: nested
  double UpperIntegrationLimit = 40., AcceptableRatio = 0.999, MaxRelDiff = 1.0e-3, AbsDiffThreshold = 1.0e-14, MaxTiming = 1.;
  int Ns = 150, Na = Ns, MaxIter = 10;
  //end USERINPUT
  
  ofstream testHintIntegralFile;
  testHintIntegralFile.open("mpDPFT_testHintIntegral.dat");
  testHintIntegralFile << std::setprecision(16);
  
  int NumSummationCases = 0;
  vector<vector<int>> TESTCASES;//{{i,j,k,l}}
  vector<int> testcase(4);
  
  int a = 0;
  double div = 0.25; if(NumTestCases<L4) div = 0.333;
  while(TESTCASES.size()<NumTestCases){
    if(NumTestCases>L4 && a<L4){//take all possible indices for summation formula
      if(a>90){ double expon = floor(log10((double)a)); for(int i=1;i<10;i++){ if(i*(int)pow(10.,expon)==a){ cout << "a=" << a << endl; break; } } }
      int b = a%L3;//b = a-i*L3
      int i = (a-b)/L3;
      int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
      int j = (a-i*L3-c)/L2;
      int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
      int k = (a-i*L3-j*L2-l)/L;
      testcase[0] = min(i,j);
      testcase[1] = max(i,j);
      testcase[2] = max(k,l);
      testcase[3] = min(k,l);
      a++;
    }
    else{
      double rn = ex.RNpos(ex.MTGEN);
      if(NumTestCases<L4 && rn<div) for(int i=0;i<4;i++) testcase[i] = (int)((double)L*ex.RNpos(ex.MTGEN));
      else if(rn<2.*div) for(int i=0;i<4;i++) testcase[i] = (int)(MaxL-0.2*(double)MaxL*ex.RNpos(ex.MTGEN));
      else for(int i=0;i<4;i++) testcase[i] = (int)(MaxL*ex.RNpos(ex.MTGEN));
    }
    int sum = accumulate(testcase.begin(),testcase.end(),0);
    if(sum%2==0 || sum==0){
      int I = testcase[0]; if(I>testcase[1]){ testcase[0] = testcase[1]; testcase[1] = I; }//then j>i
      I = testcase[3]; if(I>testcase[2]){ testcase[3] = testcase[2]; testcase[2] = I; }//then k>l  
      TESTCASES.push_back(testcase);
      bool HintSum = true; 
      for(int i=0;i<4;i++) if(testcase[i]>=L) HintSum = false;
      if(HintSum) NumSummationCases++;
    }
  }
  testHintIntegralFile << "NumSummationCases/NumTestCases = " << NumSummationCases << "/" << NumTestCases << endl;

  //normal integral
  double crit=1.0e-3;//...1.0e-8//subsequent bisectioning relative accuracy criterion
  double small=1.0e-3;//...1.0e-8//relative criterion when to neglect small conributions to integral from bisecting
  double subdiv=10.;//...1000.// integrand evaluations per bisection interval
  double infinity=10.;//..100.//initial guess of upper integration boundary
  double crit1=small/subdiv;//relative criterion for cut-off for stopping to increase upper integration boundary  
  
  vector<vector<double>> PARAMS;
  if(ParamProgression==0){
    PARAMS.resize(NumParamChoices);
    for(int p=0;p<NumParamChoices;p++){
      PARAMS[p].resize(5);
      PARAMS[p][0] = exp(-6-2.*ex.RNpos(ex.MTGEN));
      PARAMS[p][1] = exp(-6-2.*ex.RNpos(ex.MTGEN));
      PARAMS[p][2] = 40.+160.*ex.RNpos(ex.MTGEN);
      PARAMS[p][3] = 20.+20.*ex.RNpos(ex.MTGEN);
      PARAMS[p][4] = PARAMS[p][1]/PARAMS[p][2];  
    }
  }
  else if(ParamProgression==1){
    PARAMS.resize(NumParamChoices);
    for(int p=0;p<NumParamChoices;p++){
      PARAMS[p].resize(5);
      double pp = (double)p/((double)NumParamChoices);
      PARAMS[p][0] = exp(-7.-pp*6.);//crit - (crit-1.0e-8)*pp;
      PARAMS[p][1] = exp(-7.-pp*6.);//small - (small-1.0e-8)*pp;
      PARAMS[p][2] = subdiv + (400.-subdiv)*pp;
      PARAMS[p][3] = infinity + (100.-infinity)*pp;
      PARAMS[p][4] = PARAMS[p][1]/PARAMS[p][2];
    }
  }
  else if(ParamProgression==2){
    int num1 = max(2,(int)(pow(NumParamChoices,0.25)+0.1)), num2 = num1*num1, num3 = num2*num1;
    double P = (double)(num1-1);
    NumParamChoices = num3*num1;
    PARAMS.resize(NumParamChoices);
    vector<double> StartWith(4);
    StartWith[0] = -2.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[1] = -2.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[2] = 20.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[3] = 10.+0.01*ex.RNpos(ex.MTGEN);
    for(int p0=0;p0<num1;p0++){
      double pp0 = (double)p0/P;
      for(int p1=0;p1<num1;p1++){
	double pp1 = (double)p1/P;
	for(int p2=0;p2<num1;p2++){
	  double pp2 = (double)p2/P;
	  for(int p3=0;p3<num1;p3++){
	    double pp3 = (double)p3/P;
	    int p = p0*num3+p1*num2+p2*num1+p3;
	    PARAMS[p].resize(5);
	    PARAMS[p][0] = pow(10.,StartWith[0]-2.*pp0);
	    PARAMS[p][1] = pow(10.,StartWith[1]-2.*pp1);
	    PARAMS[p][2] = StartWith[2] + 180.*pp2;
	    PARAMS[p][3] = StartWith[3] + 40.*pp3;
	    PARAMS[p][4] = PARAMS[p][1]/PARAMS[p][2];
	  }
	}
      }
    }
  }  
  else if(ParamProgression==3){
    int num1 = max(2,(int)(pow(NumParamChoices,1./3.)+0.1)), num2 = num1*num1;
    double P = (double)(num1-1);
    NumParamChoices = num2*num1;
    PARAMS.resize(NumParamChoices);
    vector<double> StartWith(4);
    StartWith[0] = -5.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[1] = -10.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[2] = 100.+0.01*ex.RNpos(ex.MTGEN);
    StartWith[3] = 10.;//10. seems to be just fine
    for(int p0=0;p0<num1;p0++){
      double pp0 = (double)p0/P;
      for(int p1=0;p1<num1;p1++){
	double pp1 = (double)p1/P;
	for(int p2=0;p2<num1;p2++){
	  double pp2 = (double)p2/P;
	  int p = p0*num2+p1*num1+p2;
	  PARAMS[p].resize(5);
	  PARAMS[p][0] = pow(10.,StartWith[0]-1.*pp0);
	  PARAMS[p][1] = pow(10.,StartWith[1]-4.*pp1);
	  PARAMS[p][2] = StartWith[2] + 200.*pp2;
	  PARAMS[p][3] = StartWith[3];
	  PARAMS[p][4] = PARAMS[p][1]/PARAMS[p][2];
	}
      }
    }
  }    
  else if(ParamProgression==4){
    int num1 = max(2,(int)(sqrt(NumParamChoices)));
    double P = (double)(num1-1);
    NumParamChoices = num1*num1;
    PARAMS.resize(NumParamChoices);
    vector<double> StartWith(4);
    StartWith[0] = -4.+0.001;
    StartWith[1] = AbsDiffThreshold;//seems to be just fine
    StartWith[2] = 30.+0.001;
    StartWith[3] = 10.;//seems to be just fine
    for(int p0=0;p0<num1;p0++){
      double pp0 = (double)p0/P;
      for(int p1=0;p1<num1;p1++){
	double pp1 = (double)p1/P;
	int p = p0*num1+p1;
	PARAMS[p].resize(5);
	PARAMS[p][0] = pow(10.,StartWith[0]-4.*pp0);
	PARAMS[p][1] = StartWith[1];
	PARAMS[p][2] = StartWith[2] + 140.*pp1;
	PARAMS[p][3] = StartWith[3];
	PARAMS[p][4] = PARAMS[p][1]/PARAMS[p][2];//MIT: not used anymore
      }
    }
  } 
  
  GetHint(ex);//-->Hint[a], benchmarking against summation formula
  double prefactor = ex.params[0]; 
  vector<vector<double>> RESULTS(10); for(int r=0;r<RESULTS.size();r++) RESULTS[r].resize(NumTestCases*NumParamChoices);
  vector<double> avtiming(NumParamChoices); fill(avtiming.begin(),avtiming.end(),0.);
  #pragma omp parallel for schedule(dynamic)
  for(int p=0;p<NumParamChoices;p++){
    double maxRelDiff = 0., maxAbsDiff = 0.;
    double worstRel0 = 1.234567, worstRel1 = 1.234567, worstAbs0 = 1.234567, worstAbs1 = 1.234567, estimate = 1.234567, firstinterval = 1.234567;
    vector<int> testcaseRel(4), testcaseAbs(4);
    for(int t=0;t<NumTestCases;t++){
      double expon = floor(log10((double)t)); for(int i=1;i<10;i++){ if(t>90 && i*(int)pow(10.,expon)==t){ cout << "p=" << p << " t=" << t << endl; break; } }
      vector<double> result(RESULTS.size());
      chrono::high_resolution_clock::time_point Timer = high_resolution_clock::now();
      
//       double Estimate = 1.;//integral_div(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,PARAMS[p][3]*tailMultiplier,subdivRiemannEst,Ns,Na);
//       //double FirstInterval = integral_div(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,PARAMS[p][3],PARAMS[p][2],Ns,Na);//Jonah
//       double FirstInterval = JonahIntegral(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,PARAMS[p][3],PARAMS[p][0],PARAMS[p][2],PARAMS[p][1],Ns,Na,1.0e+100,Estimate);//MIT
//       result[1] = prefactor*2.*inf_integral(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],PARAMS[p][3],PARAMS[p][3],PARAMS[p][0],PARAMS[p][2],PARAMS[p][1],Ns,Na,PARAMS[p][4],PARAMS[p][1],FirstInterval,Estimate);
      
      //result[1] = JonahIntegral(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,PARAMS[p][3]*tailMultiplier,PARAMS[p][0],PARAMS[p][2],PARAMS[p][1],Ns,Na,1.0e+100,1.);
      result[1] = GetTotalIntegral(TESTCASES[t][0],TESTCASES[t][1],TESTCASES[t][2],TESTCASES[t][3],0.,UpperIntegrationLimit,MaxIter,0,PARAMS[p][0],PARAMS[p][1],PARAMS[p][2],Ns,Na,ex);
      
      chrono::high_resolution_clock::time_point endTimer = high_resolution_clock::now();
      duration<double>time_span = duration_cast<duration<double>>(endTimer - Timer);
      bool HintSum = true; for(int i=0;i<4;i++) if(TESTCASES[t][i]>=L) HintSum = false;
      if(HintSum) result[0] = Hint[TESTCASES[t][0]*L3+TESTCASES[t][1]*L2+TESTCASES[t][2]*L+TESTCASES[t][3]];
      else result[0] = result[1];
      result[2] = RelDiff(result[0],result[1]);
      result[3] = ABS(result[0]-result[1]);
      if(result[3]>maxAbsDiff){ maxAbsDiff = result[3]; worstAbs0 = result[0]; worstAbs1 = result[1]; testcaseAbs = TESTCASES[t]; }
      if(result[3]>AbsDiffThreshold && result[2]>maxRelDiff){ maxRelDiff = result[2]; worstRel0 = result[0]; worstRel1 = result[1]; testcaseRel = TESTCASES[t]; }
      result[4] = (double)time_span.count(); avtiming[p] += result[4];
      for(int r=0;r<5;r++) result[5+r] = PARAMS[p][r];
      for(int r=0;r<result.size();r++) RESULTS[r][t*NumParamChoices+p] = result[r];
    }
    avtiming[p] /= (double)NumTestCases;
    string str1 = "calc for parameters [" + to_string(p) + "]:  " + vec_to_str_with_precision(PARAMS[p],8) + " --- average timing = " + to_string_with_precision(avtiming[p],8) + "\n maxRelDiff = " + to_string_with_precision(maxRelDiff,8) + " for " + "  summation -> " + to_string_with_precision(worstRel0,8) + "  integration -> " + to_string_with_precision(worstRel1,8) + " @indices " + vec_to_str(testcaseRel) + "\n maxAbsDiff = " + to_string_with_precision(maxAbsDiff,8) + " for " + "  summation -> " + to_string_with_precision(worstAbs0,8) + "  integration -> " + to_string_with_precision(worstAbs1,8) + " @indices " + vec_to_str(testcaseAbs) + "\n";
    cout << endl << str1; cout.flush();
    testHintIntegralFile << str1;
  }
  
  //ToDo: print only for those parameter choices for which all matrix elements have relDiff<0.01 and time<1ms
  // colums --- 1: Hint[summation] --- 2: Hint[integral] --- 3: RelDiff --- 4: AbsDiff --- 5: timing --- 6: crit --- 7: small --- 8: subdiv --- 9: infinity --- 10: small/subdiv
  
/*  vector<vector<double>> RESULTSrelDiff(RESULTS.size()); for(int r=0;r<RESULTS.size();r++) RESULTSrelDiff[r].resize(NumTestCases*NumParamChoices);
  int count = 0;
  for(auto i: sort_indices(RESULTS[2])){
    //for(int r=0;r<RESULTS.size();r++) RESULTSrelDiff[r][count] = RESULTS[r][i];
    for(int r=0;r<RESULTS.size();r++){ RESULTSrelDiff[r][count] = RESULTS[r][i]; testHintIntegralFile << to_string_with_precision(RESULTS[r][i],16) << " "; } testHintIntegralFile << endl; if(count>100) break;
    count++;
  }
  vector<vector<double>> RESULTStime(RESULTS.size()); for(int r=0;r<RESULTS.size();r++) RESULTStime[r].resize(NumTestCases*NumParamChoices);
  testHintIntegralFile << endl;
  count = 0;
  for(auto i: sort_indices(RESULTS[4])){
    //for(int r=0;r<RESULTS.size();r++) RESULTStime[r][count] = RESULTS[r][i];
    for(int r=0;r<RESULTS.size();r++){ RESULTStime[r][count] = RESULTS[r][i]; testHintIntegralFile << to_string_with_precision(RESULTS[r][i],16) << " "; } testHintIntegralFile << endl; if(count>100) break;
    count++;
  }*/   
  
  vector<double> GoodParamsQ(NumParamChoices), AvRelDiff(NumParamChoices);
  vector<vector<double>> RelDiffVec(NumParamChoices);
  int count = 0;
  for(auto p: sort_indices(avtiming)){
    RelDiffVec[p].resize(NumTestCases); fill(RelDiffVec[p].begin(),RelDiffVec[p].end(),0.);
    cout << count << "/" << NumParamChoices-1 << " " << p << endl;
    GoodParamsQ[p] = (double)NumSummationCases;
    AvRelDiff[p] = 0.;
    int RelDiffCases = 0;
    for(int t=0;t<NumTestCases;t++){
      double reldiff = RESULTS[2][t*NumParamChoices+p];
      double absdiff = RESULTS[3][t*NumParamChoices+p];
      if(absdiff>AbsDiffThreshold){
	RelDiffCases++;
	RelDiffVec[p][t] = reldiff;
	AvRelDiff[p] += reldiff;
	if(reldiff>MaxRelDiff) GoodParamsQ[p]--;
      }
    }
    if(RelDiffCases>0) AvRelDiff[p] /= (double)RelDiffCases;
    if(GoodParamsQ[p]>AcceptableRatio*(double)NumSummationCases && avtiming[p]<MaxTiming){
      testHintIntegralFile << "p=" << p << ": Ratio(SuccessfullTestCases)=" << to_string_with_precision(GoodParamsQ[p]/((double)NumSummationCases),8) << " --- ";
      for(int r=5;r<RESULTS.size()-1;r++) testHintIntegralFile << to_string_with_precision(RESULTS[r][p],8) << " ";
      testHintIntegralFile << " --- MaxRelDiff = " << to_string_with_precision(*max_element(RelDiffVec[p].begin(),RelDiffVec[p].end()),8) << " --- AverageRelDiff = " << to_string_with_precision(AvRelDiff[p],8) << " --- AverageTiming = " << to_string_with_precision(avtiming[p],8) << endl;
    }
    count++;
  } 

  cout << "testHintIntegral done..." << endl;
  testHintIntegralFile << "testHintIntegral done..." << endl;
  testHintIntegralFile.close();
  usleep(100*1000000);
      
}

void testGetRho(exDFTstruct &ex){
  cout << "testGetRho:" << endl;
  int L = ex.settings[0], aMax = (int)1.0e+10;
  #pragma omp parallel for schedule(dynamic)
  for(int a=0;a<aMax;a++){
    vector<vector<double>> rho1p(L); for(int i=0;i<L;i++) rho1p[i].resize(L);
    vector<double> occ = InitializeOccNum(false,ex);
    GetRho1p(rho1p,occ,ex);
    bool rhoOK = rho1pConsistencyCheck(rho1p,occ,ex);
    double expon = floor(log10((double)a)); for(int i=1;i<10;i++){ if(i*(int)pow(10.,expon)==a){ cout << a << "/" << aMax << endl; break; } }
  } 
}

void testrho1pConsistency(exDFTstruct &ex){
  cout << "testrho1pConsistency:" << endl;
  int L = ex.settings[0];
  vector<vector<double>> rho1p(L); for(int i=0;i<L;i++) rho1p[i].resize(L);
  //vector<double> occ = {{1.980002082363983,1.21980267693668,0.008007776885512354,0.7409487629369488,0.4617101720367526,0.797846205341626,0.01118371344825275,0.537733476786234,0.1706896174923248,0.0384933156476226,0.0258218633759727,0,0,0.007760336748089795}};
  vector<double> occ = {{2.,1.265203257857272,0.707940930364663,0.5006295887528635,0.06449117263547401,0.7252799531785176,0.368139051978064,0.3535655438187404,0.,0.,0.,0.,0.009230699130033939,0.005519802284371681}};
  GetRho1p(rho1p,occ,ex);
  if(!rho1pConsistencyCheck(rho1p,occ,ex)) cout << "rho1p INCONSISTENT" << endl; else cout << "rho1p OK" << endl;
}

void GetHint(exDFTstruct &ex){
  cout << "1pEx-DFT: GetHint ";
  int L = ex.settings[0], L2 = L*L, L3 = L2*L, L4 = L2*L2;
  double prefactor = ex.params[0];
  Hint.clear(); Hint.resize(L4);
  
  if(ex.HintIndexThreshold>0.){
	Hint4DArray.clear(); Hint4DArray.resize(L);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<L;i++){
      Hint4DArray[i].resize(L);
      for(int j=0;j<L;j++){
		Hint4DArray[i][j].resize(L);
		for(int k=0;k<L;k++){
		  Hint4DArray[i][j][k].resize(L);
		}
      }
    }	
  }
  
	//read and amend one-column file
// 	Hint.resize(0);
// 	ReadVec("TabFunc_Hint.dat", Hint);
// 	#pragma omp parallel for schedule(dynamic)
//     for(int a=0;a<L4;a++) if(abs(Hint[a])<1.0e-12) Hint[a] = 0.;
// 	int Z = ex.params[1];
// 	exportHint("TabFunc_Hint_relativistic_hydrogenic_Tabcd_Z=" + to_string(Z) + "_L=" + to_string(L) + ".dat", ex);
// 	cout << " --- abort manually now!" << endl;

  if(ex.settings[2]==22){//1D harmonic interaction
    #pragma omp parallel for schedule(dynamic)
    for(int a=0;a<L4;a++){//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
      int b = a%L3;//b = a-i*L3
      int i = (a-b)/L3;
      int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
      int j = (a-i*L3-c)/L2;
      int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
      int k = (a-i*L3-j*L2-l)/L;
	  //beware of indices ordering, historically bad notation...
      if(ex.settings[13]==0) Hint[a] = prefactor*HarmonicInteractionElement(i,l,k,j);
      else if(ex.settings[13]==1) Hint[a] = -prefactor*0.5*HarmonicInteractionElement(i,j,k,l);
      else if(ex.settings[13]==2) Hint[a] = prefactor*(HarmonicInteractionElement(i,l,k,j)-0.5*HarmonicInteractionElement(i,j,k,l));  
      if(i==j && i==k && i==l) cout << Hint[a] << endl;
    }    
  }  
  else if(ex.settings[7]<0){//load from "TabFunc_Hint.dat"
    cout << " from TabFunc_Hint.dat with CompensationFactor = " << ex.CompensationFactor << endl;
    vector<double> LoadHint(0);
	ReadVec("TabFunc_Hint.dat", LoadHint);
	int loadL = (int)(pow((double)LoadHint.size(),0.25)+0.5), loadL2 = loadL*loadL, loadL3 = loadL2*loadL, loadL4 = loadL2*loadL2;
	cout << "LoadHint.size()=" << LoadHint.size() << " loadL=" << loadL << endl;
    vector<vector<vector<vector<double>>>> loadHint(loadL);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<loadL;i++){
      loadHint[i].resize(loadL);
      for(int j=0;j<loadL;j++){
		loadHint[i][j].resize(loadL);
		for(int k=0;k<loadL;k++){
		  loadHint[i][j][k].resize(loadL);
		  for(int l=0;l<loadL;l++){
			  loadHint[i][j][k][l] = ex.CompensationFactor*LoadHint[i*loadL3+j*loadL2+k*loadL+l];
			  if(i==j && i==k && i==l) cout << "loadHint = " << loadHint[i][j][k][l] << endl;
		  }
		}
      }
    }

	//if ex.settings[2]==24, but TabFunc_Hint.dat (i.e., loadHint[][][][]) holds the nonrelativistic Tabcd, then we produce the relativistic Tabcd from scratch if ex.settings[15]==1
	if(ex.settings[2]==24 && ex.settings[15]==1){
		if(loadL!=L) cout << "GetHint: Error!!!" << endl;
		#pragma omp parallel for schedule(dynamic)
		for(int a=0;a<loadL4;a++){
			Hint[a] = 0.;
			//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
			int b = a%L3;//b = a-i*L3
			int i = (a-b)/L3;
			int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
			int j = (a-i*L3-c)/L2;
			int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
			int k = (a-i*L3-j*L2-l)/L;
			for(int kappa=0;kappa<L;kappa++){
				for(int lambda=0;lambda<L;lambda++){
					for(int mu=0;mu<L;mu++){
						for(int nu=0;nu<L;nu++){
							Hint[a] += ex.CMatrixLoaded[kappa][i]*ex.CMatrixLoaded[lambda][j]*ex.CMatrixLoaded[mu][k]*ex.CMatrixLoaded[nu][l]*loadHint[kappa][lambda][mu][nu];
						}
					}
				}
			}
			if(ABS(Hint[a])<1.0e-12) Hint[a] = 0.;
			if(i==j && i==k && i==l) cout << i << " produce TabFunc_Hint_relativistic_hydrogenic_Tabcd = " << Hint[a] << endl;
		}
		int Z = ex.params[1];
		exportHint("TabFunc_Hint_relativistic_hydrogenic_Tabcd_Z=" + to_string(Z) + "_L=" + to_string(L) + ".dat", ex);
		cout << "Relativistic Tabcd produced from scratch as TabFunc_Hint_relativistic_hydrogenic_Tabcd_....dat --- abort manually now!" << endl;
		SleepForever();
	}
    
	
    if(ex.settings[2]==23 || ex.settings[2]==24){//3D Coulomb interaction, produced in mathematica
      cout << " ... and prefactor = " << prefactor << " Hint.size()=" << Hint.size() << " L4=" << L4 << " loadHint.size()^4=" << pow((double)loadHint.size(),4.) << endl;//prefactor = Z is the prefactor included in T^{NR}_abcd and thus remains in effect also for ex.settings[2]==24, which sums over T^{NR}
      #pragma omp parallel for schedule(dynamic)
      for(int a=0;a<L4;a++){//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
	int b = a%L3;//b = a-i*L3
	int i = (a-b)/L3;
	int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
	int j = (a-i*L3-c)/L2;
	int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
	int k = (a-i*L3-j*L2-l)/L;
	if(ex.settings[13]==0) Hint[a] = prefactor*loadHint[i][j][k][l];
	else if(ex.settings[13]==1) Hint[a] = -prefactor*0.5*loadHint[i][l][k][j];
	else if(ex.settings[13]==2) Hint[a] = prefactor*(loadHint[i][j][k][l]-0.5*loadHint[i][l][k][j]);
	//Hint[a] *= (1.-1.0e-2)+2.0e-2*ex.RNpos(ex.MTGEN);//add noise for testing purposes
	if(i==j && i==k && i==l) cout << "Hint = " << Hint[a] << endl;
      }        
    }
    else{
      #pragma omp parallel for schedule(dynamic)
      for(int i=0;i<L;i++) for(int j=0;j<L;j++) for(int k=0;k<L;k++) for(int l=0;l<L;l++){
	//int intArray[4] = {i, j, k, l}; sort(intArray, intArray + 4); Hint[i*L3+j*L2+k*L+l] = loadHint[intArray[0]*L3+intArray[1]*L2+intArray[2]*L+intArray[3]];//for Alex Hint file
	Hint[i*L3+j*L2+k*L+l] = loadHint[i][j][k][l];
	if(i==j && i==k && i==l) cout << "Hint = " << Hint[i*L3+j*L2+k*L+l] << endl;
      }
    }
  }
  else if(ex.settings[7]==0){//default (numerical) 
    cout << " (default)"  << endl;
    //ToDo
  }
  else if(ex.settings[2]==10 && ex.settings[7]>0){//1D contact
    //double tmpHint[L][L][L][L];
    vector<vector<vector<vector<double>>>> tmpHint(L);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<L;i++){
      tmpHint[i].resize(L);
      for(int j=0;j<L;j++){
	tmpHint[i][j].resize(L);
	for(int k=0;k<L;k++){
	  tmpHint[i][j][k].resize(L);
	}
      }
    }
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<L;i++){ for(int j=0;j<L;j++){ for(int k=0;k<L;k++){ for(int l=0;l<L;l++){ tmpHint[i][j][k][l] = /*(long double)*/0.; } } } }
    if(ex.settings[7]==1){//Jonah's summation
      cout << " (Jonah's summation)"  << endl;
      long double s0123=1.;
      long double s01xx=1.;
      long double syy23=0.5;
      long double sy123=0.5;
      long double s01yy=0.5;
      long double s01y3=0.5;
      for(int l0=0;l0<L;l0++){
	s0123=1.;//reset for alpha=alpha1=l0=l3=0
	for(int alpha=0;alpha<(L-l0);alpha=alpha+2){//alpha increases in steps of 2
	  for(int l3=0;l3<L;l3++){
            s01xx=s0123;//reset s01xx at alpha1=0
            for(int alpha1=0;alpha1<(L-l3);alpha1=alpha1+2){
	      tmpHint[l0][l0+alpha][l3+alpha1][l3]=(double)T(l0,alpha,alpha1,l3,s01xx);
              s01xx=alph_up(l3,alpha1,alpha,s01xx);//update s01xx
            }
	  }
	  s0123=alph_up(l0,alpha,0,s0123);//Update s0123 for increased alpha
	}
      }
      for(int l0=0;l0<L;l0++){
	syy23=sy123;//reset for alpha=alpha1=1,l0=l3=0
	for(int alpha=1;alpha<(L-l0);alpha=alpha+2){//alpha increases in steps of 2
	  s01y3=syy23;//reset s01y3
	  for(int l3=0;l3<L;l3++){
	    s01yy=s01y3;//reset s01xx at alpha1=1
	    for(int alpha1=1;alpha1<(L-l3);alpha1=alpha1+2){
	      tmpHint[l0][l0+alpha][l3+alpha1][l3]=(double)T(l0,alpha,alpha1,l3,s01yy);
	      s01yy=alph_up(l3,alpha1,alpha,s01yy);//update s01yy
	    }
	    s01y3=lamb_up(l3,1,s01y3);//update s01y3 at alpha1=1
	  }
	  syy23=alph_up(l0,alpha,1,syy23);//Update syy23 for increased alpha
	}
	sy123=lamb_up(l0, 1, sy123);//Update sy123 for increase lambda
      }  
    }
    else if(ex.settings[7]==2){//Jonah's integration
      cout << " (Jonah's integration)"  << endl;
      double T0=0;
      double T00=0.;
      double T02=0.;
      double T002=0.;
      int Ns=50;
      int Na=100;
      //normal integral
      double crit=1E-3;//subsequent bisectioning relative accuracy criterion
      double small=1E-3;//relative criterion when to neglect small conributions to integral from bisecting
      double subdiv=50.;// integrand evaluations per bisection interval
      double infinity=10.;//initial guess of upper integration boundary
      double crit1=small/subdiv;//relative criterion for cut-off for stopping to increase upper integration boundary
      //Square root integral
      double crit_2=1E-6;
      double small_2=1E-12;
      double subdiv_2=100000.;
      double infinity_2=100.;
      double crit1_2=1E-6;      
      #pragma omp parallel for schedule(dynamic)
      for(int l0=0;l0<L;l0++){
	//cout << "l0 = " << l0 << endl;
	for(int alpha=0;alpha<(L-l0);alpha=alpha+2){//alpha increases in steps of 2
	  for(int l3=0;l3<L;l3++){
            for(int alpha1=0;alpha1<(L-l3);alpha1=alpha1+2){
	      T0=integral_div(l0,l0+alpha,l3+alpha1,l3,0.,infinity,subdiv,Ns,Na);//reset T0
	      //T02=integral_div2(l0,l0+alpha,l3+alpha1,l3,0.,infinity_2,subdiv_2,Ns,Na);//reset T02
              T00=2.*inf_integral(l0,l0+alpha,l3+alpha1,l3,infinity,infinity,crit,subdiv,small,Ns,Na,crit1,small,T0,T0);//converge integral
              //T002=inf_integral2(l0,l0+alpha,l3+alpha1,l3,infinity_2,infinity_2,crit_2,subdiv_2,small_2,Ns,Na,crit1_2,small_2,T02,T02);
//               if(abs(T00/sum0)<0.99999 or abs(T00/sum0)>1.00001){//alert if we get >10e-5 discrepancy
//                     cout<<"discrepancy="<< std::setprecision(14) <<abs((T00-sum0)/T00)<< endl;//fractional discrepancy between sum and integral
//                     cout<<"T002="<<std::setprecision(14)<<T002<< endl;//3rd party check
//               }
	      tmpHint[l0][l0+alpha][l3+alpha1][l3]=T00;
	      cout << "tmpHint:[" << l0 << "][" << l0+alpha << "][" << l3+alpha1 << "][" << l3 << "] = " << tmpHint[l0][l0+alpha][l3+alpha1][l3] << endl;
            }
	  }
	}
	//cout << " check k=l=0: " << tmpHint[l0][l0][0][0] << endl; 
      }
      #pragma omp parallel for schedule(dynamic)
      for(int l0=0;l0<L;l0++){
	//cout << "l0 = " << l0 << endl;
	for(int alpha=1;alpha<(L-l0);alpha=alpha+2){
	  for(int l3=0;l3<L;l3++){
            for(int alpha1=1;alpha1<(L-l3);alpha1=alpha1+2){
	      T0=integral_div(l0,l0+alpha,l3+alpha1,l3,0.,infinity,subdiv,Ns,Na);//reset T0
	      //T02=integral_div2(l0,l0+alpha,l3+alpha1,l3,0.,infinity_2,subdiv_2,Ns,Na);//reset T02
              T00=2.*inf_integral(l0,l0+alpha,l3+alpha1,l3,infinity,infinity,crit,subdiv,small,Ns,Na,crit1,small,T0,T0);//converge integral
              //T002=inf_integral2(l0,l0+alpha,l3+alpha1,l3,infinity_2,infinity_2,crit_2,subdiv_2,small_2,Ns,Na,crit1_2,small_2,T02,T02);
//               if(abs(T00/sum0)<0.99999 or abs(T00/sum0)>1.00001){//alert if we get >10e-5 discrepancy
//                     cout<<"discrepancy="<< std::setprecision(14) <<abs((T00-sum0)/T00)<< endl;//fractional discrepancy between sum and integral
//                     cout<<"T002="<<std::setprecision(14)<<T002<< endl;//3rd party check
//               }
              tmpHint[l0][l0+alpha][l3+alpha1][l3]=T00;
            }
	  }
	}
	//cout << " check k=l=1: " << tmpHint[l0][l0][1][1] << endl; 
      }
    }
    else if(ex.settings[7]==3){//Jonah's recursion
      cout << " (Jonah's recursion)"  << endl;
int L0=4*L+1;
vector<vector<vector<vector<double>>>> I(L0);
    for(int i=0;i<L0;i++){
      I[i].resize(L0);
      for(int j=0;j<L0;j++){
	        I[i][j].resize(L0);
	         for(int k=0;k<L0;k++){
	            I[i][j][k].resize(L0);
	}
      }
    }
vector<vector<vector<vector<double>>>> TmpHint(L0);
      for(int i=0;i<L0;i++){
        TmpHint[i].resize(L0);
        for(int j=0;j<L0;j++){
  	        TmpHint[i][j].resize(L0);
  	         for(int k=0;k<L0;k++){
  	            TmpHint[i][j][k].resize(L0);
  	}
        }
      }
int parity=1;
I[0][0][0][0]=1;
for(int i=0;i+2<L0;i=i+2){//I[a][0][0][0],a<L
    I[i+2][0][0][0]=-(i+1)*I[i][0][0][0];
    I[i+1][1][0][0]=(i+1)*I[i][0][0][0];
    //std::cout <<i+2<<0<<0<<0 <<"="<<I[i+2][0][0][0] << '\n';
}
for(int l1=2;l1<L0;l1++){//I[a][b][0][0],a+b<L
  for(int l2=l1;l2+l1<L0;l2=l2+2){//l2>l1
    I[l2][l1][0][0]=l2*I[l2-1][l1-1][0][0]-(l1-1)*I[l2][l1-2][0][0];//Now we always have l2 >= l1
    std::cout << l2<<l1<<0<<0<< "="<<I[l2][l1][0][0] << '\n';
    I[l2][l1-1][1][0]=(l2)*I[l2-1][l1-1][0][0]+(l1-1)*I[l2][l1-2][0][0];//I[a][b][1][0]
    //std::cout << l2<<l1-1<<1<<0<< "="<<I[l2][l1-1][1][0] << '\n';
      //for(int alpha=0;alpha<L-l1-l2-l3;alpha=alpha+2){
        //I[l1][l2][l3][alpha+1]=}
  }
}//below, l3 is a variable(lambda)
for(int lambda=2;lambda+1<L0;lambda++){//I[a][b][c][0],a+b+c<L and b>a>c>0,
  for(int l1=lambda-1;l1+lambda<L0;l1++){
    parity=(lambda)%2;
    for(int l2=l1+parity;l1+l2+lambda<L0;l2=l2+2){//l1 starts from either 1 or 2
      if(l1>lambda-1){
        I[l2][l1][lambda][0]=l1*I[l2][max(l1-1,lambda-1)][min(l1-1,lambda-1)][0]+l2*I[max(l1,l2-1)][max(min(l1,l2-1),lambda-1)][min(min(l1,l2-1),lambda-1)][0]-(lambda-1)*I[l2][l1][lambda-2][0];
      }
      I[l2][l1][lambda-1][1]=l1*I[l2][max(l1-1,lambda-1)][min(l1-1,lambda-1)][0]+l2*I[max(l1,l2-1)][max(min(l1,l2-1),lambda-1)][min(min(l1,l2-1),lambda-1)][0]+(lambda-1)*I[l2][l1][lambda-2][0];
      //std::cout << l1<<l2<<lambda<<0<< "="<<I[l2][l1][lambda][0] << '\n';
      //std::cout << l1<<l2<<lambda-1<<1<< "="<<I[l2][l1][lambda-1][1] << '\n';
    }//for all c, a+b+c<L and all a<b
  }
}
for(int alpha=1;alpha+1<L0;alpha++){//I[a][b][c][d],a+b+c+d<L,b>a>c>d
  for(int l3=alpha+1;l3+alpha+1<L0;l3++){
    for(int l1=l3;l1+l3+alpha+1<L0;l1++){
      parity=(l3+alpha+1)%2;
      for(int l2=l1+parity;l1+l2+l3+alpha+1<L0;l2=l2+2){
        if(l1>0){
        I[l2][l1][l3][alpha+1]=l1*I[l2][max(l1-1,l3)][min(l3,l1-1)][alpha]+l2*I[max(l1,l2-1)][max(min(l1,l2-1),l3)][min(min(l1,l2-1),l3)][alpha]+l3*I[l2][l1][max(l3-1,alpha)][min(l3-1,alpha)]-alpha*I[l2][l1][l3][alpha-1];
        //std::cout << l1<<l2<<l3<<alpha+1<< "="<<I[l2][l1][l3][alpha+1] << '\n';
        }
      }
    }
  }
}
for(int i=0;i<L0;i++){
for(int j=i;j<L0-i;j++){
  for(int l=j;l<L0-j-i;l++){
    parity=(i+j)%2;
    for(int k=l+parity;k<L0-l-j-i;k=k+2){//i<j<l<k
    TmpHint[i][j][k][l]=0.5*sqrt(0.5/3.141592653589793)*(double)I[k][l][j][i]*exp(-0.5*(gsl_sf_lnfact(i)+gsl_sf_lnfact(j)+gsl_sf_lnfact(k)+gsl_sf_lnfact(l)))/(double)(pow(2,(i+j+k+l)/2));
    //std::cout << i<<j<<k<<l << '\n';
    //std::cout << "TmpHint="<<TmpHint[i][j][k][l] << '\n';
    }
  }
}
}//end result produces H_lambda1_lambda2_lambda3_lambda4 directly, but lambdas in ascending order.



    #pragma omp parallel for schedule(dynamic)
    for(int a=0;a<L*L*L*L;a++){//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
      int b = a%L3;//b = a-i*L3
      int i = (a-b)/L3;
      int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
      int j = (a-i*L3-c)/L2;
      int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
      int k = (a-i*L3-j*L2-l)/L;
      //sorting
      int intArray[4] = {i, j, k, l};
      sort(intArray, intArray + 4);
      tmpHint[i][j][k][l]=2.*TmpHint[intArray[0]][intArray[1]][intArray[3]][intArray[2]];
    }

    }
    else if(ex.settings[7]==4){//MIT integration
      cout << "MIT integration " << endl;
      prefactor = 1.;
      double LowerIntegrationLimit = 0.;
      double UpperIntegrationLimit = 40.;
      double relCrit = 1.0e-10;
      double absCrit = 1.0e-16;
      double NumSubdivisions = 100.;
      int MaxIter = 10;
      int Ns = 150;
      int Na = 150;
      #pragma omp parallel for schedule(dynamic)
      for(int a=0;a<L4;a++){//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
	int b = a%L3;//b = a-i*L3
	int i = (a-b)/L3;
	int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
	int j = (a-i*L3-c)/L2;
	int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
	int k = (a-i*L3-j*L2-l)/L;
	int sum = i+j+k+l;
	if(sum%2==0 || sum==0){
	  int l0=min(i,j);
	  int l1=max(i,j);
	  int l2=max(k,l);
	  int l3=min(k,l);
	  tmpHint[l0][l1][l2][l3] = GetTotalIntegral(l0,l1,l2,l3,LowerIntegrationLimit,UpperIntegrationLimit,MaxIter,0,relCrit,absCrit,NumSubdivisions,Ns,Na,ex);
	}
	double expon = floor(log10((double)a)); for(int i=1;i<10;i++){ if(i*(int)pow(10.,expon)==a){ cout << a << "/" << L4 << endl; break; } }
      }
//       #pragma omp parallel for schedule(dynamic)
//       for(int i=0;i<L;i++){
// 	for(int j=0;j<L;j++){
// 	  for(int k=0;k<L;k++){
// 	    for(int l=0;l<L;l++){
// 	      //cout << i << " " << j << " " << k << " " << l << " "; cout.flush();
// 	      int l0 = i, l1 = j, l2 = k, l3 = l;
// 	      int sum = i+j+k+l;
// 	      if(sum%2==0 || sum==0){
// 		int I = l0; if(I>l1){ l0 = l1; l1 = I; }//then j>i
// 		I = l3; if(I>l2){ l3 = l2; l2 = I; }//then k>l
// 		tmpHint[l0][l1][l2][l3] = GetTotalIntegral(l0,l1,l2,l3,LowerIntegrationLimit,UpperIntegrationLimit,MaxIter,0,relCrit,absCrit,NumSubdivisions,Ns,Na,ex);
// 	      }
// 	      //cout << tmpHint[l0][l1][l2][l3] << endl;
// 	    }
// 	  }
// 	  cout << "(i,j)=(" << i << "," << j << ") completed" << endl;
// 	}
//       }
    }
  
    #pragma omp parallel for schedule(dynamic)
    for(int a=0;a<L4;a++){//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
      int b = a%L3;//b = a-i*L3
      int i = (a-b)/L3;
      int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
      int j = (a-i*L3-c)/L2;
      int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
      int k = (a-i*L3-j*L2-l)/L;
      int l0=min(i,j);
      int l1=max(i,j);
      int l2=max(k,l);
      int l3=min(k,l); 
      //*(ex.ptrHint + a) = prefactor*((double)tmpHint[l0][l1][l2][l3]-0.5*(double)tmpHint[min(l1,l2)][max(l1,l2)][max(l3,l0)][min(l3,l0)]); if(i==j && i==k && i==l) cout << *(ex.ptrHint + a) << endl;
      //Hint[a] = prefactor*((double)tmpHint[l0][l1][l2][l3]-0.5*(double)tmpHint[min(l1,l2)][max(l1,l2)][max(l3,l0)][min(l3,l0)]);
      if(ex.settings[13]==0) Hint[a] = prefactor*(double)tmpHint[l0][l1][l2][l3];
      else if(ex.settings[13]==1) Hint[a] = -prefactor*(double)tmpHint[min(l1,l2)][max(l1,l2)][max(l3,l0)][min(l3,l0)];
      else if(ex.settings[13]==2) Hint[a] = prefactor*((double)tmpHint[l0][l1][l2][l3]-0.5*(double)tmpHint[min(l1,l2)][max(l1,l2)][max(l3,l0)][min(l3,l0)]);      
      if(i==j && i==k && i==l) cout << Hint[a] << endl;
    }
    
  }
  
  if(ex.settings[7]<0) cout << " --- Hint loaded." << endl;
  else cout << " --- Hint produced." << endl;

  //store indices with nonvanishing entries of Hint
  if(ex.HintIndexThreshold>0.){
	  //int co = 0;
	  for(int a=0;a<L4;a++){
		//a = i*L3+b = i*L3+j*L2+c = i*L3+j*L2+k*L+l
		int b = a%L3;//b = a-i*L3
		int i = (a-b)/L3;
		int c = (a-i*L3)%L2;//a-i*L3 = a' = j*L2+c
		int j = (a-i*L3-c)/L2;
		int l = (a-i*L3-j*L2)%L;//a-i*L3-j*L2 = a'' = k*L+l
		int k = (a-i*L3-j*L2-l)/L;		
		Hint4DArray[i][j][k][l] = Hint[a];
		if(abs(Hint[a])>ex.HintIndexThreshold){
			vector<int> hintindexvec(0);
			hintindexvec.push_back(i);
			hintindexvec.push_back(j);
			hintindexvec.push_back(k);
			hintindexvec.push_back(l);
			HintIndexVec.push_back(hintindexvec);
			  //cout << co << " " << HintIndexVec[co] << " " << to_string_with_precision(Hint[a]/prefactor,16) << endl;
			  //co++;
		}
		//else cout << " ..." << a << " " << to_string_with_precision(Hint[a]/prefactor,16) << endl;
		//if(co==200) SleepForever();
	  }
	  cout << " --- " << HintIndexVec.size() << " HintIndexVec with nonvanishing [abs(Hint)>" << ex.HintIndexThreshold << "] interaction tensor elements determined." << endl;
  }
  else cout << " --- use full Hint." << endl;
  
}

double HarmonicInteractionElement(int i, int j, int k, int l){  
  return HIE(2,i,l)*Kd(j,k)-2.*HIE(1,i,l)*HIE(1,j,k) + HIE(2,j,k)*Kd(i,l);
}

double HIE(int a, int i, int l){
  if(a==1){
    if(SameParityQ(i,l)) return 0.;
    else{
      int M = (int)(max((double)i,(double)l)+0.1), m = (int)(min((double)i,(double)l)+0.1);
      return sqrt(0.5*(double)M)*Kd(m,M-1);
    }
  }
  else if(a==2){
    if(!SameParityQ(i,l)) return 0.;
    else if(i==l) return 0.5*(2.*(double)i+1.);
    else{
      int M = (int)(max((double)i,(double)l)+0.1), m = (int)(min((double)i,(double)l)+0.1);
      return 0.5*sqrt((double)(M*(M-1)))*Kd(M-2,m);
    }
  }
  else return Kd(i,l);//for a==0
}

void exportHint(string filename, exDFTstruct &ex){
  ofstream HintFile;
  HintFile.open(filename);
  HintFile << std::setprecision(16);
  int L = ex.L, L2 = L*L, L3 = L2*L;
  for(int i=0;i<L;i++){
    for(int j=0;j<L;j++){
      for(int k=0;k<L;k++){
	for(int l=0;l<L;l++){
	  //HintFile << (*(ex.ptrHint + (i*L3+j*L2+k*L+l))) << endl;
	  HintFile << Hint[i*L3+j*L2+k*L+l] << endl;
	}
      } 
    }
  }
  HintFile.close();
  cout << "Hint exported to " << filename << " in row-major order." << endl;
}

bool varianceConvergedQ(vector<double> Vec, int testlength, double crit){
  if(Vec.size()<testlength) return false;
  else{
    int n = testlength/2;
    vector<double> vec1(n), vec2(n); for(int i=0;i<n;i++){ vec1[i] = Vec[Vec.size()-2*n+i]; vec2[i] = Vec[Vec.size()-n+i]; }
    double mean1 = VecAv(vec1), mean2 = VecAv(vec2), CurrentVariance1 = 0., CurrentVariance2 = 0.;
    for(int i=0;i<n;i++){ CurrentVariance1 += (vec1[i]-mean1)*(vec1[i]-mean1)/((double)n); CurrentVariance2 += (vec2[i]-mean2)*(vec2[i]-mean2)/((double)n); }
    if(sqrt(CurrentVariance1+CurrentVariance2)<crit){ /*cout << "SH variance converged" << endl;*/ return true; }
    else if(abs(ASYM(CurrentVariance1,CurrentVariance2))<crit){ /*cout << "SH variance stalled" << endl;*/ return true; }
    else return false;
  }
}

MatrixXd orthogonal_matrix(bool randomSeedQ, exDFTstruct &ex){
  int L = ex.L, count = 0, maxCount = 10;
  MatrixXd mat1(L,L);
  double MinimalOffDiagonalElement = 0.;
  while(MinimalOffDiagonalElement<0.01){
    MatrixXd mat0 = MatrixXd::Zero(L,L);
    for(int i=0;i<L;i++){
      if(randomSeedQ){ if(ex.RNpos(ex.MTGEN)<0.5) mat0(i,i) = -1.; else mat0(i,i) = 1.; }
      else{ if(i<L/2) mat0(i,i) = -1.; else mat0(i,i) = 1.; }
    }
    //cout << "get rotation matrix" << endl;
    MatrixXd rot = rotation_matrix(randomSeedQ,ex);
    mat1 = rot.transpose()*mat0*rot;
    MinimalOffDiagonalElement = mat1(0,0); for(int i=0;i<L;i++) for(int j=0;j<L;j++) if(i!=j && abs(mat1(i,j))<MinimalOffDiagonalElement) MinimalOffDiagonalElement = abs(mat1(i,j));
    count++;
    if(count==maxCount) break;
  }
  return mat1;
}

MatrixXd rotation_matrix(bool randomSeedQ, exDFTstruct &ex){
  int L = ex.L;
  MatrixXd A(L,L), M(L,L);
  const VectorXd ones(VectorXd::Ones(L));
  for(int i=0;i<L;++i) for(int j=0;j<L;++j){ if(randomSeedQ) A(i,j) = ex.RNpos(ex.MTGEN); else A(i,j) = 1.-(double)(i*L+j+1)/((double)(L*L+2)); }
    
  //cout << "get QR-decomposition of A" << endl;
  const HouseholderQR<MatrixXd> qr(A);
  const MatrixXd Q = qr.householderQ();
  
  M = Q * (qr.matrixQR().diagonal().array()<0).select(-ones,ones).asDiagonal();
  if(M.determinant()<0.) for(int i=0;i<L;++i) M(i,0) = -M(i,0);

  return M;
}

double MinimizeEint(vector<vector<double>> &rho1p, exDFTstruct &ex){
	cout << "MinimizeEint:" << endl;
    OPTstruct opt;
    SetDefaultOPTparams(opt);
    opt.ex = ex;
    opt.ex.rho1p = rho1p;
    opt.printQ = 0;
    opt.function = -11;
    opt.D = opt.ex.L;
    opt.epsf = opt.ex.RelAcc*opt.ex.UNITenergy;
    opt.ex.settings[4] = 1; opt.SearchSpaceMin = -3.141592653589793; opt.SearchSpaceMax = 3.141592653589793;     
    
    if(ex.settings[2]==0) return 0.;//noninteracting
    else if(ex.settings[10]==0){//set phases to zero, no minimization
      if(Norm(opt.ex.ZeroVec)>1.0e-12){ cout << "MinimizeEint ERROR" << endl; usleep(1000000); }
      return Eint_1pExDFT(opt.ex.ZeroVec,opt.ex);
    }
    else if(ex.Optimizers[1]==100){//CGD
      SetDefaultCGDparams(opt);
      //opt.printQ = 1;
      CGD(opt);
      if(ex.Optimizers[0]==-100) ex.Phases = opt.cgd.bestx;
      return opt.cgd.bestf;
    }  
    else if(ex.Optimizers[1]==101){//PSO 
      SetDefaultPSOparams(opt);
      //opt.printQ = 1;
      PSO(opt);
      if(ex.Optimizers[0]==-100) ex.Phases = opt.pso.bestx;
      return opt.pso.bestf;
    }
    else return 0.;
}

double Eint_1pExDFT(vector<double> var, exDFTstruct &ex){
  int L = ex.L, L2 = L*L, L3 = L2*L;
  double N = ex.Abundances[0];
  double OptSum = 0.;

  for(int i=0;i<L;i++){
    for(int j=0;j<L;j++){
      for(int k=0;k<L;k++){
	for(int l=0;l<L;l++){
	  //OptSum += ex.rho1p[i][j]*ex.rho1p[k][l]*cos(var[i]-var[j]+var[k]-var[l]) * (*(ex.ptrHint + (i*L3+j*L2+k*L+l)));
	  OptSum += ex.rho1p[i][j]*ex.rho1p[k][l]*cos(var[i]-var[j]+var[k]-var[l]) * Hint[i*L3+j*L2+k*L+l];
	}
      } 
    }
  }

  return 0.5*N*N*OptSum;
}    

double Etot_1pExDFT(vector<double> var, exDFTstruct &ex){
  double result = 0., TmpNum = accumulate(var.begin(),var.end(),0.0);
  int L = ex.settings[0];
  vector<vector<double>> rho1p(L);
  for(int l=0;l<L;l++){
    result += var[l]*Energy1pEx(l,ex);
    rho1p[l].resize(L);
  }
//   typedef chrono::high_resolution_clock Time;
//   typedef chrono::duration<float> fsec;
//   auto t0 = Time::now(), t1 = Time::now();
  GetRho1p(rho1p,var,ex);
//   t1 = Time::now(); fsec floatsec = t1 - t0; cout << "Timing GetRho1p: " << to_string(floatsec.count()) << endl;
//   t0 = Time::now();
  result += MinimizeEint(rho1p,ex);
//   t1 = Time::now(); floatsec = t1 - t0; cout << "Timing MinimizeEint: " << to_string(floatsec.count()) << endl; usleep(1000000);
  
  double penalty = 0.;
  if(ex.settings[6]==1){//works so far... 
    double DiffN = ABS(TmpNum-ex.Abundances[0]), DiffE = ABS(Energy1pEx(L-1,ex)-Energy1pEx(L-2,ex));
    penalty = 10.*DiffE*pow(DiffN/(ex.Abundances[0]*ex.RelAcc),2.001);
  }

  return result + penalty;  
}

double Eint_1pExDFT_CombinedMin(vector<double> var, vector<vector<double>> &rho1p, vector<vector<double>> &rho1pIm, exDFTstruct &ex){
  int L = ex.L, L2 = L*L, L3 = L2*L;
  double N = ex.Abundances[0];
  double OptSum = 0.;

  //return 0.;//uncomment for testing optimizer speed with closed-formula E_int[n]
  
  if(ex.HintIndexThreshold==0.){
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			for(int k=0;k<L;k++){
				for(int l=0;l<L;l++){
					OptSum += rho1p[i][j]*rho1p[k][l]*cos(var[i]-var[j]+var[k]-var[l]) * Hint[i*L3+j*L2+k*L+l];
				}
			} 
		}
	}
  }
  else if(ex.settings[20]>0) for(int A=0;A<HintIndexVec.size();A++){
	  int a = HintIndexVec[A][0], b = HintIndexVec[A][1], c = HintIndexVec[A][2], d = HintIndexVec[A][3];
	  double RDMproduct = rho1p[a][b]*rho1p[c][d] - rho1pIm[a][b]*rho1pIm[c][d];
	  OptSum += RDMproduct*cos(var[a]-var[b]+var[c]-var[d]) * Hint4DArray[a][b][c][d];
  }
  else for(int A=0;A<HintIndexVec.size();A++){
	  int a = HintIndexVec[A][0], b = HintIndexVec[A][1], c = HintIndexVec[A][2], d = HintIndexVec[A][3];
	  double RDMproduct = rho1p[a][b]*rho1p[c][d];
	  OptSum += RDMproduct*cos(var[a]-var[b]+var[c]-var[d]) * Hint4DArray[a][b][c][d];
  }

  return 0.5*N*N*OptSum;
}

double Etot_1pExDFT_CombinedMin(vector<double> AllAngles, exDFTstruct &ex){
  double E1p = 0.;
  int L = ex.settings[0], NumPairs = (L*L-L)/2;
  vector<vector<double>> rho1p(L), rho1pIm(L);
  vector<double> phases(L), occnum(L), unitaries(NumPairs), extraphases(2*NumPairs);
  
  for(int l=0;l<L;l++){
    E1p += (1.+cos(AllAngles[l]))*Energy1pEx(l,ex);
    rho1p[l].resize(L);
	rho1pIm[l].resize(L);
    occnum[l] = AllAngles[l];
    phases[l] = AllAngles[L+l];
  }
  //cout << "Etot_1pExDFT_CombinedMin: AnglesToOccNum(occnum,ex) = " << vec_to_str(AnglesToOccNum(occnum,ex)) << endl;
  
  GetRho1p(rho1p,AnglesToOccNum(occnum,ex),ex);
  if(ex.settings[20]==1){
	  for(int l=0;l<NumPairs;l++) unitaries[l] = AllAngles[2*L+l];
	  for(int l=0;l<2*NumPairs;l++) extraphases[l] = AllAngles[2*L+NumPairs+l];
	  TransformRho1p(rho1p,rho1pIm,unitaries,extraphases,ex);
  }
  if(ex.settings[10]==0){ phases = ex.ZeroVec; if(Norm(phases)>1.0e-12){ cout << "Etot_1pExDFT_CombinedMin ERROR" << endl; usleep(1000000); } }
  
  //phases = ex.ZeroVec;//comment out!!!
  double Eint = Eint_1pExDFT_CombinedMin(phases,rho1p,rho1pIm,ex);

  return E1p + Eint;  
}

void EXprint(int printQ, string str, exDFTstruct &ex){
  if(printQ==0 && ex.settings[16]==1) ex.control += str + "\n";
  else if(printQ>0){
    cout << std::setprecision(16) << str << endl;
    if(printQ==2) cout.flush();
    if(ex.settings[16]==1) ex.control += str + "\n";
  }
}
