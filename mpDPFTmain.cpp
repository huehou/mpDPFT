// This is mpDPFT (multi-purpose Density-Potential Functional Theory)
// Author: Martin-Isbjörn Trappe (martin.trappe@quantumlah.org)
// with contributions from: Jun Hao Hue (Plugin_Triangulation.cpp & Plugin_KD.cpp), Thanh Tri Chau (Plugin_KD.cpp), and Jonathan Wei Zhong Lau (dipolar interaction potential in 3D)

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <stdio.h>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include <stdlib.h>
//#include <math.h>//possible conflict with cmath
#include <omp.h>
#include <string>
#include <vector>
#include <gsl/gsl_sf_gamma.h>
#include <fftw3.h>
#include "MPDPFT_HEADER_ONEPEXDFT.h"
#include "mpDPFT.h"
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
using namespace alglib;

typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> fsec;


int main(int argc, char** argv){
  
  auto T0 = Time::now();
  
  taskstruct task;
  datastruct tmpdata;
  GetInputParameters(task,tmpdata);
  task.ParameterCount = tmpdata.TaskParameterCount;
  task.Type = tmpdata.TaskType;
  task.HyperBox = GetTaskHyperBox(tmpdata);
  task.count.resize(1);
  string built_YYYYMMDD = "built " + YYYYMMDD();

  TASKPRINT("\n ******************************************************************",task,1);
  TASKPRINT(" ***************************   mpDPFT   ***************************",task,1);  //"multi-purpose Density-Potential Functional Theory"
  TASKPRINT(" *************************    (v0.2.1)    *************************",task,1);
  TASKPRINT(" ***********************   " + built_YYYYMMDD + "   ***********************",task,1);
  TASKPRINT(" ******************************************************************\n",task,1);    
  
  // T A S K   ( O U T E R   P A R A M E T E R   L O O P )
  
  string FileName = "mpDPFT_ControlTask.dat";
  task.ControlTaskFile.open (FileName.c_str());  
  
  vector<double> res; res.clear();

  omp_set_nested(1);
  string max_threads_str = to_string(omp_get_max_threads());
  setenv("SUNW_MP_MAX_POOL_THREADS", max_threads_str.c_str(), 1);
  setenv("SUNW_MP_MAX_NESTED_LEVELS", "2", 1);

  if(task.Type<2){
    if(task.Type==0){ task.lastQ = true; task.ParameterCount = 1; }
    for(task.count[0]=0;task.count[0]<task.ParameterCount;task.count[0]++){
      if(task.count[0]==task.ParameterCount-1) task.lastQ = true;
      cout << " ***** Get data *****" << endl;  
      datastruct data; 
      task.dataset.push_back(datastruct()); 
      GetData(task,data);    
    }
  }
  else if(task.Type==2){
    //task.dataset.push_back(datastruct());
    GetGlobalMinimum_muVec(task);
  }
  else if(task.Type==3){
    //task.dataset.push_back(datastruct());
    //task = GetGlobalMinimum_Abundances(task);
    GetGlobalMinimum_Abundances(task);
    cout << "GetGlobalMinimum_Abundances done" << endl;
    cout << "task.Type==" << task.Type << endl;
    cout << task.controltaskfile.str() << endl;
    TASKPRINT(" abcd",task,1);
  }
  else if(task.Type==4 || task.Type==44){
    GetAuxilliaryFit(task);//store FitParameters in task.VEC
    datastruct data;//recompute best fit
    task.lastQ = true;
    task.Type = 444;
    data.VEC.resize(task.VEC.size()); data.VEC = task.VEC;
    TASKPRINT("task.Type==44 -> task.Type==444 with fit parameters task.VEC = " + vec_to_CommaSeparatedString_with_precision(data.VEC,16) + "\n",task,1);
    GetData(task,data);
    TASKPRINT("recomputed FigureOfMerit = " + to_string(GetFigureOfMerit(data)) + "\n",task,1);
  }
  else if(task.Type==5 || task.Type==6 || task.Type==7){
    if(task.Type==5) task.VEC.resize(2);
    else if(task.Type==6) task.VEC.resize(3);
    else if(task.Type==7) task.VEC.resize(1);
    task.count[0] = 0;
    for(int tc1 = 0;tc1<task.ParameterCount;tc1++){
      task.VEC[0] = task.HyperBox[0][0] + (task.HyperBox[0][1]-task.HyperBox[0][0]) * (double)tc1/((double)task.ParameterCount-1.);
      if(task.Type==7){
	    if(tc1==task.ParameterCount-1) task.lastQ = true;
	    res.clear();
	    task.count[0]++;
	    cout << " ***** Get data *****" << endl;
	    datastruct data; 
	    task.dataset.push_back(datastruct()); 
	    GetData(task,data);
	    res.push_back(data.FigureOfMerit); res.push_back(task.VEC[0]);
	    task.ParameterExplorationContainer.push_back(res);
      }
      else{
	for(int tc2 = 0;tc2<task.ParameterCount;tc2++){
	  task.VEC[1] = task.HyperBox[1][0] + (task.HyperBox[1][1]-task.HyperBox[1][0]) * (double)tc2/((double)task.ParameterCount-1.);
	  if(task.Type==5){
	    if(tc2==task.ParameterCount-1) task.lastQ = true;
	    res.clear();
	    task.count[0]++;
	    cout << " ***** Get data *****" << endl;
	    datastruct data; 
	    task.dataset.push_back(datastruct()); 
	    GetData(task,data);
	    res.push_back(data.FigureOfMerit); res.push_back(task.VEC[0]); res.push_back(task.VEC[1]);
	    task.ParameterExplorationContainer.push_back(res);
	  
	  }
	  else if(task.Type==6){
	    for(int tc3 = 0;tc3<task.ParameterCount;tc3++){
	      if(tc3==task.ParameterCount-1) task.lastQ = true;
	      res.clear();
	      task.count[0]++;
	      task.VEC[2] = task.HyperBox[2][0] + (task.HyperBox[2][1]-task.HyperBox[2][0]) * (double)tc3/((double)task.ParameterCount-1.);
	      cout << " ***** Get data *****" << endl;
	      datastruct data; 
	      task.dataset.push_back(datastruct()); 
	      GetData(task,data);
	      res.push_back(data.FigureOfMerit); res.push_back(task.VEC[0]); res.push_back(task.VEC[1]); res.push_back(task.VEC[2]);
	      task.ParameterExplorationContainer.push_back(res);
	    }
	  }
	}
      }
    }
  }
  else if(task.Type==8 || task.Type==88){
    for(task.count[0]=0;task.count[0]<task.ParameterCount;task.count[0]++){
      //cout << " ***** task: " << task.count[0] << " " << task.ParameterCount << " *****" << endl;
      if(tmpdata.Print==2) task.lastQ = true;
      datastruct data; 
      task.dataset.push_back(datastruct());
      cout << " ***** Get data *****" << endl;
      GetData(task,data);
      task.VEC.clear();
      for(int s=0;s<data.S;s++) task.VEC.push_back(data.muVec[s]);
      if(data.DensityExpression==5) for(int s=0;s<data.S;s++) for(int a=0;a<data.MppVec[s].size();a++) task.VEC.push_back(data.MppVec[s][a]);
      task.Aux = data.DeltamuModifier;
      cout << "continue with task.VEC = " << vec_to_str(task.VEC) << endl;
    }
  }
  else if(task.Type==9){
    task.count.resize(4); fill(task.count.begin(), task.count.end(), 1);
    GetGlobalMinimum_Abundances_AnnealedGradientDescent(task);
  }
  else if(task.Type==10){//TaskParameterCount repetitions -> then, repeat with InterpolVQ of lowest Etot
    double minEtot;
    if(task.ParameterCount<1) task.ParameterCount = 1;
    for(int tc1=0;tc1<=task.ParameterCount;tc1++){
      if(tc1==task.ParameterCount || task.ABORT){ task.lastQ = true; task.ParameterCount = tc1;}
      task.count[0]++;
      cout << " ***** Get data *****" << endl;
      datastruct data; 
      task.dataset.push_back(datastruct()); 
      GetData(task,data);
      res.push_back(data.Etot); sort(res.begin(),res.end()); 
      if(tc1==0 || data.Etot<minEtot){
	minEtot = data.Etot; task.V = data.V;
	task.VEC.clear(); task.VEC.resize(data.S); task.VEC = data.muVec;
	TASKPRINT(" ***** task.Type==10: new minEtot = " + to_string_with_precision(minEtot,16) + " @ task.count = " + to_string(tc1),task,1);
	TASKPRINT("\n       Etot history (sorted) = " + vec_to_str(res) + "\n",task,1);
      }
      else if(data.Etot>=minEtot) TASKPRINT(" ***** task.Type==10: no improvement in minEtot @ task.count = " + to_string(tc1) + "\n       Etot history (sorted) = " + vec_to_str(res) + "\n",task,1);
    }
  }
  else if(task.Type==61){
    for(task.DynDFTe.mode=0;task.DynDFTe.mode<=task.ParameterCount;task.DynDFTe.mode++){
      datastruct data; 
      task.dataset.push_back(datastruct());
      cout << " ***** Get data for task.DynDFTe.mode = " << task.DynDFTe.mode << " *****" << endl;
      GetData(task,data);
    }
  }  
  else if(task.Type==100){//default execution of 1p-exact DFT, in conjunction with data.System>=100
    task.lastQ = true; task.ParameterCount = 1;
    cout << " ***** Get data *****" << endl;  
    datastruct data; 
    task.dataset.push_back(datastruct()); 
    GetData(task,data);
  }  
  TASKPRINT(" ***** Process task data *****",task,1);
  ProcessTASK(task);
  
  
  // E N D   P R O G R A M  
  
  auto T1 = Time::now(); fsec floatsec = T1 - T0;
  TASKPRINT(" Total Program Run Time: " + to_string((int)floatsec.count()) + " seconds",task,1);
  TASKPRINT(" EndOfProgram",task,1);
  TASKPRINT(" post-processing shell scripts...\n\n *****************************************************************",task,1);
  task.ControlTaskFile << task.controltaskfile.str();
  task.ControlTaskFile.close();

  return 0;
}
