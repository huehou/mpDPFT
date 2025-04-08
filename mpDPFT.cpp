//#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
//#include <tr1/cmath>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <regex>
#include <chrono>
#include <functional>
#include <limits>
#include <numeric>
#include <cstdlib>  // For system()
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
//#include <math.h>//possible conflict with cmath
#include <algorithm>
#include <set>
#include <omp.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <array>
#include <thread>
#include <xc.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fftw3.h>
#include <boost/multiprecision/mpfr.hpp>
//#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/legendre.hpp>
//#include <boost/math/interpolators/barycentric_rational.hpp>//not so great, see ExpandSymmetry for comparison with ALGLIB
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include "MPDPFT_HEADER_ONEPEXDFT.h"
#include "mpDPFT.h"
#include "Plugin_Triangulation.h"
#include "Plugin_KD.h"
#include "Plugin_1pEx_Rho1p.h"
#include "Plugin_1pEx.h"
#include "Plugin_OPT.h"
#include "Plugin_cec14_test_func.h"
#include "stdafx.h"
#include "statistics.h"
#include "specialfunctions.h"
#include "solvers.h"
#include "optimization.h"
#include "linalg.h"
#include "kernels_sse2.h"
#include "kernels_fma.h"
#include "kernels_avx2.h"
#include "interpolation.h"
#include "integration.h"
#include "fasttransforms.h"
#include "diffequations.h"
#include "dataanalysis.h"
#include "ap.h"
#include "alglibmisc.h"
#include "alglibinternal.h"
#include "sys/types.h"
//#include <sys/wait.h>
#include "sys/sysinfo.h"





#pragma omp declare reduction(vec_double_plus : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

struct sysinfo memInfo;

using namespace std;

//using namespace std::tr1;

using namespace alglib;


// I N I T I A L I Z E    G L O B A L    P A R A M E T E R S

double PI = 3.141592653589793, SQRTPI = sqrt(PI), MachinePrecision = 1.0e-16, MP = 1.0e-12;
double Gammam1half = -2.*SQRTPI, Gamma1half = SQRTPI, Gamma3half = 0.5*SQRTPI, Gamma5half = 0.75*SQRTPI, Gamma7half = 1.875*SQRTPI;
double RiemannBareSumCompensationFactor = 1.;
int sec = 1000000, millisec = 1000, GLOBALINTEGER = 0;

datastruct DATA;
taskstruct TASK;
OPTstruct OPTSCO;
//vector<double> GLOBALVEC;
int GLOBALCOUNT = 0;

//timing variables
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> fsec;
auto t0 = Time::now(), t1 = Time::now();

typedef boost::multiprecision::mpfr_float mp_type;
boost::math::policies::policy<boost::math::policies::max_series_iterations<10000000>> my_pFq_policy;
unsigned pFq_NumDigitsPrecision = 100;
double pFq_TimeoutSeconds = 10.;

 
// P R I M A R Y    R O U T I N E S

void GetData(taskstruct &task, datastruct &data){
    
    // I N I T I A L I Z A T I O N  
  
    double TotalTiming; time_t Timing0,Timing0b; time(&Timing0); time(&data.Timing1); data.Timing0 = Timing0;
    data.errorCount = 0; data.warningCount = 0;

    if(task.lastQ) PRINT("\n **************************************************************************************************************************************************",data);
    if(task.lastQ) PRINT(" ************************   GetData: Task = " + to_string(task.Type) + "; count = " + vec_to_str(task.count) + "; VEC = {" + vec_to_str(task.VEC) + "}   ************************",data);
    if(task.lastQ) PRINT(" **************************************************************************************************************************************************\n",data);
    
    if(task.lastQ) PRINT("\n I N I T I A L I Z A T I O N ...\n",data);

    SetOmpThreads(task,data);
    if(task.lastQ) PRINT(" ***** OMP_NUM_THREADS = " + to_string(data.ompThreads) + " *****",data);

    if(task.lastQ) PRINT(" ***** get input parameters *****",data); 
    GetInputParameters(task,data);
    if(data.FLAGS.Export) PrintInputParameters(data);
    
    if(data.FLAGS.Export) PRINT(" ***** run auxilliary tasks 1 *****",data);
    SetStreams(data);
    InitializeRandomNumberGenerator(data);
    
    if(data.FLAGS.Export) PRINT(" ***** initialize support, input derivatives, additional and system-specific parameters *****",data); 
    InitializeHardCodedParameters(data,task);   
    
    StartTimer("AuxTasks 2 & task parameterization",data);
    if(data.FLAGS.Export) PRINT(" ***** run auxilliary tasks 2 *****",data);
    ProcessAuxilliaryData(data);    
    if(data.FLAGS.Export) PRINT(" ***** set up task parameterization *****",data); 
    if(!(task.Type==8 || task.Type==88)) RetrieveTask(data,task);
	if(data.FLAGS.SCO) SetupSCO(data);
    EndTimer("AuxTasks 2 & task parameterization",data);

    if(data.FLAGS.Export) PRINT(" ***** allocate memory and FFT plans *****",data);
    AllocateDensityMemory(data);
    SetupMovieStorage(data);
    if(!(task.Type==44 || task.Type==444)) InitializeGlobalFFTWplans(data);//ToDo: Why is data.FLAGS.FFTW sometimes equal to 1 (although it is initialized as 0, and not changed afterwards) if task.Type==44 with openMP ???
    AllocateTMPmemory(data);
  
    if(data.FLAGS.Export) PRINT(" ***** initialize tabulated functions *****",data);
    GetResources(data);    
    LoadReferenceDensity(data);
    GetSpecialFunctions(data);
    InitializeKD(data);
  
    if(data.FLAGS.Export) PRINT(" ***** initialize potentials ****",data);
    InitializeEnvironments(data);
    if(data.InterpolVQ>0){ if(task.lastQ) PRINT(" ***** load existing V ***** ",data); LoadV(data,task); }
  
    if(data.FLAGS.Export) PRINT(" ***** test ground *****",data);
    RunTests(task,data);  

    if(data.FLAGS.Export) PRINT(" ***** run auxilliary tasks 3 ****",data);
	RunAuxTasks(task,data);
    SanityChecks(data);
    Misc(data);

	if(data.FLAGS.InitializeDensities){
		if(data.FLAGS.Export) PRINT(" ***** initialize densities ****",data);
		GetDensities(task,data);
		ImposeNoise(1,data);
		GetFieldStats(0,0,data);
		if(data.MovieQ) StoreMovieData(data);
		InitializePulayMixer(data);
	}
    
    time(&data.Timing2);
    if(data.FLAGS.Export) PRINT("\n ***** Initialization complete;  Timing = " + to_string(difftime(data.Timing2,data.Timing1)) + " *****\n",data); 
    abortQ(data);
  

    // S E L F C O N S I S T E N T   L O O P
  
    if(data.maxSCcount>0){
		time(&Timing0b); time(&data.Timing1);
		if(data.FLAGS.Export) PRINT("\n S E L F C O N S I S T E N T   L O O P ...\n",data); 
		while(!data.ExitSCloop){
			StoreCurrentIteration(data);
			UpdateV(data); //COUTalongX("V[0]",data.V[0],data);
			//AdmixPotentials(data);
			PreparationsForSCdensity(data);
			GetDensities(task,data);
			//ImposeNoise(2,data);
			CheckConvergence(data); //COUTalongX("Den[0]",data.Den[0],data);
			Intercept(data);
			AdmixDensities(data);
			InjectSchedule(data);
			PrepareNextIteration(task,data);
      }
      time(&data.Timing2);
      if(data.FLAGS.Export) PRINT("\n Loop Time = " + to_string(difftime(data.Timing2, Timing0b)) + "\n",data);
      if(data.SCcount>=data.maxSCcount && !data.FLAGS.ForestGeo){ data.warningCount++; PRINT("\n Warning!!! SC loop not converged (CC=" + to_string_with_precision(data.CC,16) + ") \n",data); }
    }
    else if(data.FLAGS.Export) PRINT(" ***** skip self-consistent loop ****",data);

    
    // O U T P U T
  
    if(data.FLAGS.Export) PRINT("\n O U T P U T ...\n",data);  
    ExportData(data);
  
    time(&data.Timing2); TotalTiming = difftime(data.Timing2, Timing0);
    if(data.FLAGS.Export) PRINT(" TotalRunTime = " + to_string(TotalTiming) + ";  " + to_string(data.errorCount) + " Error(s);  " + to_string(data.warningCount) + " Warning(s)" + "\n" + " EndOfControlFile",data);
    WriteControlFile(data);
    StoreTASKdata(task,data);
    CleanUp(data);
    //cout << "...GetData() finished..." << endl;
}

void GetGlobalMinimum_muVec(taskstruct &task){
  GetInputParameters(task,DATA); InitializeRandomNumberGenerator(DATA); vector<double> Start = {{0.}};
  //task.Aux = (double)DATA.maxSCcount;
  task.AuxVec.resize(DATA.S); task.AuxVec = DATA.thetaVec; //cout << "DATA.thetaVec = " << vec_to_str(DATA.thetaVec) << endl;
  TASK.AuxVec.resize(DATA.S); TASK.AuxVec = DATA.thetaVec;
  ConjugateGradientDescent(-2, 1, 1, Start, task, DATA);
  //ConjugateGradientDescent(-2, task.ParameterCount, task.ParameterCount, Start, task, DATA);
}

// taskstruct GetGlobalMinimum_Abundances(taskstruct &task){
void GetGlobalMinimum_Abundances(taskstruct &task){

  GetInputParameters(task,DATA); InitializeRandomNumberGenerator(DATA); vector<double> Start = {{0.}};
  ConjugateGradientDescent(-3, 1, 1, Start, task, DATA);

//   GetInputParameters(TASK,DATA); InitializeRandomNumberGenerator(DATA); vector<double> Start = {{0.}};
//   ConjugateGradientDescent(-3, 1, 1, Start, TASK, DATA);
//   taskstruct Task = TASK;
  
//   cout << "sanity-checks on TASK:" << endl;
//   TASKPRINT(" abc",task,1);//OK!
//   cout << task.controltaskfile.str() << endl;
//   cout << " good()=" << TASK.controltaskfile.good();
//   cout << " eof()=" << TASK.controltaskfile.eof();
//   cout << " fail()=" << TASK.controltaskfile.fail();
//   cout << " bad()=" << TASK.controltaskfile.bad();
//   cout << TASK.controltaskfile.str() << endl;
//   cout << " good()=" << task.controltaskfile.good();
//   cout << " eof()=" << task.controltaskfile.eof();
//   cout << " fail()=" << task.controltaskfile.fail();
//   cout << " bad()=" << task.controltaskfile.bad();  
//   cout << task.controltaskfile.str() << endl;//*** Error in `./mpDPFT': corrupted size vs. prev_size: 0x00000000027a5130 ***
  
//   return Task;
}

double FitFunction_ForestGeo_PSO(vector<double> var, int p){// var are the fit parameters
  datastruct localdata;
  localdata.VEC.resize(var.size()); localdata.VEC = var;
  localdata.ompThreads = 1;
  localdata.SwarmParticle = p;
  GetData(TASK,localdata);
  if(TASK.GuessMuVecQ){
    TASK.Report[p].activeQ = true;
    TASK.Report[p].muVec = localdata.muVec;
    TASK.Report[p].V = localdata.V;
  }
  //double a = Norm(VecDiff(localdata.TmpAbundances,localdata.Abundances))/localdata.AccumulatedAbundances, b = GetFigureOfMerit(localdata);
  //cout << "FitFunction_ForestGeo_PSO " << a << " " << b << endl;
  //if(a>0.01 && localdata.RelAcc>MP) return 1.0e+100; else return b;
  return GetFigureOfMerit(localdata);
}

double EquilibriumAbundances_ForestGeo_PSO(vector<double> var, int p){
  datastruct localdata;
  localdata.VEC.resize(var.size()); localdata.VEC = var;
  localdata.ompThreads = 1;
  localdata.SwarmParticle = p;
  GetData(TASK,localdata);
  return localdata.Etot;  
}

void GetAuxilliaryFit(taskstruct &task){//for ForestGEO data on Barro Colorado Island (BCI)
  if(task.Type==4){
    InitializeRandomNumberGenerator(DATA);
    vector<double> Start = {{0.}};
    ConjugateGradientDescent(-4, 5, 100, Start, task, DATA);
  }
  else if(task.Type==44){
    //begin USER INPUT
    
    //do fit or use manual task.VEC below
    TASK.fitQ = 1; //0: use manualtaskVEC --- 1: yes,PSO --- 2/4&5: no,load parameters from TabFunc_FitParameters.dat/mpDPFT_FitResults.dat --- 3: yes,GAO
    vector<double> manualtaskVEC = {{-2.508935695363862,-466496906.8760188,-1147404761.059853,0.05866912391859476,0.07272639566522333,0.008940666339053577,0.03309914260927597,0.03381519566152848,0.08643069877298412,0.008974286730437417,0.08433890523580365,0.01127892856195681,0.09353412355444157,0.1189469930846409,1662.629616144475,-21410.73738936825,-8432.46443006337,-171397.0200939592,-2321069660.084265,-1391534243.961914,0.07365042329933862,0.08422574551662618,0.2532561448187923,0.0729540924222795,0.04384480069690211,0.1171398796882476,0.1914918840869335,0.1465048651192586,0.0008921868506962575,0.1248898461055152,0.09522671687840502,1907.287917308552,45478.50094978847,-2917550.421704559,-10578.19620256191}};
    TASK.CoarseGrain = true;//coarse-grain area according to data.steps
    int averageQ = 0; //average averageQ-times over neigboring pixels in refData
    TASK.refDataType = 2;// --- 0: dbh-ordering --- 1: bas-ordering --- 2: total area bas-ordering
    TASK.maskratio = 1.;// ratio of GridSize to be used for random TASK.mask in fomOverlap
    bool manualMaskQ = false; // load TASK.mask-indices manually below
    vector<int> manualMask = {{0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 120, 121, 122, 123, 124, 125, 126, 127, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 204, 205, 206, 210, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 256, 257, 258, 259, 260, 262, 263, 264, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 306, 307, 308, 309, 310, 311, 313, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 408, 409, 410, 411, 412, 414, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 575, 576, 577, 578, 579, 581, 582, 583, 584, 585, 586, 612, 613, 614, 615, 616, 617, 618, 619, 620, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 635, 636, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 733, 734, 735, 736, 737, 739, 765, 767, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 788, 789, 790, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 867, 868, 869, 870, 871, 872, 874, 877, 878, 879, 880, 881, 883, 884, 885, 886, 887, 888, 889, 890, 891, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 930, 931, 932, 933, 934, 935, 936, 937, 938, 940, 941, 942, 943, 970, 972, 973, 976, 977, 978, 980, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1041, 1042, 1043, 1044, 1071, 1072, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1083, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1093, 1094, 1095, 1096, 1123, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1136, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1147, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1189, 1190, 1191, 1192, 1193, 1195, 1196, 1197, 1198, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1235, 1237, 1239, 1240, 1241, 1242, 1244, 1245, 1247, 1248, 1249, 1276, 1278, 1279, 1280, 1281, 1282, 1283, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1343, 1344, 1345, 1346, 1348, 1349, 1350, 1351, 1377, 1380, 1381, 1383, 1384, 1385, 1386, 1389, 1390, 1391, 1392, 1393, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1428, 1429, 1430, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1450, 1451, 1452, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1530, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1545, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1581, 1582, 1583, 1584, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1708, 1734, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1755, 1756, 1757, 1758, 1759, 1785, 1786, 1787, 1788, 1789, 1790, 1792, 1793, 1794, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1846, 1847, 1848, 1849, 1850, 1852, 1853, 1854, 1855, 1857, 1858, 1859, 1860, 1861, 1887, 1888, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1909, 1910, 1911, 1912, 1938, 1939, 1940, 1941, 1942, 1944, 1945, 1947, 1948, 1949, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1989, 1990, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2064, 2065, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2099, 2101, 2103, 2104, 2105, 2107, 2108, 2109, 2110, 2111, 2112, 2114, 2115, 2116, 2142, 2143, 2144, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2157, 2158, 2160, 2161, 2162, 2163, 2164, 2165, 2167, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2203, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2213, 2215, 2216, 2217, 2218, 2244, 2245, 2246, 2247, 2248, 2249, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2260, 2261, 2262, 2263, 2265, 2266, 2267, 2268, 2269, 2295, 2296, 2297, 2299, 2300, 2301, 2302, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2319, 2320, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2397, 2398, 2399, 2400, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2422, 2448, 2449, 2450, 2451, 2453, 2454, 2455, 2456, 2457, 2458, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2468, 2469, 2470, 2471, 2472, 2473, 2499, 2500, 2501, 2502, 2503, 2505, 2507, 2509, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575}};
    TASK.Corr.Measure = 1;//Correlation measure --- 1: Least-Squares --- 2: Pearson --- 3: LargeDataDeviants
    TASK.Corr.POW = 10;//Measure raised to power
    TASK.Corr.PercentageOfOutliers = 0.;
    TASK.PositiveTauQ = true;
    TASK.NumGlobalParam = 1;
    TASK.NumEnvParam = 2;
    TASK.NumIntParam = 4;//1 for noninteracting
    TASK.GuessMuVecQ = false;//something odd is happening if GuessMuVecQ = true: recomputed FigureOfMerit differs from so far encountered bestf...
    bool FixFitParams = false;//true: fix some parameters (specified below; at present: sigma, env params, resource params, and tau) to the ones loaded from mpDPFT_FitResults.dat
    //end USER INPUT
    
    
    TASK.Type = 44;
    GetInputParameters(task,DATA);
    LoadrefData(averageQ,DATA,TASK);
    InitializeHardCodedParameters(DATA,task);
    TASK.muVec.resize(DATA.S); TASK.muVec = DATA.muVec;
    
    TASK.mask.clear();
    int n = DATA.EdgeLength;
    if(TASK.fitQ==0 || TASK.fitQ==2 || TASK.fitQ==4 || TASK.fitQ==5) TASK.GuessMuVecQ = false;
    if(TASK.maskratio<0.){ TASK.mask.resize(DATA.GridSize); for(int m=0;m<DATA.GridSize;m++) TASK.mask[m] = m; }
    else if(manualMaskQ){ TASK.mask.resize(manualMask.size()); TASK.mask = manualMask; TASKPRINT("GetAuxilliaryFit: manual fit mask loaded: \n" + vec_to_str(TASK.mask),task,1); }
    else{
      double gridsize = (double)(DATA.GridSize);
      int jLimit = (n+1)/2;
      if(TASK.CoarseGrain && TASK.refDataType==2){ n = 51; gridsize = n*n; jLimit = (n+1)/2-1; }
      if(TASK.maskratio>1.-MP && TASK.refDataType==2) for(int i=0;i<n;i++) for(int j=0;j<jLimit;j++) TASK.mask.push_back(i*n+j);
      else{
	if(TASK.refDataType==2) TASK.maskratio *= 0.5;
	int M = (int)(TASK.maskratio*gridsize);
	TASK.mask.resize(M);
	InitializeRandomNumberGenerator(DATA);
	if(TASK.refDataType==2) for(int m=0;m<M;m++){ int i = (int)(DATA.RNpos(DATA.MTGEN)*(double)(n)); int j = (int)(DATA.RNpos(DATA.MTGEN)*(double)jLimit); TASK.mask[m] = i*n+j; }
	else for(int m=0;m<M;m++) TASK.mask[m] = (int)(DATA.RNpos(DATA.MTGEN)*gridsize);
	bool maskFilledQ = false;
	while(!maskFilledQ){
	  maskFilledQ = true;
	  sort(TASK.mask.begin(),TASK.mask.end());
	  for(int m=1;m<M;m++) if(TASK.mask[m]==TASK.mask[m-1]){
	    maskFilledQ = false;
	    if(TASK.refDataType==2){ int i = (int)(DATA.RNpos(DATA.MTGEN)*(double)(n)); int j = (int)(DATA.RNpos(DATA.MTGEN)*(double)jLimit); TASK.mask[m] = i*n+j; }
	    else TASK.mask[m] = (int)(DATA.RNpos(DATA.MTGEN)*gridsize);
	  }
	}
      }
      TASKPRINT("GetAuxilliaryFit: random fit mask initialized (" + to_string(TASK.mask.size()) + " elements): \n" + IntVec_to_CommaSeparatedString(TASK.mask),task,1);
    }
    LoadFOMrefData(averageQ, DATA,TASK);
    
    int NumGlobalParam = TASK.NumGlobalParam, NumEnvParam = TASK.NumEnvParam, NumResParam = DATA.Mpp.size(), NumIntParam = TASK.NumIntParam, NumberOfFitParametersPerSpecies = NumEnvParam+NumResParam+NumIntParam, NumMu = 0;
    if(DATA.RelAcc<MP) NumMu = DATA.S;
    vector<double> Start(task.AuxVec); TASK.VEC.resize(Start.size());
    vector<string> FitParametersNames(Start.size());
    if(NumGlobalParam>0) FitParametersNames[0] = "sigma";
    for(int s=0;s<DATA.S;s++){
      for(int m=0;m<NumEnvParam;m++) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+m] = "S" + to_string(s) + "-Env" + to_string(m);
      for(int k=0;k<NumResParam;k++) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+k] = "S" + to_string(s) + "-nu" + to_string(k);
      if(NumIntParam>0) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+0] = "InternalPressureS" + to_string(s);
      if(NumIntParam>1) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+1] = "AmensalismS" + to_string(s);
      if(NumIntParam>2) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+2] = "RepMutS" + to_string(s);
      if(NumIntParam>3) FitParametersNames[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+3] = "AsymCompS" + to_string(s);
    }
    for(int s=0;s<NumMu;s++) FitParametersNames[NumGlobalParam+DATA.S*NumberOfFitParametersPerSpecies+s] = "mu_S" + to_string(s);
    
    task.VEC.resize(Start.size());
    if(FixFitParams) LoadFitResults(task.VEC);
    
    TASK.TotalNumFitParams = NumGlobalParam+DATA.S*NumberOfFitParametersPerSpecies+NumMu;
    
    TASKPRINT("GetAuxilliaryFit: DATA abundances = " + vec_to_str(DATA.Abundances),task,1);
    
    if(TASK.fitQ==1 || TASK.fitQ==3){
      TASK.optInProgress = true;
      OPTstruct opt;
      SetDefaultOPTparams(opt);
      opt.reportQ = 1;
      opt.printQ = 1;
      opt.ExportIntermediateResults= true;
      opt.UpdateSearchSpaceQ = 1;
      opt.threads = (int)omp_get_max_threads();//1;
      opt.function = -44;
      opt.D = Start.size();
      opt.SearchSpaceLowerVec.resize(opt.D); opt.SearchSpaceUpperVec.resize(opt.D); opt.VariableLowerBoundaryQ.resize(opt.D); opt.VariableUpperBoundaryQ.resize(opt.D);
      
      if(NumGlobalParam>0){
	if(FixFitParams){ opt.SearchSpaceLowerVec[0] = task.VEC[0]; opt.SearchSpaceUpperVec[0] = opt.SearchSpaceLowerVec[0]; }
	else{ opt.SearchSpaceLowerVec[0] = -40.; opt.SearchSpaceUpperVec[0] = 0.; }
	opt.VariableLowerBoundaryQ[0] = false;
	opt.VariableUpperBoundaryQ[0] = false; 
      }
      
      for(int s=0;s<DATA.S;s++){
	for(int m=0;m<NumEnvParam;m++){
	  int index = NumGlobalParam+s*NumberOfFitParametersPerSpecies+m;
	  if(FixFitParams){
	    opt.SearchSpaceLowerVec[index] = task.VEC[index];
	    opt.SearchSpaceUpperVec[index] = opt.SearchSpaceLowerVec[index];
	    opt.VariableLowerBoundaryQ[index] = false;
	    opt.VariableUpperBoundaryQ[index] = false;
	  }
	  else{
	    opt.SearchSpaceLowerVec[index] = -1.*ABS(DATA.mpp[NumEnvParam*s+m]);
	    opt.SearchSpaceUpperVec[index] = +1.*ABS(DATA.mpp[NumEnvParam*s+m]);
	    opt.VariableLowerBoundaryQ[index] = true;
	    opt.VariableUpperBoundaryQ[index] = true;
	  }
	}
	for(int k=0;k<NumResParam;k++){
	  int index = NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+k;
	  if(FixFitParams){
	    opt.SearchSpaceLowerVec[index] = task.VEC[index];
	    opt.SearchSpaceUpperVec[index] = opt.SearchSpaceLowerVec[index];
	    opt.VariableLowerBoundaryQ[index] = false;
	    opt.VariableUpperBoundaryQ[index] = false;
	  }
	  else{	  
	    opt.SearchSpaceLowerVec[index] = 0.;//*DATA.Consumption[k][s];
	    opt.SearchSpaceUpperVec[index] = 1.*DATA.Consumption[k][s];
	    opt.VariableLowerBoundaryQ[index] = false;
	    opt.VariableUpperBoundaryQ[index] = true;
	  }
	}
	for(int m=0;m<NumIntParam;m++){
	  int index = NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+m;
	  if(TASK.PositiveTauQ && m==0){
	    opt.SearchSpaceLowerVec[index] = -0.999*DATA.tauVec[s];
	    opt.VariableLowerBoundaryQ[index] = false;
	  }
	  else{
	    opt.SearchSpaceLowerVec[index] = -1.*DATA.MPP[s][m];
	    opt.VariableLowerBoundaryQ[index] = true;
	  }	  
	  opt.SearchSpaceUpperVec[index] = 1.*DATA.MPP[s][m];
	  opt.VariableUpperBoundaryQ[index] = true;
	  if(m==0 && FixFitParams){
	    opt.SearchSpaceLowerVec[index] = task.VEC[index];
	    opt.SearchSpaceUpperVec[index] = opt.SearchSpaceLowerVec[index];
	    opt.VariableLowerBoundaryQ[index] = false;
	    opt.VariableUpperBoundaryQ[index] = false;	    
	  }
	}
      }
      for(int s=0;s<NumMu;s++){
	int index = NumGlobalParam+DATA.S*NumberOfFitParametersPerSpecies+s;
	opt.SearchSpaceLowerVec[index] = -5.*ABS(DATA.muVec[s]);
	opt.SearchSpaceUpperVec[index] = +5.*ABS(DATA.muVec[s]);   
	opt.VariableLowerBoundaryQ[index] = true;
	opt.VariableUpperBoundaryQ[index] = true;	
      }
      int NumFitParams = NumGlobalParam+DATA.S*NumberOfFitParametersPerSpecies+NumMu, amendment = 0; if(FixFitParams) amendment = NumGlobalParam+DATA.S*(NumEnvParam+NumResParam+1);
      TASKPRINT("Fitting (threads=" + to_string(opt.threads) + ") initialized. Number of fit parameters = " + to_string(NumFitParams-amendment),task,1);

      vector<double> FitParameters(NumFitParams);
      int maxeval;
      double OptimalFuncValue;
      if(TASK.fitQ==1){
	SetDefaultPSOparams(opt);
	opt.pso.InitialSwarmSize = 5*opt.D;/*DATA.S*opt.D;*//*opt.D*opt.D;*//*10*opt.D;*//*100*opt.D;*/
	InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
	opt.pso.runs = /*100;*/20;
	opt.pso.increase = 1.;//increase of swarm size towards final run
	opt.pso.epsf = 1.0e-3/((double)DATA.S);//ForestGEOknob
	
	//opt.pso.MaxLinks = 3;//not so good, apparently
	//opt.pso.eval_max_init = (int)1.0e+5*DATA.S;
	//opt.pso.VarianceCheck = 10;
	//opt.pso.elitism = 0.7;
	//opt.pso.AlwaysUpdateSearchSpace = true;
	opt.pso.reseed = 0.5;
	opt.pso.reseedEvery = opt.D;
	//opt.pso.CoefficientDistribution = 3;
	if(TASK.GuessMuVecQ){
	  TASK.Report.resize(opt.pso.InitialSwarmSize);
	  for(int p=0;p<TASK.Report.size();p++){
	    TASK.Report[p].SwarmParticle = p;
	    TASK.Report[p].muVec = DATA.muVec;
	    TASK.Report[p].V.resize(DATA.S);
	    for(int s=0;s<DATA.S;s++) TASK.Report[p].V[s].resize(DATA.GridSize);
	  }	
	}
	TASKPRINT("start PSO ... ",task,1);
	PSO(opt);
	OptimalFuncValue = opt.pso.bestf;
	FitParameters = opt.pso.bestx;
	maxeval = opt.pso.eval_max_init;
      }
      else if(TASK.fitQ==3){
	opt.gao.runs = opt.threads;
	opt.pso.epsf = DATA.RelAcc;
	SetDefaultGAOparams(opt);
	opt.gao.populationSize = opt.D;
	opt.gao.generationMax = 10*opt.gao.populationSize;
	opt.evalMax = opt.gao.populationSize*opt.gao.generationMax;
	//opt.gao.AllowImmigrationAt = (int)(0.5*(double)opt.gao.generationMax);
	//InitializeGenerationMaxDependentVariables(opt.gao.generationMax,opt);
	GAO(opt); 
	OptimalFuncValue = opt.gao.bestf;
	FitParameters = opt.gao.bestx;
	maxeval = opt.gao.runs*opt.evalMax;
      }

      TASKPRINT(opt.control.str(),task,0);
      TASKPRINT("Fitting completed: " + to_string(opt.nb_eval) + "/" + to_string(maxeval) + " function evaluations\n",task,1);
      TASKPRINT("bestf = " + to_string_with_precision(OptimalFuncValue,16) + " at final fit parameters [ratio within range] <- from start values:\n",task,1);
      for(int i=0;i<FitParameters.size();i++) TASKPRINT(FitParametersNames[i] + " " + to_string_with_precision(FitParameters[i],8) + " [" + to_string((FitParameters[i]-opt.SearchSpaceLowerVec[i])/(opt.SearchSpaceUpperVec[i]-opt.SearchSpaceLowerVec[i])) + "] <- " + to_string_with_precision(Start[i],8),task,1);   
    
      task.VEC = FitParameters;    
      
      TASKPRINT("Analysis of FitParameters:",task,1);
      ofstream FitResults;
      FitResults.open("mpDPFT_FitResults.dat");      
      vector<double> SortedHistory;
      vector<vector<double>> SortedHistoryX;
      for(auto i: sort_indices(opt.pso.History)){
	SortedHistory.push_back(opt.pso.History[i]);
	SortedHistoryX.push_back(opt.pso.HistoryX[i]);
      }
      for(int h=0;h<SortedHistory.size();h++){
	TASKPRINT(to_string_with_precision(SortedHistory[h],16),task,1);
	TASKPRINT(vec_to_str_with_precision(SortedHistoryX[h],16),task,1);
	FitResults << to_string_with_precision(SortedHistory[h],16) << " " << vec_to_str_with_precision(SortedHistoryX[h],16) << endl;
      }
      FitResults.close(); 
      
    }
    else{//initialize with (e.g. fitted) parameters
      if(TASK.fitQ==0) task.VEC = manualtaskVEC;
      else if(TASK.fitQ==2) LoadFitParameters(task.VEC);
      else if(TASK.fitQ==4) LoadFitResults(task.VEC);
      //cout << vec_to_str(task.VEC) << endl; usleep(10*sec);
/*      TASK.Report.resize(1);
      TASK.Report[0].SwarmParticle = 0;
      TASK.Report[0].activeQ = false;   */   
    }
    
    if(TASK.fitQ==5){//find ecosystem-equilibrium by minimizing energy in space of abundances
      TASK.VEC.clear(); TASK.VEC.resize(TASK.TotalNumFitParams); LoadFitResults(TASK.VEC); task.VEC = TASK.VEC;
      
      OPTstruct opt;
      SetDefaultOPTparams(opt);
      opt.reportQ = 1;
      opt.printQ = 1;
      opt.ExportIntermediateResults= true;
      opt.UpdateSearchSpaceQ = 1;
      opt.threads = (int)omp_get_max_threads();//1;
      opt.function = -440;
      opt.D = DATA.S;
      opt.SearchSpaceLowerVec.resize(opt.D); opt.SearchSpaceUpperVec.resize(opt.D); opt.VariableLowerBoundaryQ.resize(opt.D); opt.VariableUpperBoundaryQ.resize(opt.D);      
      
      for(int d=0;d<opt.D;d++){
	opt.SearchSpaceLowerVec[d] = 1.;
	opt.SearchSpaceUpperVec[d] = 120.;
	opt.VariableLowerBoundaryQ[d] = false;
	opt.VariableUpperBoundaryQ[d] = true;
      }
      
      SetDefaultPSOparams(opt);
      //opt.pso.InitialSwarmSize = 5*opt.D;/*DATA.S*opt.D;*//*opt.D*opt.D;*//*10*opt.D;*/
      //InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
      opt.pso.runs = 20;
      opt.pso.increase = 1.;//increase of swarm size towards final run
      opt.pso.epsf = (double)DATA.S;
      //opt.pso.AlwaysUpdateSearchSpace = true;
      opt.pso.reseed = 0.5;
      opt.pso.reseedEvery = opt.D;
      TASKPRINT("start PSO ... ",task,1);
      PSO(opt);
	
      TASKPRINT(opt.control.str(),task,0);
      TASKPRINT("Minimization completed: " + to_string(opt.nb_eval) + "/" + to_string(opt.pso.eval_max_init) + " function evaluations\n",task,1);
      TASKPRINT("bestf = " + to_string_with_precision(opt.pso.bestf,16) + " at final fit parameters:\n" + vec_to_str(opt.pso.bestx),task,1);    	
      
      task.abundances = opt.pso.bestx;
    }
  }
}

void GetGlobalMinimum_Abundances_AnnealedGradientDescent(taskstruct &task){
  GetInputParameters(task,DATA);
  InitializeRandomNumberGenerator(DATA);
  GradientDescent(task);
}

vector<vector<double>> GetTaskHyperBox(datastruct &tmpdata){
  int I = tmpdata.TaskHyperBox.size()/2;
  vector<vector<double>> hyperbox(I);
  for(int i=0;i<I;i++){
    hyperbox[i].resize(2);
    hyperbox[i][0] = tmpdata.TaskHyperBox[2*i];
    hyperbox[i][1] = tmpdata.TaskHyperBox[2*i+1];
  }
  return hyperbox;
}

void GetInputParameters(taskstruct &task, datastruct &data){
  ifstream infile;
  infile.open("mpDPFT.input");
  string line;
  int lineCount = 0;
  vector<int> dummyIntVec = {{0}}; vector<double> dummyDoubleVec = {{0.}};
  
  bool expand = false;
  
  double val;
  int ival;
  string err = "GetInputParameters: Input error";
  while(getline(infile, line)){
    istringstream iss(line);   
    switch(lineCount){
      case 0: {
	iss >> data.System;
	if(data.System==31/*L,dbh*/ || data.System==32/*R,dbh*/ || data.System==33/*L,bas*/ || data.System==34/*R,bas*/ || data.System==35/*all,dbh*/ || data.System==36/*all,bas*/){ data.FLAGS.ForestGeo = true; expand = true; }
	data.InputParameterNames.push_back("System"); break;
      }
      case 1: { iss >> data.Symmetry; data.InputParameterNames.push_back("Symmetry"); break; }
      case 2: { iss >> data.TaskType; data.InputParameterNames.push_back("TaskType"); break; }
      case 3: { iss >> data.TaskParameterCount; data.InputParameterNames.push_back("TaskParameterCount"); break; }
      case 4: { iss.str(line); while(iss){ iss >> val; data.TaskHyperBox.push_back(val); } data.TaskHyperBox.erase(data.TaskHyperBox.end()-1); data.InputParameterNames.push_back("TaskHyperBox"); break; }
      case 5: { iss >> data.DIM; data.InputParameterNames.push_back("DIM"); break; }
      case 6: { iss >> data.Units; data.InputParameterNames.push_back("Units"); break; }
      case 7: { iss >> data.edge; data.InputParameterNames.push_back("edge"); break; }
      case 8: { iss >> data.steps; data.InputParameterNames.push_back("steps"); break; }
      case 9: { iss >> data.S; data.InputParameterNames.push_back("S"); break; }
      case 10: { iss.str(line); if(data.FLAGS.ForestGeo){ LoadAbundances(data); } else for(int i=0;i<data.S;i++){ iss >> val; data.Abundances.push_back(val);} data.InputParameterNames.push_back("Abundances"); break; }
      case 11: { iss >> data.RelAcc; data.InputParameterNames.push_back("RelAcc"); break; }
      case 12: { iss >> data.InternalAcc; data.InputParameterNames.push_back("InternalAcc"); break; }
      case 13: { iss >> data.K; data.InputParameterNames.push_back("K"); break; }
      case 14: { iss.str(line); if(expand){ data.Resources.resize(data.K); iss >> val; ExpandInput(data.Resources,val,data.K); } else for(int i=0;i<data.K;i++){ iss >> val; data.Resources.push_back(val);} data.InputParameterNames.push_back("Resources"); break; }
      case 15: { iss >> data.DensityExpression; data.InputParameterNames.push_back("DensityExpression"); break; }
      case 16: { iss.str(line); while(iss){ iss >> val; data.mpp.push_back(val); } data.mpp.erase(data.mpp.end()-1); data.InputParameterNames.push_back("mpp"); break; }
      case 17: { iss.str(line); while(iss){ iss >> val; data.Mpp.push_back(val); } data.Mpp.erase(data.Mpp.end()-1); data.InputParameterNames.push_back("Mpp"); break; }
      case 18: { iss.str(line); if(expand){ iss >> val; ExpandInput(data.tauVec,val,data.S); } else for(int i=0;i<data.S;i++){ iss >> val; data.tauVec.push_back(val);} data.InputParameterNames.push_back("tauVec"); break; }
      case 19: { iss.str(line); if(expand){ iss >> val; ExpandInput(data.tVec,val,data.S); } else for(int i=0;i<data.S;i++){ iss >> val; data.tVec.push_back(val);} data.InputParameterNames.push_back("tVec"); break; }
      case 20: { iss.str(line); if(expand){ iss >> val; ExpandInput(data.TVec,val,data.S); } else for(int i=0;i<data.S;i++){ iss >> val; data.TVec.push_back(val);} data.InputParameterNames.push_back("TVec"); break; }
      case 21: { iss.str(line); if(expand){ iss >> ival; ExpandInput(data.Environments,ival,data.S); } else for(int i=0;i<data.S;i++){ iss >> ival; data.Environments.push_back(ival);} data.InputParameterNames.push_back("Environments"); break; }
      case 22: { iss >> data.stretchfactor; data.InputParameterNames.push_back("stretchfactor"); break; }
      case 23: { iss >> data.Wall; data.InputParameterNames.push_back("Wall"); break; }
      case 24: { iss >> data.Noise; data.InputParameterNames.push_back("Noise"); break; }
      case 25: { iss >> data.InterpolVQ; data.InputParameterNames.push_back("InterpolVQ"); break; } 
      case 26: { iss.str(line); while(iss){ iss >> ival; data.Interactions.push_back(ival); } data.Interactions.erase(data.Interactions.end()-1); data.InputParameterNames.push_back("Interactions"); break; }
      case 27: { iss >> data.incrementalV; data.InputParameterNames.push_back("incrementalV"); break; }
      case 28: { iss.str(line); if(expand){ iss >> val; ExpandInput(data.muVec,val,data.S); /*cout << vec_to_str(data.muVec) << endl;*/ } else for(int i=0;i<data.S;i++){ iss >> val; data.muVec.push_back(val);} data.InputParameterNames.push_back("muVec"); break; }
      case 29: { iss >> data.DeltamuModifier; data.InputParameterNames.push_back("DeltamuModifier"); break; }
      case 30: { iss.str(line); for(int i=0;i<3;i++){ iss >> val; data.Mixer.push_back(val);} data.InputParameterNames.push_back("Mixer"); break; }
      case 31: { iss.str(line); if(expand){ iss >> val; ExpandInput(data.thetaVec,val,data.S); } else for(int i=0;i<data.S;i++){ iss >> val; data.thetaVec.push_back(val);} data.InputParameterNames.push_back("thetaVec"); break; }
      case 32: { iss >> data.SCcriterion; data.InputParameterNames.push_back("SCcriterion"); break; }
      case 33: { iss >> data.maxSCcount; data.InputParameterNames.push_back("maxSCcount"); break; }
      case 34: { iss >> data.method; data.InputParameterNames.push_back("method"); break; }
      case 35: { iss >> data.DelFieldMethod; data.InputParameterNames.push_back("DelFieldMethod"); break; }
      case 36: { iss >> data.MovieQ; data.InputParameterNames.push_back("MovieQ"); break; }
      case 37: { iss.str(line); while(iss){ iss >> ival; data.Schedule.push_back(ival); } data.Schedule.erase(data.Schedule.end()-1); data.InputParameterNames.push_back("Schedule"); break; }
      case 38: { iss.str(line); while(iss){ iss >> ival; data.EkinTypes.push_back(ival); } data.EkinTypes.erase(data.EkinTypes.end()-1); data.InputParameterNames.push_back("EkinTypes"); break; }
      case 39: { iss >> data.alpha; data.InputParameterNames.push_back("alpha"); break; }
      case 40: { iss >> data.beta; data.InputParameterNames.push_back("beta"); break; }
      case 41: { iss >> data.gamma; data.InputParameterNames.push_back("gamma"); break; }
      case 42: { iss >> data.gammaH; data.InputParameterNames.push_back("gammaH"); break; }
      case 43: { iss >> data.degeneracy; data.InputParameterNames.push_back("degeneracy"); break; }
      case 44: { iss.str(line); while(iss){ iss >> ival; data.regularize.push_back(ival); } data.regularize.erase(data.regularize.end()-1); data.InputParameterNames.push_back("regularize"); break; }
      case 45: { iss >> data.RegularizationThreshold; data.InputParameterNames.push_back("RegularizationThreshold"); break; }
      case 46: { iss >> data.Print; data.InputParameterNames.push_back("Print"); break; }
      default: { PRINT(err,data); break; }
    }
    lineCount++;
  }
  
  if(!data.InterpolVQ){ ofstream AuxFile; AuxFile.open("mpDPFT_Aux.dat", std::ofstream::trunc); AuxFile.close(); }
  
  if(data.DensityExpression==5){
    data.MppVec.clear(); data.MppVec.resize(data.S);
    for(int s=0;s<data.S;s++) for(int i=0;i<3;i++) data.MppVec[s].push_back(data.Mpp[i]);
  }
  
  if(data.FLAGS.ForestGeo && task.lastQ) data.Print = 2;

  if(task.Type==8 || task.Type==88) RetrieveTask(data,task);

  if(data.Environments[0]==10){
    data.FLAGS.ExportCubeFile = 1;
    data.FLAGS.vW = 1;
  }
  //data.FLAGS.ExportCubeFile = 1;
  //data.FLAGS.vW = 1;

  for(int i=0;i<data.Interactions.size();i++) if(data.Interactions[i]<0) data.FLAGS.LIBXC = 1;
  if(data.DensityExpression==3 || data.DensityExpression==5 || data.DensityExpression==6 || data.DensityExpression==9 || IntegerElementQ(8,data.Interactions) || IntegerElementQ(19,data.Interactions) || IntegerElementQ(20,data.Interactions) || data.Mixer[0]>0. || IntegerElementQ(3,data.regularize) || data.FLAGS.LIBXC || data.DelFieldMethod==4 || data.DelFieldMethod==5) data.FLAGS.FFTW = 1; //cout << "data.FLAGS.FFTW " << data.FLAGS.FFTW << endl;
  if(task.Type>=4 && task.Type<=7) data.FLAGS.RefData = 1;
  if(data.DensityExpression==5 || data.DensityExpression==11) data.FLAGS.CurlyA = 1;
  if(data.DensityExpression==5 || data.DensityExpression==8) data.FLAGS.PolyLogs = 1;
  if(data.DensityExpression==4 || data.DensityExpression==5 || data.DensityExpression==11 || data.DIM==3 || data.FLAGS.LIBXC || (data.FLAGS.ForestGeo && TASK.NumEnvParam>2) || data.FLAGS.vW) data.FLAGS.Del = 1;
  if(task.lastQ || task.Type==8 || task.Type==61 || task.Type==88 || task.Type==444){ data.FLAGS.Export = 1; if(data.Print<0) data.Print = 1; }
  if(task.lastQ && (task.Type==2 || task.Type==3)) data.Print = 0;
  if(data.DensityExpression==9) data.FLAGS.Gyk = 1;
  
  infile.close();
}

// void ExpandInput(int doubleQ, vector<int> &IntVec, vector<double> &DoubleVec, int size){
//   if(doubleQ){
//       double input = DoubleVec[0];
//       DoubleVec.resize(size);
//       for(int s=0;s<size;s++) DoubleVec[s] = input;
//   }
//   else{
//       int input = IntVec[0];
//       IntVec.resize(size);
//       for(int s=0;s<size;s++) IntVec[s] = input;
//   }
// }

void PrintInputParameters(datastruct &data){
  PRINT("\n ***** Input loaded *****",data);
  ostringstream oss;
  oss << data.InputParameterNames[0] << " " << data.System << "\n";
  oss << data.InputParameterNames[1] << " " << data.Symmetry << "\n";
  oss << data.InputParameterNames[2] << " " << data.TaskType << "\n";
  oss << data.InputParameterNames[3] << " " << data.TaskParameterCount << "\n";
  oss << data.InputParameterNames[4]; for(int i=0;i<data.TaskHyperBox.size();i++){ oss << " " << data.TaskHyperBox[i]; } oss << "\n";
  oss << data.InputParameterNames[5] << " " << data.DIM << "\n";
  oss << data.InputParameterNames[6] << " " << data.Units << "\n";
  oss << data.InputParameterNames[7] << " " << data.edge << "\n";
  oss << data.InputParameterNames[8] << " " << data.steps << "\n";
  oss << data.InputParameterNames[9] << " " << data.S << "\n";
  oss << data.InputParameterNames[10]; for(int i=0;i<data.S;i++){ oss << " " << data.Abundances[i]; } oss << "\n";
  oss << data.InputParameterNames[11] << " " << data.RelAcc << "\n";
  oss << data.InputParameterNames[12] << " " << data.InternalAcc << "\n";
  oss << data.InputParameterNames[13] << " " << data.K << "\n";
  oss << data.InputParameterNames[14]; for(int i=0;i<data.K;i++){ oss << " " << data.Resources[i]; } oss << "\n";
  oss << data.InputParameterNames[15] << " " << data.DensityExpression << "\n";
  oss << data.InputParameterNames[16]; for(int i=0;i<data.mpp.size();i++){ oss << " " << data.mpp[i]; } oss << "\n";
  oss << data.InputParameterNames[17]; for(int i=0;i<data.Mpp.size();i++){ oss << " " << data.Mpp[i]; } oss << "\n";
  oss << data.InputParameterNames[18]; for(int i=0;i<data.S;i++){ oss << " " << data.tauVec[i]; } oss << "\n";
  oss << data.InputParameterNames[19]; for(int i=0;i<data.S;i++){ oss << " " << data.tVec[i]; } oss << "\n";
  oss << data.InputParameterNames[20]; for(int i=0;i<data.S;i++){ oss << " " << data.TVec[i]; } oss << "\n";
  oss << data.InputParameterNames[21]; for(int i=0;i<data.S;i++){ oss << " " << data.Environments[i]; } oss << "\n";
  oss << data.InputParameterNames[22] << " " << data.stretchfactor << "\n";
  oss << data.InputParameterNames[23] << " " << data.Wall << "\n";
  oss << data.InputParameterNames[24] << " " << data.Noise << "\n";
  oss << data.InputParameterNames[25] << " " << data.InterpolVQ << "\n";
  oss << data.InputParameterNames[26]; for(int i=0;i<data.Interactions.size();i++){ oss << " " << data.Interactions[i]; } oss << "\n";
  oss << data.InputParameterNames[27] << " " << data.incrementalV << "\n";
  oss << data.InputParameterNames[28]; for(int i=0;i<data.S;i++){ oss << " " << data.muVec[i]; } oss << "\n";
  oss << data.InputParameterNames[29] << " " << data.DeltamuModifier << "\n";
  oss << data.InputParameterNames[30]; for(int i=0;i<3;i++){ oss << " " << data.Mixer[i]; } oss << "\n";
  oss << data.InputParameterNames[31]; for(int i=0;i<data.S;i++){ oss << " " << data.thetaVec[i]; } oss << "\n";
  oss << data.InputParameterNames[32] << " " << data.SCcriterion << "\n";
  oss << data.InputParameterNames[33] << " " << data.maxSCcount << "\n";
  oss << data.InputParameterNames[34] << " " << data.method << "\n";
  oss << data.InputParameterNames[35] << " " << data.DelFieldMethod << "\n";
  oss << data.InputParameterNames[36] << " " << data.MovieQ << "\n";
  oss << data.InputParameterNames[37]; for(int i=0;i<data.Schedule.size();i++){ oss << " " << data.Schedule[i]; } oss << "\n";
  oss << data.InputParameterNames[38]; for(int i=0;i<data.EkinTypes.size();i++){ oss << " " << data.EkinTypes[i]; } oss << "\n";
  oss << data.InputParameterNames[39] << " " << data.alpha << "\n";
  oss << data.InputParameterNames[40] << " " << data.beta << "\n";
  oss << data.InputParameterNames[41] << " " << data.gamma << "\n";
  oss << data.InputParameterNames[42] << " " << data.gammaH << "\n";
  oss << data.InputParameterNames[43] << " " << data.degeneracy << "\n";
  oss << data.InputParameterNames[44]; for(int i=0;i<data.regularize.size();i++){ oss << " " << data.regularize[i]; } oss << "\n";
  oss << data.InputParameterNames[45] << " " << data.RegularizationThreshold << "\n";
  oss << data.InputParameterNames[46] << " " << data.Print << "\n";
  PRINT(oss.str(),data);
  
  data.txtout << "threads = " << data.ompThreads << "\\\\";
  data.txtout << "**** InputParameters from mpDPFT.input ****\\\\";
  data.txtout << data.InputParameterNames[0] << " " << data.System << "\\\\";
  data.txtout << data.InputParameterNames[1] << " " << data.Symmetry << "\\\\";
  data.txtout << data.InputParameterNames[2] << " " << data.TaskType << "\\\\";
  data.txtout << data.InputParameterNames[3] << " " << data.TaskParameterCount << "\\\\";
  data.txtout << data.InputParameterNames[4]; for(int i=0;i<data.TaskHyperBox.size();i++){ data.txtout << " " << data.TaskHyperBox[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[5] << " " << data.DIM << "\\\\";
  data.txtout << data.InputParameterNames[6] << " " << data.Units << "\\\\";
  data.txtout << data.InputParameterNames[7] << " " << data.edge << "\\\\";
  data.txtout << data.InputParameterNames[8] << " " << data.steps << "\\\\";
  data.txtout << data.InputParameterNames[9] << " " << data.S << "\\\\";
  data.txtout << data.InputParameterNames[10]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.Abundances[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[11] << " " << data.RelAcc << "\\\\";
  data.txtout << data.InputParameterNames[12] << " " << data.InternalAcc << "\\\\";
  data.txtout << data.InputParameterNames[13] << " " << data.K << "\\\\";
  data.txtout << data.InputParameterNames[14]; for(int i=0;i<data.K;i++){ data.txtout << " " << data.Resources[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[15] << " " << data.DensityExpression << "\\\\";
  data.txtout << data.InputParameterNames[16]; for(int i=0;i<data.mpp.size();i++){ data.txtout << " " << data.mpp[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[17]; for(int i=0;i<data.Mpp.size();i++){ data.txtout << " " << data.Mpp[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[18]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.tauVec[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[19]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.tVec[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[20]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.TVec[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[21]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.Environments[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[22] << " " << data.stretchfactor << "\\\\";
  data.txtout << data.InputParameterNames[23] << " " << data.Wall << "\\\\";
  data.txtout << data.InputParameterNames[24] << " " << data.Noise << "\\\\";
  data.txtout << data.InputParameterNames[25] << " " << data.InterpolVQ << "\\\\";
  data.txtout << data.InputParameterNames[26]; for(int i=0;i<data.Interactions.size();i++){ data.txtout << " " << data.Interactions[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[27] << " " << data.incrementalV << "\\\\";
  data.txtout << data.InputParameterNames[28]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.muVec[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[29] << " " << data.DeltamuModifier << "\\\\";
  data.txtout << data.InputParameterNames[30]; for(int i=0;i<3;i++){ data.txtout << " " << data.Mixer[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[31]; for(int i=0;i<data.S;i++){ data.txtout << " " << data.thetaVec[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[32] << " " << data.SCcriterion << "\\\\";
  data.txtout << data.InputParameterNames[33] << " " << data.maxSCcount << "\\\\";
  data.txtout << data.InputParameterNames[34] << " " << data.method << "\\\\";  
  data.txtout << data.InputParameterNames[35] << " " << data.DelFieldMethod << "\\\\";  
  data.txtout << data.InputParameterNames[36] << " " << data.MovieQ << "\\\\";
  data.txtout << data.InputParameterNames[37]; for(int i=0;i<data.Schedule.size();i++){ data.txtout << " " << data.Schedule[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[38]; for(int i=0;i<data.EkinTypes.size();i++){ data.txtout << " " << data.EkinTypes[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[39] << " " << data.alpha << "\\\\";
  data.txtout << data.InputParameterNames[40] << " " << data.beta << "\\\\";
  data.txtout << data.InputParameterNames[41] << " " << data.gamma << "\\\\";
  data.txtout << data.InputParameterNames[42] << " " << data.gammaH << "\\\\";
  data.txtout << data.InputParameterNames[43] << " " << data.degeneracy << "\\\\";
  data.txtout << data.InputParameterNames[44]; for(int i=0;i<data.regularize.size();i++){ data.txtout << " " << data.regularize[i]; } data.txtout << "\\\\";
  data.txtout << data.InputParameterNames[45] << " " << data.RegularizationThreshold << "\\\\";
  data.txtout << data.InputParameterNames[46] << " " << data.Print << "\\\\";
}

void ActivateDefaultParameters(datastruct &data){
  
  data.ConsumedResources.resize(data.K);
  
  data.Competition.resize(data.S*data.S); fill(data.Competition.begin(), data.Competition.end(), 0.);
  data.RepMut.resize(data.S*data.S); fill(data.RepMut.begin(), data.RepMut.end(), 0.);
  data.Amensalism.resize(data.S*data.S); fill(data.Amensalism.begin(), data.Amensalism.end(), 0.);
  
  data.Consumption.resize(data.K);
  for(int k=0;k<data.K;k++){
    data.Consumption[k].resize(data.S);
    for(int s=0;s<data.S;s++) data.Consumption[k][s] = 1.23456789;
  }
  
  vector<double> tmpvec2(data.S); fill(tmpvec2.begin(), tmpvec2.end(), 1.);
  data.fitness = tmpvec2;
  
}

void InitializeHardCodedParameters(datastruct &data, taskstruct &task){
	StartTimer("InitializeHardCodedParameters",data);
  
	data.txtout << "\\newpage \n";
	data.txtout << "**** HardCodedParameters ****\\\\";
	data.EdgeLength = data.steps+1; //number of grid points in each cartesian direction; choose data.steps even !!!
	data.DDIM = (double)data.DIM;
	data.GridSize = (int)(pow((double)data.EdgeLength,data.DDIM)+0.5);
	data.CentreIndex = (data.GridSize-1)/2;
	if(data.Interactions[0]>=1000 && data.Interactions[0]<2000) data.FLAGS.SCO = 1;
	if(data.FLAGS.SCO) data.edge = (double)data.steps;//ensure that data.Deltax==1. for proper interpretation of density (necessary?)
	data.area = pow(data.edge,data.DDIM);
	//data.Wall *= (1.+PI*MP);//to avoid the wall on a grid point, necessary?
	data.rW = 0.5*data.edge; if(ABS(data.Wall)>MP) data.rW *= ABS(data.Wall);
	data.WallArea = data.area; if(data.Wall>MP) data.WallArea = PI*pow(data.rW,data.DDIM); else data.WallArea = pow(data.rW,data.DDIM);
	data.HDI.resize(data.steps/2+1);
	for(int hdi=0;hdi<data.HDI.size();hdi++){
		if(data.DIM==1) data.HDI[hdi] = data.CentreIndex+hdi;
		else if(data.DIM==2) data.HDI[hdi] = data.CentreIndex+hdi*data.EdgeLength+hdi;
		else if(data.DIM==3) data.HDI[hdi] = data.CentreIndex+hdi*data.EdgeLength*data.EdgeLength+hdi*data.EdgeLength+hdi;
		//cout << data.GridSize << " " << data.CentreIndex << " " << hdi << " " << data.HDI[hdi] << endl;//ToDo: Test for 1D,3D
	}
	//int hdi=0; for(int i=0;i<data.GridSize;i++) if(HalfDiagonalQ(i,data.DIM,data.steps)){ data.HDI[hdi] = i; hdi++; }
	if(data.DIM==1){ data.frame = {{-data.edge/2.,data.edge/2.}}; }
	else if(data.DIM==2) data.frame = {{-data.edge/2.,data.edge/2.,-data.edge/2.,data.edge/2.}};
	else if(data.DIM==3) data.frame = {{-data.edge/2.,data.edge/2.,-data.edge/2.,data.edge/2.,-data.edge/2.,data.edge/2.}};
	data.Lattice = rGrid(data.DIM,data.steps,data.frame);
	data.lattice = rGrid(1,data.steps,data.frame);
	data.VecAt = GetVecsAt(data);
	data.Deltax = data.edge/((double)(data.steps)); data.txtout << "Deltax " << data.Deltax << "\\\\";
	data.Deltax2 = data.Deltax*data.Deltax;
	data.k0 = -PI/data.Deltax; data.txtout << "k0 " << data.k0 << "\\\\";
	data.Deltak=2.*PI/data.edge;
	if(data.DIM==1){ data.kframe = {{data.k0,-data.k0}}; }
	else if(data.DIM==2) data.kframe = {{data.k0,-data.k0,data.k0,-data.k0}};
	else if(data.DIM==3) data.kframe = {{data.k0,-data.k0,data.k0,-data.k0,data.k0,-data.k0}};
	data.kLattice = rGrid(data.DIM,data.steps,data.kframe);
	data.kVecAt = GetkVecsAt(data);
	GetNorm2kVecsAt(data);
	data.minmaxfi.resize(data.S);
	//UpdateApVec(1,data);  
	data.AdmixSignal.resize(data.S); data.OldAdmixSignal.resize(data.S); fill(data.AdmixSignal.begin(), data.AdmixSignal.end(), 1.); fill(data.OldAdmixSignal.begin(), data.OldAdmixSignal.end(), 1.);
	data.thetaVecOriginal = data.thetaVec;
	data.RelAccOriginal = data.RelAcc;
	data.gammaOriginal = data.gamma;
	data.NoiseOriginal = data.Noise; if( (task.Type==8 || task.Type==88) && task.count[0]==data.TaskParameterCount-1 ) data.NoiseOriginal = 0.;
    if(data.Units==1) data.U = 1.; else if(data.Units==2) data.U = 7.619964158443410;/*U=hbar^2/(m*Length^2*Energy)*/ data.txtout << "U " << data.U << "\\\\";


    //data.FLAGS.AdiabaticEnergy = 1;//kinetic energy formula based on Cangi2011_Electronic Structure via Potential Functional Approximations

	if(data.DensityExpression==11) data.FLAGS.KD = 1;
	if(data.FLAGS.KD){
      	//BEGIN USER INPUT for KD
      	data.KD.ExportCleanDensity = 1.0e+2;//<1.0e-12: don't modify anything --- >1.0e-12: plot only densities with absolute values smaller than ExportCleanDensity
      	data.txtout << "KD.ExportCleanDensity " << data.KD.ExportCleanDensity << "\\\\";
      	data.KD.UseTriangulation = 0;//0: switch off all triangulation code --- 1: use triangulation for KD, for now only used together with METHOD==3
      	data.txtout << "KD.UseTriangulation " << data.KD.UseTriangulation << "\\\\";
      	data.KD.UpdateTriangulation = 1;//update the existing GoodTriangle file, for now only used together with METHOD==3
      	data.txtout << "KD.UpdateTriangulation " << data.KD.UpdateTriangulation << "\\\\";
      	data.KD.IntermediateCleanUp = 0;//clean up after each update of Triangulation (doesn't work, so keep it at 0)
      	data.txtout << "KD.IntermediateCleanUp " << data.KD.IntermediateCleanUp << "\\\\";
      	data.KD.ReevaluateTriangulation = 100;//>0: up- or downgrade the triangles' quality in the (eventually) produced GoodTriangle file (relative to the old GoodTriangle file). This function separates the quality from (i) the stored triangles and (ii) the triangles (checked with NumChecks) during the current instance of the calculation. When used on a file with quality==ReevaluateTriangulation, then the existing triangles are reevaluated (also with new random points if ReevaluateTriangulation>4)
      	data.txtout << "KD.ReevaluateTriangulation " << data.KD.ReevaluateTriangulation << "\\\\";
      	data.KD.NumChecks = 100;//1: minimum (centroid) --- >1: more points to check in GoodTriangleQ. Discard triangles that do not pass NumChecks
      	data.txtout << "KD.NumChecks " << data.KD.NumChecks << "\\\\";
      	int FocalDensitySteps = data.steps/4;//data.steps;//min(512,data.steps);//determines grid size for density n7 (should be less than data.steps), will be truncated automatically if necessary, for now only used together with Getn7()-METHOD==3.
      	data.txtout << "FocalDensitySteps " << FocalDensitySteps << "\\\\";
      	data.KD.MergerRatioThreshold = 0.1;//minimum percentage of #CurrentMergers to merge good triangles again
      	data.txtout << "KD.MergerRatioThreshold " << data.KD.MergerRatioThreshold << "\\\\";
        data.KD.RemovalFraction = 0.6;//percentage of smallest good triangles to be removed after each merging
        data.txtout << "KD.RemovalFraction " << data.KD.RemovalFraction << "\\\\";
        data.KD.RetainSurplusFraction = 0.1;//percentage of new good triangles (relative to initial #GoodTriangles) to be retained
        data.txtout << "KD.RetainSurplusFraction " << data.KD.RetainSurplusFraction << "\\\\";
      	//END USER INPUT for KD
      	data.KD.CoarseGridSize = min((int)(POW((double)(FocalDensitySteps+1),data.DIM)+0.5),data.GridSize);
      	if(FocalDensitySteps<data.steps){
        	if(data.KD.CoarseGridSize<data.GridSize){
				set<int> CoarseIndexSet;
				while(CoarseIndexSet.size()<data.KD.CoarseGridSize) CoarseIndexSet.insert((int)(data.RNpos(data.MTGEN)*data.GridSize));
				data.KD.CoarseIndices.resize(0);
				data.KD.CoarseIndices.insert(data.KD.CoarseIndices.end(),CoarseIndexSet.begin(),CoarseIndexSet.end());
				PRINT("InitializeHardCodedParameters: data.KD.CoarseIndices = " + partial_vec_to_str(data.KD.CoarseIndices,0,min(10,(int)data.KD.CoarseIndices.size())) + "... " + partial_vec_to_str(data.KD.CoarseIndices,max(0,(int)data.KD.CoarseIndices.size()-10),(int)data.KD.CoarseIndices.size()-1),data);
        	}
        }
        else if(FocalDensitySteps==data.steps && data.Symmetry!=1){
          	data.Symmetry = 0;
            data.KD.CoarseIndices.resize(data.GridSize);
            for(int i=0;i<data.GridSize;i++) data.KD.CoarseIndices[i] = i;
          	PRINT("InitializeHardCodedParameters: Warning !!! data.Symmetry changed to data.Symmetry==0",data);
        }
	}
	data.txtout << "Symmetry " << data.Symmetry << "\\\\";

  data.SymmetryMask.resize(data.GridSize,0);
  if(data.Symmetry==0) fill(data.SymmetryMask.begin(),data.SymmetryMask.end(),1);
  else fill(data.SymmetryMask.begin(),data.SymmetryMask.end(),0);
  if(data.Symmetry==1) for(int hdi=0;hdi<data.HDI.size();hdi++) data.SymmetryMask[data.HDI[hdi]] = 1;
  else if(data.Symmetry==2) data.SymmetryMask[data.CentreIndex] = 1;
  else if(data.Symmetry==3){
    if((data.steps % 32)!=0){
      data.Symmetry = 0; fill(data.SymmetryMask.begin(),data.SymmetryMask.end(),1);
      data.warningCount++; PRINT("InitializeHardCodedParameters: Warning !!! data.Symmetry==3, but #steps not multiple of 32; data.Symmetry changed to data.Symmetry==0",data);
    }
    else{
      int maxstride = data.steps/32, n = data.steps/maxstride+1, EL = data.EdgeLength, maxStrideLevel = (int)(log((double)maxstride)/log(2.)+0.5);
      data.StrideLevel = maxStrideLevel; data.MaxStrideLevel = maxStrideLevel;
      GetStridelattice(data);      
      if(data.DIM==1){
	data.TmpSpline1D.clear(); data.TmpSpline1D.resize(data.S);
	for(int i=0;i<n;i++) data.SymmetryMask[data.stride*i] = 1;
      }
      else if(data.DIM==2){
	data.TmpSpline2D.clear(); data.TmpSpline2D.resize(data.S);
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<n;i++) for(int j=0;j<n;j++) data.SymmetryMask[(data.stride*i)*EL+(data.stride*j)] = 1;
      }
      else if(data.DIM==3){
	data.TmpSpline3D.clear(); data.TmpSpline3D.resize(data.S);
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) data.SymmetryMask[(data.stride*i)*EL*EL+(data.stride*j)*EL+(data.stride*k)] = 1;
      }
      cout << "TmpSplines allocated" << endl;
    }
  }
  else if(data.FLAGS.KD){
    if(data.Symmetry==4){
		int c = 0;
		for(int i=0;i<data.GridSize;i++){
			if(i==data.KD.CoarseIndices[c]){
				data.SymmetryMask[i] = 1;
				c++;
                if(c==data.KD.CoarseIndices.size()) break;
			}
		}
    }
  }
  for(int i=0;i<data.GridSize;i++) if(data.SymmetryMask[i]) data.IndexList.push_back(i);

  //PERIODIC SYSTEMS
  if(data.FLAGS.periodic){
    data.TaperOff = 0; data.txtout << "TaperOff " << data.TaperOff << "\\\\";
    data.RegularizeCoulombSingularity = 0; data.txtout << "RegularizeCoulombSingularity " << data.RegularizeCoulombSingularity << "\\\\";
  }

  data.AverageDensity.resize(data.S); for(int s=0;s<data.S;s++) data.AverageDensity[s] = data.Abundances[s]/data.area;
  data.AccumulatedAbundances = accumulate(data.Abundances.begin(),data.Abundances.end(),0.0);

	if(task.Type==61){//DynDFTe (inherited from data.TaskType)
		TASK.Type = task.Type;
		TASK.DynDFTe.mode = task.DynDFTe.mode;
		TASK.DynDFTe.InitializationPhase = task.DynDFTe.InitializationPhase;
		if(TASK.DynDFTe.mode==0){
			TASK.DynDFTe.SnapshotID = {{0,3,5,10,20,30,40,45,50,55,60,70,80,100}};//percentages of maxSCcount 
			for(int sid=0;sid<TASK.DynDFTe.SnapshotID.size();sid++) TASK.DynDFTe.SnapshotID[sid] = (int)((double)TASK.DynDFTe.SnapshotID[sid]*(double)data.maxSCcount/100.+MP);
			TASK.DynDFTe.VectorScale.clear(); TASK.DynDFTe.VectorScale.resize(0);
			TASK.DynDFTe.n0.resize(data.S);
			TASK.DynDFTe.nf.resize(data.S);
			TASK.DynDFTe.v.resize(data.S);
			TASK.DynDFTe.g.resize(data.S);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int s=0;s<data.S;s++){
				TASK.DynDFTe.n0[s].resize(data.GridSize);
				TASK.DynDFTe.nf[s].resize(data.GridSize);
				TASK.DynDFTe.v[s].resize(data.GridSize);
				TASK.DynDFTe.g[s].resize(data.GridSize);
				for(int i=0;i<data.GridSize;i++){
					TASK.DynDFTe.v[s][i].resize(data.DIM);
					TASK.DynDFTe.g[s][i].resize(data.DIM);
				}
			}
		}
	}
  
  if(data.TaskType==100){//1p-exact DFT, ToDo: more than one species in all dimensions, and other than unpolarized spin-1/2 (-> account for degeneracies)
    TASK.ex.settings.clear(); TASK.ex.settings.resize(21);
    //retrieved from mpDPFT.input:
    TASK.ex.System = data.System;
    TASK.ex.Abundances = data.Abundances;
    TASK.ex.RelAcc = data.RelAcc;
    if(data.Units==1){ TASK.ex.UNITenergy = 1.; TASK.ex.UNITlength = 1.; } //ToDo    
    TASK.ex.settings[2] = data.Interactions[0];//type of Eint --- inherited from data.Interactions
    TASK.ex.settings[0] = (int)(data.Mpp[0]+0.5);//number of one-particle-Hamiltonian-levels to be used
    for(int l=0;l<TASK.ex.settings[0];l++) TASK.ex.ZeroVec.push_back(0.);
    TASK.ex.settings[1] = data.DensityExpression;//type of 1-RDM expression to be calculated --- 100: diagonal RDM --- 101: MIT special case --- 102: Berge/Mikolaj algorithm --- 103: Berge's TF-RDM ('cot'-version) --- 104: manual input in Plugin_1pEx.cpp --- 105: HF(N=2)-type [sqrt(n_i*n_j)/N] --- 106: random RDM via Schur-Horn algorithm (differential equation Chu1995) --- 107: Berge's TF-RDM (cos-version)
    
    //BEGIN USER INPUT for 1pEx-DFT
    TASK.ex.settings[3] = 3;//initialization of Occupation numbers --- 0: noninteracting filling --- 1: linear decrease --- 2: random(version1) --- 3: random(version2) --- 4: manual input in Plugin_1pEx.cpp --- 5: read from mpDPFT_IntermediateOPTresults.dat
    TASK.ex.settings[4] = 1;//(reset TASK.ex.settings[4] in individual optimization routines) minimize over --- 0: OccNum --- 1: OccNumAngles
    TASK.ex.settings[5] = 4;//method of finding nearest proper occupation numbers --- 0: Do nothing --- 1: rescaling and iterative redistribution (MIT) --- 2: simplex-type (Alex,rescaling) --- 3: simplex-type (Alex,projection) --- 4: randomly alternate Alex' rescaling & projection
    TASK.ex.settings[6] = 0;//penalty function --- 0: none --- 1: basic
    TASK.ex.settings[7] = -1;//HintMethod --- <0: load from "TabFunc_Hint.dat" --- 0: default(numerical) --- 1: 1D-contact-summation(Jonah) --- 2: 1D-contact-integration(Jonah) --- 3: 1D-contact-recursion(Jonah) --- 4: 1D-contact-integration(MIT)
    TASK.ex.settings[8] = 1;//minimization over (i) occupation numbers and (ii) phases --- 0: separate --- 1: combined(to be used for calculating spatial density in the end)
    TASK.ex.settings[9] = 1;//1: check&ensure rho1p consistency --- 0: don't --- 2: check&ensure rho1p consistency and print additional stuff
    TASK.ex.settings[10] = 1;//0: put phases to zero, e.g., for TF-analog or manual input (supply manually in-code) --- 1: optimize phases
    TASK.ex.settings[11] = 0;//level of self-optimization if SelfOptQ>0, obsolete!!!
    TASK.ex.settings[12] = 0;//post-processing after optimization --- 0: none --- 1: phases --- 2: phases and OccNum
    TASK.ex.settings[13] = 2;//use: 0: only direct interaction --- 1: only exchange interaction --- 2: both direct and exchange interaction
    TASK.ex.settings[14] = 0;//number of principal components for RBF-interpolation-based estimate of objective function
    TASK.ex.settings[15] = 0;/*0;*/ //--- 1: export Hint (to "mpDPFT_1pExDFT_Hint.dat" by default, or problem-specific) --- 0: don't
    TASK.ex.settings[17] = 0;//case-specific restrictions for Occupation Numbers, like spherical averaging --- 0: None --- 1: equal occnum (in magnetic quantum number) for atomic states of fixed n,l --- 2: equal occnum of all atomic states in shell n
    TASK.ex.settings[18] = 0;//kappa criterion of GenerateRho1pMixer
    TASK.ex.settings[19] = 0;//Freeze(>0): freeze mod(settings[19],100) levels at occ=2; use only S (Freeze<200), S&P (Freeze<300), S&P&D (Freeze<400), etc. levels -> freeze other levels at occ=0
    TASK.ex.settings[20] = 0;//2: include optimization over generalized 2x2-unitaries & ExtraPhases --- 1: include optimization over generalized 2x2-unitaries --- 0: don't
    TASK.ex.HintIndexThreshold = 1.0e-9;/*0.;*/ //threshold for vanishing interaction tensor element Hint_abcd --- 0.: include all Hint_abcd in 4-dimensional Eint-summation
    TASK.ex.CompensationFactor = 1.; //compensation factor to be multiplied to loaded "TabFunc_Hint.dat", see below --- 1.: default
    TASK.ex.Optimizers = {{/*-100*/101,100}};//list of optimizer types for consecutive optimization levels. ManualInputInGetOptOccNum(-100) --- CGD(100) --- PSO(101) --- LCO(102) --- AUL(103) --- GAO(104) --- cSA(105)
    //END USER INPUT for 1pEx-DFT
    
    //problem-specific parameters
    if(TASK.ex.settings[1]==103) TASK.ex.settings[9] = 0;//rho_tf is not consistent with Dirac-approximation anyway
    if(TASK.ex.settings[2]==10){
		TASK.ex.params.push_back(4.*data.beta);
		TASK.ex.CompensationFactor = data.beta/5.;
	}
	else if(TASK.ex.settings[2]==22) TASK.ex.params.push_back((data.alpha-1.)/(2.*data.Abundances[0]));
	else if(TASK.ex.settings[2]==23 || TASK.ex.settings[2]==24){
      TASK.ex.params.push_back(data.beta);//nuclear charge Z as prefactor included in the building block Tabcd of the nonrelativistic interaction tensor elements Hint[abcd], from the hydrogenic basis
      TASK.ex.params.push_back(data.alpha);//nuclear charge Z for (nonrelativistic) single-particle energies, for LoadE1p_X2C, and for LoadCMatrix_X2C
    }
    GetQuantumNumbers(TASK.ex);
	if(TASK.ex.settings[20]>0){ TASK.ex.settings[8] = 1; if(TASK.ex.HintIndexThreshold < MachinePrecision) TASK.ex.HintIndexThreshold = MachinePrecision; }
    TASK.ex.settings[16] = data.ompThreads;
    if(TASK.ex.settings[10]==0) TASK.ex.settings[12] = 0;
    if(TASK.ex.settings[1]==104) TASK.ex.Optimizers[0] = -100;
    if(TASK.ex.settings[8]==1) TASK.ex.Optimizers[1] = 0;    
    if((double)(TASK.ex.settings[0])<0.5*data.Abundances[0]){ data.errorCount++; cout << "InitializeHardCodedParameters: Error!!! Input 'Abundances' requires more energy levels for 1p-exact DFT." << endl; data.ABORT = true; }
    if(TASK.ex.settings[1]==106){ TASK.ex.SHrandomSeedQ = false; TASK.ex.SHdt = 0.001; TASK.ex.SHmaxCount = (int)(100./TASK.ex.SHdt), TASK.ex.SHConvergenceCrit = data.InternalAcc; }
    if(TASK.ex.System==102){
		LoadE1p_X2C(TASK.ex);
		LoadCMatrix_X2C(TASK.ex);
	}
	data.txtout << "ex.settings " << IntVec_to_CommaSeparatedString(TASK.ex.settings) << "\\\\";
	data.txtout << "ex.params " << vec_to_str(TASK.ex.params) << "\\\\";
    data.txtout << "ex.HintIndexThreshold " << TASK.ex.HintIndexThreshold << "\\\\";
	data.txtout << "ex.CompensationFactor " << TASK.ex.CompensationFactor << "\\\\";
    data.txtout << "ex.Optimizers " << IntVec_to_CommaSeparatedString(TASK.ex.Optimizers) << "\\\\";
    PRINT("InitializeHardCodedParameters: 1p-exact DFT initialized",data);
  }
  
  //DEFAULT
  ActivateDefaultParameters(data);
  data.SplineType = 1; data.txtout << "SplineType " << data.SplineType << "\\\\";
  data.InterpolAcc = 1.0e-1; data.txtout << "InterpolAcc " << data.InterpolAcc << "\\\\";
  data.UpdateSymMaskFraction = 0.05; data.txtout << "UpdateSymMaskFraction " << data.UpdateSymMaskFraction << "\\\\";
  data.FourierFilterPercentage = 50; data.txtout << "FourierFilterPercentage " << data.FourierFilterPercentage << "\\\\";
  data.MaxIteration = 4; data.txtout << "MaxIteration " << data.MaxIteration << "\\\\";
  double SmearingExtent = /*0.05*data.edge;*/6.*data.Deltax; data.txtout << "SmearingExtent " << SmearingExtent << " = " << (int)(SmearingExtent/data.Deltax+MP) << "*Deltax" << "\\\\";
  data.ConvolutionSigma = -0.5*pow(SmearingExtent,2.); data.txtout << "ConvolutionSigma " << data.ConvolutionSigma << "\\\\";
  data.txtout << "ConjugateGradientDescent: ConjGradParam ( epsf,epsg,epsx,diffstep ) = ( " << vec_to_str(task.ConjGradParam) << ")\\\\";
  if(data.DensityExpression==5) data.CalculateEnergyQ = false;

  //ENVIRONMENTS
  data.rpow = 1.;/*2.;*/ data.txtout << "rpow " << data.rpow << "\\\\";//for Environment 1
  data.flatEnv = 0.; data.txtout << "flatEnv " << data.flatEnv << "\\\\";//value of (flat) Environment 0
  data.AutomaticTratio = 1.0e-6; data.txtout << "AutomaticTratio " << data.AutomaticTratio << "\\\\"; //if(data.TVec[0]<0) T -> AutomaticTratio*NucleiEnergy
  
  //depth-parameter for Paul Ayer's all-electron Coulomb pseudopotential
  data.AllElectronPP = 2;//0: minimal Coulomb-approximation --- 1: Ayer (default) --- 2: Gygi/Lehtola
  double STEPS = (double)data.steps;
  if( data.TaskType==8 && (data.DensityExpression==6 || data.DensityExpression==9 || data.DensityExpression==10) ) STEPS = (double)data.steps*pow(2.,(double)(data.TaskParameterCount-task.count[0]));
  data.AllElectronParam = SQRTPI*(-923.*data.edge+2000.*STEPS)/(16.*data.edge*(125.+98.*SQRTPI));//automatic
  //data.AllElectronParam = SQRTPI*(-923.*data.edge+2000.*384.)/(16.*data.edge*(125.+98.*SQRTPI));//manual
  //data.AllElectronParam = 10.;//manual
  data.txtout << "AllElectronPP " << data.AllElectronPP << "\\\\";
  data.txtout << "AllElectronParam " << data.AllElectronParam << "\\\\";
  
  //INTERACTIONS
  data.libxcPolarization = XC_UNPOLARIZED;/*XC_UNPOLARIZED || XC_POLARIZED*/ data.txtout << "libxcPolarization " << data.libxcPolarization << "\\\\";
  data.incrementalVOriginal = data.incrementalV;/*default: 1.0; maybe start with 0.02 or 0.05*/
  if(data.TaskType==8 && (data.DensityExpression==6 || data.DensityExpression==9 || data.DensityExpression==10) && data.incrementalV<1. && task.count[0]>0) data.incrementalV = 1.;

    
  //data.Consumption = {{1.}};
  //data.Consumption = {{1.0e-1},{1.0e-1}};  
  // data.Consumption = {{1.0e-1},{1.0e-1},{1.0e-1},\
  //			 {2.0e-1},{5.0e-1},{9.0e-1}};
  if(data.System==0){
    data.zeta = 1.; 
  }
  else if(data.System==1){//Kenkel1991, see Ecology_DFT_25f_Kenkel.nb, matrix(nu_ks), and Ecology_DFT_25h_Kenkel.nb
    data.zeta = 1.;
    data.kappa = 1.;
  //ADJUST Consumption, Resources
  //data.Consumption = {{1.000000},{0.566896},{0.0493685}};//Poa
  //data.Consumption = {{0.435516},{1.000000},{1.1592500}};//Hord
  //data.Consumption = {{0.349545},{0.629462},{1.0000000}};//Pucc
//   data.Consumption = {{1.0000000,0.0,0.0},\
// 		      {0.0,1.000000,0.0},\
// 		      {0.0,0.0,1.000000}};//Poa & Hord & Pucc, diagonal consumption
//   data.Consumption = {{1.0000000,0.435516,0.349545},\
// 		      {0.5668960,1.000000,0.629462},\
// 		      {0.0493685,1.159250,1.000000}};//Poa & Hord & Pucc, Lennard-Jones-resource-term

    //Poa & Hord & Pucc, harmonic resource term
    for(int a=0;a<data.Interactions.size();a++){
      if(data.Interactions[a]==7){//AMENSALISM
	data.Consumption = {{1.0000000000000000,0.08657919582208944,0.165627296278239},\
			    {0.0021627055146424797,1.0000000000000000,0.3209092840461029},\
			    {0.007094916705906815,1.1592700118253083,1.0000000000000000}};
	fill(data.tauVec.begin(),data.tauVec.end(),0.1145986308992319);
	data.alpha = 0.05646246211548315;
	data.sigma = -1.0056655773230454;   
      }
      else if(data.Interactions[a]==18){//REPULSION
	data.Consumption = {{1.0000000000000000,0.1125565693805481,0.28846402121763404},\
			    {0.0019510994577284896,1.0000000000000000,0.04673596280196924},\
			    {0.0009574539011304631,1.1587925607896234,1.0000000000000000}};
	fill(data.tauVec.begin(),data.tauVec.end(),0.09337960300610507);
	data.beta = 0.019163612076962478;
	data.sigma = -2.6510386165390463;  
      }
      else if(data.Interactions[a]==4 || data.Interactions[a]==19 || data.Interactions[a]==20){//PARASITISM
	
	data.Consumption = {{1.0000000000000000,0.4199192010191171,0.3480308653747571},\
			    {0.000611785497331405,1.0000000000000000,0.6284159593563743},\
			    {0.9803304867552304,0.9473195781550088,1.0000000000000000}};
	fill(data.tauVec.begin(),data.tauVec.end(),0.08109377491188259);
	if(data.Interactions[a]==4) data.gamma = 0.011050947991975308;//PARASITISM
	else data.gamma = 0.00371684122;//HartreePARASITISM
	data.sigma = -2.805784542830516;
      }
      else if(data.S==1 && data.Interactions.size()==1 && data.Interactions[a]==2){//Poa or Pucc alone
	//AMENSALISM parameters
// 	data.Consumption = {{1.0000000000000000},{0.0021627055146424797},{0.007094916705906815}};
// 	data.tauVec[0] = 0.1145986308992319;
// 	data.sigma = -1.0056655773230454;
	//REPULSION parameters
// 	data.Consumption = {{1.0000000000000000},{0.0019510994577284896},{0.0009574539011304631}};
// 	data.tauVec[0] = 0.09337960300610507;
// 	data.sigma = -2.6510386165390463;
	//PARASITISM parameters
	//data.Consumption = {{1.0000000000000000},{0.000611785497331405},{0.9803304867552304}};//Poa
	data.Consumption = {{0.3480308653747571},{0.6284159593563743},{1.0000000000000000}};//Pucc
	data.tauVec[0] = 0.08109377491188259;
	data.sigma = -2.805784542830516;	
      }
    }
   
  }		      
  else if(data.System==2){//TygerWeb
    data.zeta = 1.0e+0;
    data.sigma = -4.;
    if(data.S==1){
      //SPECIES = {1(*Funges*) or 2(*Dear*) or 4(*Snale*)};
/*      data.Consumption = {{1.},{0.},{1.}};*/      
      //SPECIES = {5(*Trea*) or 6(*Grazz*)};
      data.Consumption = {{1.},{1.},{1.}};
    }
    else if(data.S==2){
      //SPECIES = {1(*Funges*), 5(*Trea*)}; species 1,5 and interactions (2 16)
      data.Consumption = {{1.,1.},{0.,1.},{1.,1.}};
      data.RepMut = {{0.},{0.},{-1.},{0.}};
    }    
    else if(data.S==3){
      //SPECIES = {1(*Funges*), 5(*Trea*), 6(*Grazz*)}; species 1,5,6 and interactions (2 6 16)
      data.Consumption = {{1.,1.,1.},{0.,1.,1.},{1.,1.,1.}};
      data.Amensalism = {{0.},{0.},{0.},{0.},{0.},{1.},{0.},{0.},{0.}};
      data.RepMut = {{0.},{0.},{1.},{-1.},{0.},{0.},{0.},{0.},{0.}};
    }
    else if(data.S==4){
      //SPECIES = {1(*Funges*), 2(*Dear*), 5(*Trea*), 6(*Grazz*)}; species 1,2,5,6 and interactions (2 6 16)
      data.Consumption = {{1.,1.,1.,1.},\
			  {0.,0.,1.,1.},\
			  {1.,1.,1.,1.}};
      data.Amensalism = {{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{1.},\
			 {0.},{0.},{0.},{0.}};
      data.RepMut = {{0.},{0.},{0.},{1.},\
		     {0.},{0.},{0.},{0.},\
		     {-1.},{-1.},{0.},{0.},\
		     {0.},{0.},{0.},{0.}};    
    }
    else if(data.S==5){
      //SPECIES = {1(*Funges*), 2(*Dear*), 4(*Snale*) 5(*Trea*), 6(*Grazz*)}; species 1,2,4,5,6 and interactions (2 5 6 16)
      data.Consumption = {{1.,1.,1.,1.,1.},\
			  {0.,0.,0.,1.,1.},\
			  {1.,1.,1.,1.,1.}};
      data.Amensalism = {{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{1.},\
			 {0.},{0.},{-1.},{0.},{0.}};
      data.RepMut = {{0.},{0.},{0.},{0.},{1.},\
		     {0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},\
		     {-1.},{-1.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.}};
      data.Competition = {{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},\
			  {1.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.}};		   
    }
    else if(data.S==6){
      //SPECIES = {1(*Funges*), 2(*Dear*), 3(*Bore*), 4(*Snale*), 5(*Trea*), 6(*Grazz*)}; species 1-6 and all interactions (2 5 6 16), including predation
      data.Consumption = {{0.,0.,1.,0.,0.,0.},\
			  {1.,1.,1.,1.,1.,1.},\
			  {0.,0.,0.,0.,1.,1.},\
			  {1.,1.,1.,1.,1.,1.}};
      data.Amensalism = {{0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{-1.},{0.},{0.},{1.},\
			 {0.},{0.},{0.},{-1.},{0.},{0.}};
      data.RepMut = {{0.},{0.},{0.},{0.},{0.},{1.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},\
		     {-1.},{-1.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.}};
      data.Competition = {{0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.},\
			  {1.},{0.},{0.},{0.},{0.},{0.},\
			  {1.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.}};
    }
    else if(data.S==7){
      //SPECIES = {1(*Funges*), 2(*Dear*), 3(*Bore*), 4(*Snale*), 5(*Trea*), 6(*Grazz*), 7(*Tyger*)}; all species and all interactions (2 5 6 16), including predation
      data.Consumption = {{0.,0.,1.,0.,0.,0.,0.},\
			  {0.,0.,0.,0.,0.,0.,5.},\
			  {0.,0.,0.,0.,0.,0.,1.},\
			  {1.,1.,1.,1.,1.,1.,0.},\
			  {0.,0.,0.,0.,1.,1.,0.},\
			  {1.,1.,1.,1.,1.,1.,0.}};
      data.Amensalism = {{0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			 {0.},{0.},{-1.},{0.},{0.},{1.},{0.},\
			 {0.},{0.},{0.},{-1.},{0.},{0.},{0.},\
			 {0.},{0.},{0.},{0.},{0.},{0.},{0.}};
      data.RepMut = {{0.},{0.},{0.},{0.},{0.},{1.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
		     {-1.},{-1.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
		     {0.},{0.},{0.},{0.},{0.},{0.},{0.}};
      data.Competition = {{0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {1.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {1.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{0.},{0.},{0.},{0.},{0.},{0.},\
			  {0.},{1.},{1.},{0.},{0.},{0.},{0.}};					
    }
  }
  else if(data.FLAGS.ForestGeo){//ForestGeo
    data.zeta = 1.;
    
    //Start-values for fit parameters, will get overwritten by RetrieveTask (TaskType 44 & 444)
    int NumGlobalParam = TASK.NumGlobalParam, NumEnvParam = TASK.NumEnvParam, NumResParam = data.Mpp.size(), NumIntParam = TASK.NumIntParam, NumberOfFitParametersPerSpecies = NumEnvParam+NumResParam+NumIntParam, NumMu = 0;
    if(data.RelAcc<MP) NumMu = data.S;
    
    data.sigma = -40.; 
    //Environmental parameters: Prefactors for altitude & pH, and their means
    data.mpp.clear(); data.mpp.resize(data.S*NumEnvParam); for(int s=0;s<data.S;s++) for(int m=0;m<NumEnvParam;m++) data.mpp[NumEnvParam*s+m] = 1.0e+5;
    //Nutrient requirements for Resources: 0:Al --- 1:B --- 2:Ca --- 3:Cu --- 4:Fe --- 5:K --- 6:Mg --- 7:Mn --- 8:P --- 9:Zn --- 10:N --- 11:AggregatedResource(1-7 & 9, only for L&R)
    vector<int> resindlist(data.Mpp.size()); for(int r=0;r<resindlist.size();r++) resindlist[r] = (int)(data.Mpp[r]+0.5);
    if(resindlist.size()!=data.K){ PRINT("InitializeHardCodedParameters: Error!!! Dimensional mismatch (ResourceIndexList)." + to_string(data.K) + " " + vec_to_str(resindlist),data); data.ABORT = true; }
    else{ data.ResourceIndexList.clear(); data.ResourceIndexList.resize(data.K); data.ResourceIndexList = resindlist; }
    if(data.System==31 || data.System==33) for(int s=0;s<data.S;s++) for(int k=0;k<data.K;k++) data.Consumption[k][s] = TASK.AbsResL[resindlist[k]]/data.AccumulatedAbundances;
    else if(data.System==32 || data.System==34) for(int s=0;s<data.S;s++) for(int k=0;k<data.K;k++) data.Consumption[k][s] = TASK.AbsResR[resindlist[k]]/data.AccumulatedAbundances;
    else if(data.System==35 || data.System==36) for(int s=0;s<data.S;s++) for(int k=0;k<data.K;k++) data.Consumption[k][s] = TASK.AbsRes[resindlist[k]]/data.AccumulatedAbundances;
    //for(int s=0;s<data.S;s++) for(int k=0;k<data.K;k++) data.Consumption[k][s] = 1./((double)data.S*data.Abundances[s]);
    //Interaction parameters: Fitness proxies for intra-species repulsion/attraction, amensalism/commensalism, repulsion/mutualism, and asymmetric interactions
    data.MPP.clear(); data.MPP.resize(data.S);
    for(int s=0;s<data.S;s++){
      data.MPP[s].resize(NumIntParam);
      //for(int i=0;i<NumIntParam;i++) data.MPP[s][i] = 1.0e-1;
      if(NumIntParam>0) data.MPP[s][0] = 1.0e+0;//Dispersal
      if(NumIntParam>1) data.MPP[s][1] = 1.0e+2;//Amensalism[Interactions=6]
      if(NumIntParam>2) data.MPP[s][2] = 1.0e+2;//RepMut[Interactions=16]
      if(NumIntParam>3) data.MPP[s][3] = 1.0e+2;//Competition[Interactions=5]        
/*      if(NumIntParam>0) data.MPP[s][0] = 1.0e+0;//Dispersal
      if(NumIntParam>1) data.MPP[s][1] = 1.0e+1;//Amensalism[Interactions=6]
      if(NumIntParam>2) data.MPP[s][2] = 1.0e+1/50.;//RepMut[Interactions=16]
      if(NumIntParam>3) data.MPP[s][3] = 1.0e+1/2500.;//Competition[Interactions=5]  */      
/*      if(NumIntParam>0) data.MPP[s][0] = 1.0e+0;//Dispersal
      if(NumIntParam>1) data.MPP[s][1] = 1.0e+2;//Amensalism[Interactions=6]
      if(NumIntParam>2) data.MPP[s][2] = 1.0e+2;//RepMut[Interactions=16]
      if(NumIntParam>3) data.MPP[s][3] = 1.0e+3;//Competition[Interactions=5]  */    
    }
    
    task.AuxVec.clear(); task.AuxVec.resize(NumGlobalParam+data.S*NumberOfFitParametersPerSpecies+NumMu);
    if(NumGlobalParam>0) task.AuxVec[0] = data.sigma;
    for(int s=0;s<data.S;s++){
      for(int m=0;m<NumEnvParam;m++) task.AuxVec[NumGlobalParam+s*NumberOfFitParametersPerSpecies+m] = data.mpp[NumEnvParam*s+m];//prefactors for environments
      for(int k=0;k<NumResParam;k++) task.AuxVec[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+k] = data.Consumption[k][s];//nutrient requirements
      for(int m=0;m<NumIntParam;m++) task.AuxVec[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+m] = data.MPP[s][m];//fitness proxies
    }
    for(int s=0;s<NumMu;s++) task.AuxVec[NumGlobalParam+data.S*NumberOfFitParametersPerSpecies+s] = data.muVec[s];
    //cout << "IHCP: " << vec_to_str(task.AuxVec) << endl;

    data.maxSCcount = (int)(data.maxSCcount*sqrt((double)data.S));
    //data.maxSCcount = (int)(data.maxSCcount*pow((double)data.S,0.75));
    //data.maxSCcount = 200*data.S;
    //fill(data.thetaVec.begin(),data.thetaVec.end(),min(0.1,1./(10.*(double)data.S)));
  }
  
  //OUTPUT:
  data.OutPrecision = 16; data.txtout << "OutPrecision " << data.OutPrecision << "\\\\";
  if(task.Type==61) data.OutCut = {{-data.edge/2.},{-data.edge/2.},{data.edge/2.},{data.edge/2.}};//diagonal
  else if(data.FLAGS.ForestGeo) data.OutCut = {{-data.edge/2.},{-data.edge/2.},{data.edge/2.},{0.}};
  else{
	data.OutCut = {{-data.edge/2.},{0.},{data.edge/2.},{0.}};//x-axis, default
	//data.OutCut = {{0.},{-data.edge/2.},{0.},{data.edge/2.}};//y-axis
  }
  data.txtout << "OutCut " << vec_to_str(data.OutCut) << "\\\\";
 

  //HANDLING:
  data.MonitorFlag = false; data.txtout << "MonitorFlag " << data.MonitorFlag << "\\\\";
  abortQ(data);
  
  //SANITY CHECKS:
  for(int k=0;k<data.K;k++) if(data.Consumption.size() != data.K || data.Consumption[k].size() != data.S){ data.errorCount++; cout << "InitializeHardCodedParameters: Error!!! Dimensional mismatch (Consumption)." << endl; PRINT("InitializeHardCodedParameters: Error!!! Dimensional mismatch (Consumption).",data); data.ABORT = true; }
  if(data.Competition.size() != data.S*data.S){ data.errorCount++; cout << "InitializeHardCodedParameters: Error!!! Dimensional mismatch (Competition)." << endl; PRINT("InitializeHardCodedParameters: Error!!! Dimensional mismatch (Competition).",data); data.ABORT = true; }
  if(data.fitness.size() != data.S){ data.errorCount++; cout << "InitializeHardCodedParameters: Error!!! Dimensional mismatch (fitness)." << endl; PRINT("InitializeHardCodedParameters: Error!!! Dimensional mismatch (fitness).",data); data.ABORT = true; }  
  
  EndTimer("InitializeHardCodedParameters",data);
}

void GetEnv(int s,datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  int n = data.EdgeLength;
  
  //different types of environments:
  if(data.Environments[s]==0){//flat
    SetField(data,data.Env[s], data.DIM, n, data.flatEnv);
  }
  else if(data.Environments[s]==1){//isotropic (r^2)^data.rpow with mpp prefactor
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++) data.Env[s][i] = data.mpp[s]*pow(Norm2(data.VecAt[i]),data.rpow);
  }
  else if(data.Environments[s]==2){// |x| with mpp prefactor for 2D
    double x, x0 = -0.5*data.edge, deltax = data.edge/((double)data.steps);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
	data.Env[s][i*n+j] = data.mpp[s]*ABS(x0+(double)i*deltax);
      }
    }
  }
  else if(data.Environments[s]==3){
    //anisotropic 2DHO with mpp prefactor
    double eccentricity = data.mpp[data.S+s];
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      data.Env[s][i] = data.mpp[s]*((1.-eccentricity)*data.VecAt[i][0]*data.VecAt[i][0]+(1.+eccentricity)*data.VecAt[i][1]*data.VecAt[i][1]);
    }
/*    //anharmonic 2D oscillator with mpp prefactor
    double anharmonicity = 0.1;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      data.Env[s][i] = data.mpp[s]*(data.VecAt[i][0]*data.VecAt[i][0]+pow(ABS(data.VecAt[i][1]),2.+anharmonicity));
    } */   
  }
  else if(data.Environments[s]==4){// 2*r^4-3*r^2+3/2 with mpp prefactor for 2D
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++){
      int index; double r2;
      for(int j=0;j<n;j++){
	index = i*n+j;
	r2 = Norm2(data.VecAt[index]);
	data.Env[s][index] = data.mpp[s]*(2.*r2*r2-3.*r2+1.5);
      }
    }
  }
  else if(data.Environments[s]==5){//customized
    //from Paper_DFT_for_Ecology.nb, use mpp[0] as salinity level x:
//     double x = data.mpp[0], y;
//     if(s==0) y = 0.10422152*x-0.02496305*x*x+0.0023893238*x*x*x;
//     else if(s==1) y = 0.10158408*x+0.0133263*x*x-0.00072467381*x*x*x;
//     else if(s==2) y = 0.21896763*x-0.022876275*x*x+0.00099192024*x*x*x;
//     SetField(data,data.Env[s], data.DIM, n, y);
    
    //from Paper_DFT_for_Ecology.nb, use rVec[0] as salinity level x:
//     #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int i=0;i<n;i++){
//       double x,y; int index;
//       for(int j=0;j<n;j++){
// 	index = i*n+j;
// 	x = ABS(data.VecAt[index][0]);
// 	if(s==0) y = 0.10422152*x-0.02496305*x*x+0.0023893238*x*x*x;
// 	else if(s==1) y = 0.10158408*x+0.0133263*x*x-0.00072467381*x*x*x;
// 	else if(s==2) y = 0.21896763*x-0.022876275*x*x+0.00099192024*x*x*x;
// 	data.Env[s][index] = y;
//       }
//     }
    
	//atomtronics 3-well transport
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++){
		double x,y; int index;
		for(int j=0;j<n;j++){
			index = i*n+j;
			x = data.VecAt[index][0];
			y = data.VecAt[index][1];
			data.Env[s][index] = \
			- data.mpp[0]*EXP(-(data.mpp[3]*(POW(x - data.mpp[6],2) + POW(y - data.mpp[9] ,2))))\
			- data.mpp[1]*EXP(-(data.mpp[4]*(POW(x - data.mpp[7],2) + POW(y - data.mpp[10],2))))\
			- data.mpp[2]*EXP(-(data.mpp[5]*(POW(x - data.mpp[8],2) + POW(y - data.mpp[11],2))));
		}
    }	
  }
  else if(data.Environments[s]==6){//fruit flies quasi-1D chamber
    //int lVecCounter = 0;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++){
      double x,y; int index;
      for(int j=0;j<n;j++){
	index = i*n+j;
	x = data.VecAt[index][0];
	y = data.VecAt[index][1];
	if(ABS(y)>0.5*0.8/10.*data.edge) data.Env[0][index] = data.mpp[0] * 10000.*data.edge;
	else{//Steady-state conduction
	   //lVecCounter++;
	  //data.Env[0][index] = data.mpp[0] * 1000.*x;//linear
	  //data.Env[0][index] = data.mpp[0] * 1000.*0.5*pow(x+0.5*data.edge,1.5);
	  data.Env[0][index] = data.mpp[0] * 1000.*0.5*(x+0.5*data.edge)*(x+0.5*data.edge);//quadratic
	  //data.Env[0][index] = data.mpp[0] * 1000.*0.5*pow(x+0.5*data.edge,4.);//quartic
	  //cout << "data.Env[0][index] = " << data.Env[0][index] << endl;
	}
      }
    }
  }
  else if(data.Environments[s]==7){//fruit flies stairs square chamber
    //int lVecCounter = 0;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++){
      double x,y; int index;
      for(int j=0;j<n;j++){
	index = i*n+j;
	x = data.VecAt[index][0];
	y = data.VecAt[index][1];
	double stair = ceil((x+0.5*data.edge)/(data.edge/6.));
	if(y+0.5*data.edge < (6.-stair)*data.edge/6) data.Env[0][index] = 10000.*data.edge*data.mpp[0];
	else{
	  //lVecCounter++;
	  data.Env[0][index] = data.mpp[0] * 1000.*0.5*(x+0.5*data.edge)*(x+0.5*data.edge);//quadratic
	}
      }
    }
  }
	else if(data.Environments[s]==10 && data.FLAGS.ChemistryEnvironment){//chemistry//ToDo: multi-species
		LoadNuclei(data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			if((i % (int)((double)data.GridSize/20.))==0 || i>=data.GridSize-1){
				cout << "ChemistryEnv: " << i+1 << "/" << data.GridSize << "..." << endl;
			}
			data.Env[0][i] = 0.;
			for(int nuc=0;nuc<data.NumberOfNuclei;nuc++){
				vector<double> RVec = {{data.NucleiPositions[0][nuc]},{data.NucleiPositions[1][nuc]},{data.NucleiPositions[2][nuc]}};
				double distance = Norm(VecDiff(data.VecAt[i],RVec));
				if(data.NucleiTypes[nuc]==data.ValenceCharges[nuc]) data.Env[0][i] += AllElectronPP(data.NucleiTypes[nuc],distance,data);
				else{
					int pp;
					for(pp=0;pp<data.PPlist.size();pp++){
						if(data.PPlist[pp].atom==data.NucleiTypes[nuc] && data.PPlist[pp].valence==data.ValenceCharges[nuc]) break;
					}
					if(distance < data.PPlist[pp].rMax) data.Env[0][i] += spline1dcalc(data.PPlist[pp].PPspline, distance); else data.Env[0][i] -= data.gammaH*data.ValenceCharges[nuc]/distance;
				}
			}
		}
		cout << "Chemistry environment loaded; data.Env[0][data.CentreIndex] = " << data.Env[0][data.CentreIndex] << endl;
	}
  else if(data.Environments[s]==11 || data.Environments[s]==12 || data.Environments[s]==13){
    //USERINPUT
    double offset = 0.;
    //data.txtout << "offset(Env) = " << offset << "\\\\";//a uniform offset makes absolute contributions to the combined resource-environment energy, thus affects densities
    
    double Extent = 0.2*data.edge, Extent2 = Extent*Extent;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      vector<double> rVec = data.VecAt[i];
      double x = rVec[0], y = rVec[1], r2 = x*x+y*y, r = sqrt(r2);

      double Mountain = (smoothPOS(1.-pow(r/(0.5*data.edge),6.),0.5)/smoothPOS(1.,0.5)-smoothPOS(-7.,0.5))/(1.-smoothPOS(-7.,0.5));
      double Lake = smoothPOS(1.-pow(r2/Extent2,4.),0.5)/smoothPOS(1.,0.5);
      double Precipitation = 0.1+0.9*(0.5*data.edge-x)/data.edge;
      
      if(data.Environments[s]==11) data.Env[s][i] = data.mpp[0]*(Lake+1.-Precipitation);//"TygerWeb-Env" for Funges/Snale
      else if(data.Environments[s]==12) data.Env[s][i] = offset + data.mpp[1]*(Lake+Mountain);//"TygerWeb-Environment" for Trea
      else if(data.Environments[s]==13) data.Env[s][i] = data.mpp[2]*Lake;//"TygerWeb-Env" for Dear, Bore, Grazz, Tyger
    }
  }
  else if(data.Environments[s]==14){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double r = Norm(data.VecAt[i]);
      data.Env[s][i] = data.mpp[4]*data.RN(data.MTGEN) + data.mpp[0]*pow(r,data.mpp[1])+data.mpp[2]*pow(r,data.mpp[3]);
    }
  }
  else if(data.Environments[s]==15){//1D Morse potential
    for(int i=0;i<data.GridSize;i++){
      data.Env[s][i] = data.mpp[s]*(EXP(-0.5*(data.Lattice[i]+data.mpp[data.S+s]))-2.*EXP(-0.25*(data.Lattice[i]+data.mpp[data.S+s])));
    }
  }
  else if(data.Environments[s]==16){//anharmonic oscillator in 3D
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
		double x = data.mpp[0]*data.VecAt[i][0], y = data.mpp[1]*data.VecAt[i][1], z = data.mpp[2]*data.VecAt[i][2];
      data.Env[s][i] = 0.5*(x*x+y*y+z*z);
    }
  }  
  else if(data.Environments[s]==30){
    int col, shift = 2, NumEnvParam = TASK.NumEnvParam;
    data.EnvAv[s] = 0.;
    
    if(NumEnvParam>0){
      if(data.System==31 || data.System==33) col = 2;//ForestGeo,Left
      else if(data.System==32 || data.System==34) col = 3;//ForestGeo,Right
      else if(data.System==35 || data.System==36){ col = 2; shift = 1; }
    
      vector<double> altitude(data.GridSize), pH(data.GridSize), gradient;
      for(int i=0;i<data.GridSize;i++){
	altitude[i] = TASK.refData[i][col];
	pH[i] = TASK.refData[i][col+shift]; 
      }
      double Totalaltitude = accumulate(altitude.begin(),altitude.end(),0.0), TotalpH = accumulate(pH.begin(),pH.end(),0.0), Totalgradient;
      if(NumEnvParam>2){
	gradient.resize(data.GridSize);
	for(int i=0;i<data.GridSize;i++) data.TmpField1[i] = TASK.refData[i][col];   
	DelField(data.TmpField1, s, data.DelFieldMethod, data);
	for(int i=0;i<data.GridSize;i++) gradient[i] = sqrt(data.GradSquared[s][i]);
	Totalgradient = accumulate(gradient.begin(),gradient.end(),0.0);
	for(int i=0;i<data.GridSize;i++) gradient[i] = (gradient[i]-Totalgradient/((double)data.GridSize))/Totalgradient;
      }   
      for(int i=0;i<data.GridSize;i++){
	altitude[i] = (altitude[i]-Totalaltitude/((double)data.GridSize))/Totalaltitude;
	pH[i] = (pH[i]-TotalpH/((double)data.GridSize))/TotalpH; 
      }
      for(int i=0;i<data.GridSize;i++){
	data.Env[s][i] = data.mpp[NumEnvParam*s+0]*altitude[i];
	if(NumEnvParam>1) data.Env[s][i] += data.mpp[NumEnvParam*s+1]*pH[i];
	if(NumEnvParam>2) data.Env[s][i] += data.mpp[NumEnvParam*s+2]*gradient[i];
	data.EnvAv[s] += data.Env[s][i];
      }
      data.EnvAv[s] /= ((double)data.GridSize); //cout << "data.EnvAv[s]=" << data.EnvAv[s] << endl << vec_to_str(data.mpp) << endl; usleep(10*sec);
    }
    else for(int i=0;i<data.GridSize;i++) data.Env[s][i] = 0.;
    
    for(int i=0;i<data.GridSize;i++) data.Env[s][i] -= data.EnvAv[s];
  }
  else if(data.Environments[s]==61){//DynDFTe, time t=0->1
	  double t = data.mpp[0];
	  if(TASK.DynDFTe.mode==0) t = 0.;
	  for(int i=0;i<data.GridSize;i++){
		  double x = data.VecAt[i][0], y = data.VecAt[i][1];
		  data.Env[s][i] = -25.*(0.5*EXP(-25.*POW(x + 0.25 - 0.5*t,2) - 60.*POW(y - 0.2*t,2)) + 0.4*EXP(-50.*(POW(x + 0.25 - 0.5*t,2) + POW(y + 0.2*t,2))));
		  //data.Env[s][i] = -10.*EXP(-10.*POW(x+0.25-0.5*t,2));
		  //data.Env[s][i] = -20.*EXP(-20.*(POW(x+0.25-0.5*t,2)+y*y));
	  }
  }

  
  //add Wall if specified
  //if(ABS(data.Wall)>1.) AddWall(data.Env[s],data);
  if(ABS(data.Wall)>1.) Regularize({{6}}, data.Env[s], data.Wall, data);
  
}

void AddWall(vector<double> &Field, datastruct &data){//ToDo
  double min = Field[0], max = min; for(int i=1;i<data.GridSize;i++){ double test = Field[i]; if(test>max) max = test; else if(test<min) min = test; }
  double scale = (double)((int)data.Wall), WallHeight = scale*(max-min), rW = (data.Wall-scale)*0.5*data.edge; if(data.PrintSC==1) PRINT("WallHeight = " + to_string(WallHeight),data);
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    double r = Norm(data.VecAt[i]);
    if(r>rW) Field[i] += (r-rW)*WallHeight/(0.5*data.edge-rW);
  }  
}

void GetResources(datastruct &data){
	
  if(data.K>0){
	  StartTimer("GetResources",data);
	
    data.NonUniformResourceDensities.resize(data.K); data.AvailableResources.resize(data.K); data.ResMin.resize(data.K); data.ResMax.resize(data.K);
    int NumPreyResources = 0;
    for(int k=0;k<data.K;k++) if(data.Resources[k]<-1. && data.Resources[k]>-2.) NumPreyResources++;
    for(int k=0;k<data.K;k++){
      data.NonUniformResourceDensities[k].resize(data.GridSize);
      if(data.Resources[k]>0.){
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<data.GridSize;i++){//assign uniform resource here
	  data.NonUniformResourceDensities[k][i] = data.Resources[k];
	}
      }
      else if(data.Resources[k]<-2. && data.Resources[k]>-3.){//define nonuniform resource here
      
	if(data.System==1){//discrete data resource densities for Kenkel1991, see Ecology_DFT_25f_Kenkel.nb:
	  vector<vector<double>> MonoData(8); for(int i=0;i<8;i++) MonoData[i].resize(2);
	  double MaxS = 14.;
	  if(k==0) MonoData = {{0., 4.8}, {2., 4.6}, {4., 4.5}, {6., 4.4}, {8., 4.3}, {10., 3.9}, {12., 2.6}, {14., 2.2}};//MonoPoa
	  else if(k==1) MonoData = {{0., 7.5}, {2., 7.2}, {4., 6.4}, {6., 5.7}, {8., 5.2}, {10., 5.2}, {12., 4.6}, {14., 4.5}};//MonoHord
	  else if(k==2) MonoData = {{0., 9.0}, {2., 8.5}, {4., 7.8}, {6., 7.7}, {8., 7.2}, {10., 7.3}, {12., 7.2}, {14., 6.2}};//MonoPucc
	  //interpolate discrete resource data
	  spline1dinterpolant SPLINE;
	  vector<double> X(MonoData.size()), F(MonoData.size());
	  for(int j=0;j<MonoData.size();j++){
	    X[j] = MonoData[j][0];
	    F[j] = MonoData[j][1];
	  }
	  real_1d_array x,f;
	  x.setcontent(X.size(), &(X[0]));//salinity level
	  f.setcontent(F.size(), &(F[0]));//resource density (e.g. for resource k==0: MonoData[2] = {4., 4.5} means that there is 4.5 resource k=0 per unit area at salinity level 4.)
	  spline1dbuildlinear(x, f, SPLINE);
	  //assign resource densities and adjust SP_fitness accordingly!
	  GetSmoothRandomField(1, MaxS, data);
	  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	  for(int i=0;i<data.GridSize;i++){
	    //double Salinity = min(MaxS,MaxS*Norm(data.VecAt[i])/data.rW);//isotropic linearly increasing salinity, saturating beyond data.rW 
	    //double Salinity = min(MaxS,MaxS*abs(data.VecAt[i][0])/data.rW);//linearly increasing salinity along |x|-direction, saturating beyond data.rW
	    //double Salinity = 0.; if(data.VecAt[i][0]>0.) Salinity = min(MaxS,MaxS*data.VecAt[i][0]/data.rW);//linearly increasing salinity along positive x-direction, saturating beyond data.rW
	    //double Salinity = POS(MaxS*(1.-abs(data.VecAt[i][0])/data.rW));//linearly decreasing salinity along |x|-direction, zero beyond data.rW
	    //double Salinity = MaxS*EXP(-5.*abs(data.VecAt[i][0])/data.rW);//exponentially decreasing salinity along |x|-direction
	    //double Salinity = MaxS*EXP(-10.*pow(abs(data.VecAt[i][0])/data.rW,2.));//Gaussian decreasing salinity along |x|-direction
	    //double Salinity = min(MaxS,MaxS*0.4*abs(data.VecAt[i][0])/data.rW);//slowly linearly increasing salinity along |x|-direction
	    //double Salinity = min(MaxS,MaxS*1.5*abs(data.VecAt[i][0])/data.rW);//strongly linearly increasing salinity along |x|-direction
	    double Salinity = data.SmoothRandomField[i];
	    data.NonUniformResourceDensities[k][i] = spline1dcalc(SPLINE,Salinity);//salinity-dependent, for Poa & Hord & Pucc
	
// 	    double Rk; if(k==0) Rk = 4.8; else if(k==1) Rk = 7.5; else Rk = 9.0; data.NonUniformResourceDensities[k][i] = Rk;//salinity = 0
	    }
	  //if(ABS(data.Wall)>MP) Regularize({{2}}, data.NonUniformResourceDensities[k], data.rW, data);
	}
      
	if(data.System==2){//TigerWeb Resources
	  double Extent = 0.2*data.edge, Extent2 = Extent*Extent;
	  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	  for(int i=0;i<data.GridSize;i++){
	    vector<double> rVec = data.VecAt[i];
	    double x = rVec[0], y = rVec[1], r2 = x*x+y*y, r = sqrt(r2);

	    double Mountain = (smoothPOS(1.-pow(r/(0.5*data.edge),6.),0.5)/smoothPOS(1.,0.5)-smoothPOS(-7.,0.5))/(1.-smoothPOS(-7.,0.5));
	    double Lake = smoothPOS(1.-pow(r2/Extent2,4.),0.5)/smoothPOS(1.,0.5);
	    double VariableLakeShore = Lake/(1.+EXP(-100.*(r-0.5*Extent)/(0.5*data.edge)));
	    double Precipitation = 0.1+0.9*(0.5*data.edge-x)/data.edge;
	    double NorthSouthGradient = (0.5*data.edge-y)/data.edge;
	    double Sunshine = 0.2+0.8*(1.-Precipitation)*NorthSouthGradient;
	    double InverseLake = 1.-Lake;
	    //double WaterScale = 0.04, NutrientScale = 0.1, ResourceOffset = 0.;//0.01;//for Lennard-Jones resource term
	    double WaterScale = 20., LightScale = 30., NutrientScale = 40., ResourceOffset = 0.;//0.01;//for harmonic resource term

	    if(k==NumPreyResources) data.NonUniformResourceDensities[k][i] = WaterScale*(POS(InverseLake*Precipitation+VariableLakeShore)+ResourceOffset);//Water
	    else if(k==NumPreyResources+1) data.NonUniformResourceDensities[k][i] = LightScale*(POS(Sunshine)+ResourceOffset);//Sunshine
	    else if(k==NumPreyResources+2) data.NonUniformResourceDensities[k][i] = NutrientScale*(POS(InverseLake*Precipitation*Sunshine*(2.-Mountain)+VariableLakeShore*Sunshine)+ResourceOffset);//Nutrient
	  }
	}
      
      }
      else if(data.Resources[k]<-3. && data.Resources[k]>-4.){
	//load nonuniform resource from file
	
	//load nonuniform resource from TASK.refData
	int col;
	if(data.System==31 || data.System==33) col = 6+2*data.ResourceIndexList[k];//ForestGeo,Left
	else if(data.System==32 || data.System==34) col = 6+2*data.ResourceIndexList[k]+1;//ForestGeo,Right
	else if(data.System==35 || data.System==36) col = 4+data.ResourceIndexList[k];
	  
	//at this point: load absolute amounts of resources. no normalizing of resources (they need to be used in R if fitted in L)
	if((data.System==35 || data.System==36) && TASK.CoarseGrain && TASK.refDataType==2) for(int i=0;i<data.GridSize;i++) data.NonUniformResourceDensities[k][i] = TASK.FOMrefData[i][col];
	else for(int i=0;i<data.GridSize;i++) data.NonUniformResourceDensities[k][i] = TASK.refData[i][col];

	//(un)comment: (with) without normalization of resources, (normalization ok if used in L or R only, or in total area)
	if(data.System==35 || data.System==36){
	  double total = Integrate(data.ompThreads,data.method, data.DIM, data.NonUniformResourceDensities[k], data.frame);
	  for(int i=0;i<data.GridSize;i++) data.NonUniformResourceDensities[k][i] *= 1./total;//normalize absolute amount of resources in area to 1
	}
      }
    
      if(data.Wall>MP){
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<data.GridSize;i++){
	  if(Norm(data.VecAt[i])>data.rW) data.NonUniformResourceDensities[k][i] = 0.;
	}
      }
      
      if(data.Resources[k]>0.) data.AvailableResources[k] = data.Resources[k];//uniform resource
      else if(data.Resources[k]<-1. && data.Resources[k]>-2.) data.AvailableResources[k] = data.Abundances[k];//resource k is prey (species k)
      //else if(data.Resources[k]<-2. && data.Resources[k]>-3.) data.AvailableResources[k] = Integrate(data.ompThreads,data.method, data.DIM, data.NonUniformResourceDensities[k], data.frame);//nonuniform resource
      else if(data.Resources[k]<-2. && data.Resources[k]>-4.) data.AvailableResources[k] = Integrate(data.ompThreads,data.method, data.DIM, data.NonUniformResourceDensities[k], data.frame);//nonuniform resource
      
    }
  }
  EndTimer("GetResources",data);
}

void GetOptOccNum(taskstruct &task, datastruct &data){
  //ToDo: for more than one species
  
  int L = TASK.ex.settings[0], NumPairs = (L*L-L)/2;
  TASK.ex.L = L;
  TASK.ex.rho1p.resize(L); for(int i=0;i<L;i++) TASK.ex.rho1p[i].resize(L);
  TASK.ex.rho1pIm.resize(L); for(int i=0;i<L;i++) TASK.ex.rho1pIm[i].resize(L);
  TASK.ex.OccNum = InitializeOccNum(true,TASK.ex);
  TASK.ex.OccNumAngles = OccNumToAngles(TASK.ex.OccNum);
  TASK.ex.Phases.resize(L); fill(TASK.ex.Phases.begin(),TASK.ex.Phases.end(),0.);
  TASK.ex.Unitaries.resize(NumPairs); fill(TASK.ex.Unitaries.begin(),TASK.ex.Unitaries.end(),0.);
  TASK.ex.ExtraPhases.resize(2*NumPairs); fill(TASK.ex.ExtraPhases.begin(),TASK.ex.ExtraPhases.end(),0.);
  TASK.ex.UnitariesFlag.resize(NumPairs); fill(TASK.ex.UnitariesFlag.begin(),TASK.ex.UnitariesFlag.end(),0.);
  
  GetHint(TASK.ex);
  if(TASK.ex.settings[15]==1) exportHint("mpDPFT_1pExDFT_Hint.dat", TASK.ex);

  double SingleParticleEnergy = 0.;
  for(int l=0;l<L;l++) cout << "GetOptOccNum: level a=" << l+1 << " -> 1p-energy E_a=" << Energy1pEx(l,TASK.ex) << endl;

  OPTstruct opt;
  SetDefaultOPTparams(opt);
  opt.ex = TASK.ex;
  opt.ex.FreeIndices.clear(); opt.ex.FreeIndices.resize(0); for(int i=0;i<L;i++) opt.ex.FreeIndices.push_back(i);
  opt.ex.OccNum = InitializeOccNum(false,opt.ex);
  opt.reportQ = 1;
  opt.function = -1;
  opt.D = L;//dimension
  opt.epsf = opt.ex.RelAcc*opt.ex.UNITenergy;
  opt.threads = data.ompThreads;
  opt.ActiveOptimizer = TASK.ex.Optimizers[0];
  if(TASK.ex.settings[14]>0) opt.PCARBF = true;
  
  
  if(!TASK.ex.SHrandomSeedQ) opt.ex.RandOrthMat = orthogonal_matrix(TASK.ex.SHrandomSeedQ,opt.ex);
  
  opt.ex.MonitorMatrix.clear();
  opt.ex.MonitorMatrix.resize(0);
  
  string optstring, optstring2 = "\n";
  if(TASK.ex.Optimizers[0]<=-100){//manual check for one configuration, with dimensions, abundances, etc. consistent with mpDPFT.input:
    optstring = "ManualCheck(";
    if(TASK.ex.Optimizers[0]==-100) TASK.ex.OccNum = InitializeOccNum(false,TASK.ex);
    else if(TASK.ex.Optimizers[0]==-101) LoadOptOccNum(TASK.ex);
    
    //(un)comment
    TASK.ex.OccNum = {{1.983888140532272,0.9955880411084325,0.9833661939403735,0.02301504144707989,0.004711350885767907,0.9984336274354192,0.9249193459188072,0.07704475149592616,0.0000000000001029176743827520,0.002312009883971511,0.002591409477572970,0.0002392581935236615,0.001149182267437654,0.002741695667724975}};
    TASK.ex.Phases = {{4.9370754,4.2545767,5.3453602,4.4187688,2.9020654,1.5912851,4.4362368,2.624916,4.4355586,1.3700992,3.3597113,2.3277662,5.2364405,2.4247674}};
    NearestProperOccNum(TASK.ex.OccNum,TASK.ex);
    
    cout << "GetOptOccNum, input:" << endl;
    cout << "TASK.ex.OccNum(accumulated: " + to_string_with_precision(accumulate(TASK.ex.OccNum.begin(),TASK.ex.OccNum.end(),0.),20) + ") = " << vec_to_CommaSeparatedString_with_precision(TASK.ex.OccNum,20) << endl;
    cout << "TASK.ex.Phases = " << vec_to_CommaSeparatedString_with_precision(TASK.ex.Phases,20) << endl;
    cout << "GetOptOccNum, rho1p from GetRho1p:" << endl;
    GetRho1p(TASK.ex.rho1p,TASK.ex.OccNum,TASK.ex);
    //if(GenerateRho1pMixer(TASK.ex.rho1p, L, TASK.ex.OccNum, TASK.ex.Abundances[0])>0) cout << "GetRho1p: Error !!! Do something..." << endl;
//     cout << "rho1p = " << endl;
//     for(int l=0;l<L;l++) cout << vec_to_CommaSeparatedString_with_precision(TASK.ex.rho1p[l],20) << "," << endl;
//     MatrixToFile(TASK.ex.rho1p,"mpDPFT_1pExDFT_rho1p.dat",16);
    cout << "GetOptOccNum, energy:" << endl;
    for(int l=0;l<L;l++) SingleParticleEnergy += TASK.ex.OccNum[l]*Energy1pEx(l,TASK.ex);
    if(TASK.ex.settings[10]==0) TASK.ex.Eint = Eint_1pExDFT_CombinedMin(TASK.ex.Phases, TASK.ex.rho1p, TASK.ex.rho1pIm, TASK.ex);
    else if(TASK.ex.settings[10]==1) TASK.ex.Eint = MinimizeEint(TASK.ex.rho1p, TASK.ex);
    
    TASK.ex.Etot = SingleParticleEnergy + TASK.ex.Eint;
    cout << " Etot = E1p + Eint = " << SingleParticleEnergy << " + " << TASK.ex.Eint << " = "<< TASK.ex.Etot << endl; usleep(1*sec);   
  }
  else if(TASK.ex.Optimizers[0]==100){//Conjugate Gradient Descent over theta in occupation numbers n=1+cos(theta)
    optstring = "ConjugateGradientDescent(";
    opt.ex.settings[4] = 1;
    opt.SearchSpaceMin = -PI; opt.SearchSpaceMax = PI;
    opt.ex.settings[5] = 0;
    opt.ex.settings[6] = 1;
    opt.epsf = 0;
    SetDefaultCGDparams(opt);
    //ConjugateGradientDescent(-opt.ex.3, 1, 1, opt.ex.OccNum, task, data);    //Implementation 1
    CGD(opt);
    TASK.ex = opt.ex;
    TASK.ex.FuncEvals = opt.nb_eval;
    TASK.ex.Etot = opt.cgd.bestf;
    TASK.ex.OccNum = AnglesToOccNum(opt.cgd.bestx,TASK.ex);
    TASK.ex.CGDgr = opt.cgd.CGDgr;
    //TASK.ex.OccNum = AnglesToOccNum(opt.ex.OccNumAngles,TASK.ex);
    TASK.ConjGradParam = {{opt.cgd.epsf},{opt.cgd.epsg},{opt.cgd.epsx},{opt.cgd.diffstep}};    //Implementation 2
  }
  else if(TASK.ex.Optimizers[0]==101){//Particle Swarm Optimization over theta in occupation numbers n=1+cos(theta)
    optstring = "ParticleSwarmOptimization(";
    opt.ex.settings[4] = 1;
    //opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 2.*PI;
    if(opt.ex.settings[8]==1){
      opt.function = -2;
      opt.D += L;
    }
    if(opt.ex.settings[20]>0) opt.D += NumPairs;
	if(opt.ex.settings[20]>1) opt.D += 2*NumPairs;
    opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
    opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
    opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);    
    for(int i=0;i<L;i++){
      opt.SearchSpaceLowerVec[i] = 0.;
      opt.SearchSpaceUpperVec[i] = PI;
    }
    for(int i=L;i<opt.D;i++){
	  opt.SearchSpaceLowerVec[i] = 0.;
      opt.SearchSpaceUpperVec[i] = 2.*PI;
    }
	//GetFreeIndices(opt);
	//opt.ex.OccNum = InitializeOccNum(false,opt.ex);
	
    //begin adjust OPT parameters
    //opt.BreakBadRuns = 1;
    opt.epsf = 1.0e-12;
    //opt.UpdateSearchSpaceQ = 2;
    //end adjust OPT parameters

    SetDefaultPSOparams(opt);
    //begin adjust PSO parameters
    opt.pso.runs = 10000;
    opt.pso.loopMax = 100*opt.D;//2000*opt.D;
    opt.pso.increase = 100.;//increase of swarm size towards final run
    opt.pso.SwarmDecayRate = 1.7;
    opt.pso.InitialSwarmSize = /*100**/1*opt.D; InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
    //opt.pso.MaxLinks = 3;
    //opt.pso.eval_max_init = (int)1.0e+6*opt.D;
    //opt.pso.elitism = 0.1;
    //opt.pso.PostProcessQ = 1;
    opt.pso.VarianceCheck = min(opt.D,30);
    //opt.pso.reseed = 0.5;
    opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
    //opt.pso.alpha = 0.;
    opt.pso.TargetMinEncounters = opt.pso.runs;
    //end adjust PSO parameters
    PSO(opt);
    TASK.ex = opt.ex;
    TASK.ex.FuncEvals = opt.nb_eval;
    TASK.ex.Etot = opt.pso.bestf;
    for(int l=0;l<L;l++){
      TASK.ex.OccNum[l] = 1.+cos(opt.pso.bestx[l]);
      if(opt.ex.settings[8]==1 && opt.ex.settings[10]==1) TASK.ex.Phases[l] = opt.pso.bestx[L+l];
    }
    if(opt.ex.settings[20]>0) for(int l=0;l<NumPairs;l++) TASK.ex.Unitaries[l] = opt.pso.bestx[2*L+l];
	if(opt.ex.settings[20]>1) for(int l=0;l<NumPairs;l++){ TASK.ex.ExtraPhases[l] = opt.pso.bestx[2*L+NumPairs+l]; TASK.ex.ExtraPhases[NumPairs+l] = opt.pso.bestx[2*L+2*NumPairs+l]; }
  }
  else if(TASK.ex.Optimizers[0]==102){//Linearly-Constrained Optimization over occupation numbers n
    optstring = "LinearlyConstrainedOptimization(";
    if(opt.ex.settings[8]==1) opt.D += L;
    setupLCO(100000/*runs*/, opt.D, data.InternalAcc,opt);
    LCO(opt);
    reportLCO(opt);
    TASK.ex.FuncEvals = opt.lco.rep.FuncEvals;
    TASK.ex.Etot = opt.lco.rep.bestf;
    TASK.ex.OccNum = opt.lco.rep.OccNum;
    TASK.ex.Phases = opt.lco.rep.Phases;    
  }  
  else if(TASK.ex.Optimizers[0]==103){//Augmented Lagrangian Optimization over theta in occupation numbers n=1+cos(theta)
    optstring = "AugmentedLagrangianMethod(";
    opt.ex.settings[4] = 1;
    opt.SearchSpaceMin = -PI; opt.SearchSpaceMax = PI;
    opt.epsf *= 1.0e+0;
    SetDefaultAULparams(opt);
    AUL(opt);
    TASK.ex = opt.ex;
    TASK.ex.FuncEvals = opt.nb_eval;
    TASK.ex.Etot = opt.aul.bestf;
    TASK.ex.OccNum = AnglesToOccNum(opt.aul.bestx,opt.ex);
  }
  else if(TASK.ex.Optimizers[0]==104){//implemented only for opt.ex.settings[8]==1  
    optstring = "GeneticAlgorithmOptimization(";
    opt.ex.settings[4] = 1;   
    opt.function = -2;
    opt.D += L;
	opt.epsf = data.RelAcc;
	
    opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
    opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
    opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);    
    for(int i=0;i<L;i++){
      opt.SearchSpaceLowerVec[i] = 0.;
      opt.SearchSpaceUpperVec[i] = PI;
    }
    for(int i=L;i<opt.D;i++){
	  opt.SearchSpaceLowerVec[i] = 0.;
      opt.SearchSpaceUpperVec[i] = 2.*PI;
    }	
    
	//begin adjust parameters
	//opt.UpdateSearchSpaceQ = 2;
	opt.evalMax = 1.0e+10;//start low
	opt.gao.runs = 20*opt.D;
	opt.gao.popExponent = 0.3;
	//opt.gao.RandomSearch = true;//use GAO as random-search optimizer with exactly opt.evalMax function evaluations (no further inputs are necessary)
	SetDefaultGAOparams(opt);
	//opt.gao.HyperParamsDecayQ = false;
	//opt.gao.HyperParamsSchedule = 2;	
	//opt.PickRandomParamsQ = true;	
	opt.gao.ShrinkPopulationsQ = true;
	//end adjust parameters
	
    GAO(opt);    
    TASK.ex = opt.ex;
    TASK.ex.FuncEvals = opt.nb_eval;
    TASK.ex.Etot = opt.gao.bestf;
    for(int l=0;l<L;l++){
      TASK.ex.OccNum[l] = 1.+cos(opt.gao.bestx[l]);
      if(opt.ex.settings[8]==1 && opt.ex.settings[10]==1) TASK.ex.Phases[l] = opt.gao.bestx[L+l];
    }
  }
  else if(TASK.ex.Optimizers[0]==105){//coupled simulated annealing
    optstring = "coupledSimulatedAnnealing(";
    opt.ex.settings[4] = 1;
    opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 2.*PI;   
    if(opt.ex.settings[8]==1){
      opt.function = -2;
      opt.D += L;      
    }   
    SetDefaultCSAparams(opt);
    cSA(opt);
    TASK.ex = opt.ex;
    TASK.ex.FuncEvals = opt.nb_eval;
    TASK.ex.Etot = opt.csa.bestf;
    for(int l=0;l<L;l++){
      TASK.ex.OccNum[l] = 1.+cos(opt.csa.bestx[l]);
      if(opt.ex.settings[8]==1 && opt.ex.settings[10]==1) TASK.ex.Phases[l] = opt.csa.bestx[L+l];
    }    
  }
  
  TASKPRINT(opt.control.str(),task,0);
  TASKPRINT(optstring + to_string(TASK.ex.FuncEvals) + "): Emin = " + to_string_with_precision(TASK.ex.Etot,16) + "\n @ OccNum = " + vec_to_str(TASK.ex.OccNum) + "\n @ Phases = " + vec_to_str(TASK.ex.Phases) + "\n @ Unitaries = " + vec_to_str(TASK.ex.Unitaries) + optstring2 + "\n",task,1);
  
  //NearestProperOccNum(TASK.ex.OccNum,TASK.ex);
  
  TASKPRINT("BEGIN TASK.ex.control\n" + TASK.ex.control + "END TASK.ex.control\n",task,0);
    
  TASKPRINT("--> recompute/post-process:",task,1);
  string ppp = "";
  if(TASK.ex.settings[12]==1) ppp = " [with post-processing (CGD) of Phases]";
  else if(TASK.ex.settings[12]==2){
    opt.PostProcessQ = true;
    ppp = " [with post-processing (LCO) of Phases and OccNum]";//ToDo
    setupLCO(1/*run*/,opt.D,data.InternalAcc,opt);
    LCO(opt);
    reportLCO(opt);
    TASK.ex.FuncEvals = opt.lco.rep.FuncEvals;
    TASK.ex.Etot = opt.lco.rep.bestf;
    TASK.ex.OccNum = opt.lco.rep.OccNum;
    TASK.ex.Phases = opt.lco.rep.Phases;
    NearestProperOccNum(TASK.ex.OccNum,TASK.ex);
  }
  GetRho1p(TASK.ex.rho1p,TASK.ex.OccNum,TASK.ex);
  MatrixToFile(TASK.ex.rho1p,"mpDPFT_1pExDFT_rho1p_seed.dat",16);
  if(TASK.ex.settings[20]>0){
	TransformRho1p(TASK.ex.rho1p,TASK.ex.rho1pIm,TASK.ex.Unitaries,TASK.ex.ExtraPhases,TASK.ex);
	MatrixToFile(TASK.ex.rho1p,"mpDPFT_1pExDFT_rho1p_seed_transformed.dat",16);
  }
  SingleParticleEnergy = 0.; for(int l=0;l<L;l++) SingleParticleEnergy += TASK.ex.OccNum[l]*Energy1pEx(l,TASK.ex);
  if(TASK.ex.settings[12]==2) TASK.ex.Eint = TASK.ex.Etot - SingleParticleEnergy;
  else{
    if(opt.ex.settings[8]==0) TASK.ex.Eint = Eint_1pExDFT(TASK.ex.Phases,TASK.ex);//separate
    else if(opt.ex.settings[8]==1){//combined
      if( TASK.ex.settings[10]==1 && (TASK.ex.Optimizers[0]==-100 || TASK.ex.settings[12]==1) ){//optimize phases
		if(TASK.ex.settings[12]==1) TASK.ex.Optimizers[1] = 100;
		TASK.ex.Eint = MinimizeEint(TASK.ex.rho1p, TASK.ex);
      }
      else{
		TASK.ex.Eint = Eint_1pExDFT_CombinedMin(TASK.ex.Phases, TASK.ex.rho1p, TASK.ex.rho1pIm, TASK.ex);
	  }
    }	
    TASK.ex.Etot = SingleParticleEnergy + TASK.ex.Eint;
  }
  TASKPRINT("occupation numbers =  " + vec_to_CommaSeparatedString_with_precision(TASK.ex.OccNum,16),task,1);
  TASKPRINT("       accumulated =  " + to_string_with_precision(accumulate(TASK.ex.OccNum.begin(),TASK.ex.OccNum.end(),0.),20),task,1);
  TASKPRINT("            phases =  " + vec_to_CommaSeparatedString_with_precision(TASK.ex.Phases,16),task,1);
  if(TASK.ex.settings[20]>0){
	TASKPRINT("         unitaries =  " + vec_to_CommaSeparatedString_with_precision(TASK.ex.Unitaries,16),task,1);
	TASKPRINT(" #UnitariesFlagged =  " + to_string_with_precision(accumulate(TASK.ex.UnitariesFlag.begin(),TASK.ex.UnitariesFlag.end(),0.),16.),task,1);
	if(TASK.ex.settings[20]>1) TASKPRINT("      extra phases =  " + vec_to_CommaSeparatedString_with_precision(TASK.ex.ExtraPhases,16),task,1);
  }
  TASKPRINT("GetOptOccNum recomputed" + ppp + ": Etot = E1p + Eint = " + to_string_with_precision(SingleParticleEnergy,16) + " + " + to_string_with_precision(TASK.ex.Eint,16) + " = " + to_string_with_precision(TASK.ex.Etot,16),task,1);
    
  string testrho1p = "failed!";
  if(rho1pConsistencyCheck(TASK.ex.rho1p,TASK.ex.OccNum,TASK.ex)){
	  testrho1p = "OK!";
	  //if(TASK.ex.settings[20]==1) testUnitaries(task,data);//comment out when not needed!
  }
  TASKPRINT("rho1pConsistencyCheck " + testrho1p,task,1);
  MatrixToFile(opt.ex.MonitorMatrix,"mpDPFT_1pExDFT_MonitorMatrix_L=" + to_string(L) + ".dat",16);
  
}

void testUnitaries(taskstruct &task, datastruct &data){
	TASKPRINT("test effect of 2x2-unitaries:",task,1);
	TASKPRINT( "optimal Eint = \n" + to_string_with_precision(Eint_1pExDFT_CombinedMin(TASK.ex.Phases, TASK.ex.rho1p, TASK.ex.rho1pIm, TASK.ex),16),task,1);
	TASKPRINT( "Eint(with ZeroVec Phases) = \n" + to_string_with_precision(Eint_1pExDFT_CombinedMin(TASK.ex.ZeroVec, TASK.ex.rho1p, TASK.ex.rho1pIm, TASK.ex),16),task,1);
	int NumTests = 10000, L = TASK.ex.L, NumPairs = (L*L-L)/2; NumTests = max(20,NumTests);
	vector<double> res(NumTests), uni(NumPairs), extraphases(2*NumPairs);
	double eint = 1.0e+300,expon=0.;
	vector<vector<double>> rho1p(L), rho1pIm(L); for(int l=0;l<L;l++){ rho1p[l].resize(L); rho1pIm[l].resize(L); }
	GetRho1p(TASK.ex.rho1p,TASK.ex.OccNum,TASK.ex);
	for(int t=0;t<NumTests;t++){
		for(int u=0;u<NumPairs;u++) uni[u] = 2.*PI*data.RNpos(data.MTGEN);
		for(int l=0;l<2*NumPairs;l++) extraphases[l] = 2.*PI*data.RNpos(data.MTGEN);
		//for(int u=0;u<NumPairs;u++) uni[u] = TASK.ex.Unitaries[u] + 1.0e-1*data.RN(data.MTGEN);
		rho1p = TASK.ex.rho1p;
		TransformRho1p(rho1p,rho1pIm,uni,extraphases,TASK.ex);
		res[t] = Eint_1pExDFT_CombinedMin(TASK.ex.ZeroVec, rho1p, rho1pIm, TASK.ex);
		//res[t] = Eint_1pExDFT_CombinedMin(TASK.ex.Phases, uni, rho1p, rho1pIm, TASK.ex);
		if(t==(int)pow(10.,expon)-1){ cout << res[t] << " " << t << endl; expon += 1.; }
	}
	sort(res.begin(),res.end());
	TASKPRINT("Best (and worst) 10 of " + to_string(NumTests) + " (sorted) Eint from unitaries with random phases close to the optimum",task,1);
	for(int t=0;t<10;t++) TASKPRINT(to_string_with_precision(res[t],16),task,1);
	TASKPRINT("...",task,1);
	for(int t=0;t<10;t++) TASKPRINT(to_string_with_precision(res[NumTests-10+t],16),task,1);
}

void GetDensity(int s,datastruct &data){//omp-parallelized! --- don't call from within other omp loop!!! --- ToDo: omp nested
	StartTimer("GetDensity",data);

	if(TASK.Type==61 && TASK.DynDFTe.mode==1 && TASK.DynDFTe.InitializationPhase) data.Den[s] = TASK.DynDFTe.n0[s];
	else if(data.TaskType==100){//1p-exact DFT
		StartTimer("Get1pExDen",data);
		SetField(data,data.Den[s], data.DIM, data.EdgeLength, 0.);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			for(int l=0;l<TASK.ex.settings[0];l++){
				for(int k=0;k<TASK.ex.settings[0];k++){
					double mTerm = 0.; if(TASK.ex.settings[2]==23) mTerm = SphericalCoor(data.VecAt[i])[2]*(TASK.ex.QuantumNumbers[l][2]-TASK.ex.QuantumNumbers[k][2]);
					data.Den[s][i] += cos(TASK.ex.Phases[l]-TASK.ex.Phases[k]+mTerm)*TASK.ex.Orb[l][i]*TASK.ex.Abundances[0]*TASK.ex.rho1p[l][k]*TASK.ex.Orb[k][i];
				}
			}
		}
		EndTimer("Get1pExDen",data);
	}  
	else{
    
		int n = data.EdgeLength;
  
		//StartTiming(data); cout << "Den" << endl;
  
		if(data.DensityExpression==1 || data.DensityExpression==2) GetTFinspireddensity(s,data.DensityExpression,data);//TF-inspired densities
		else if(data.DensityExpression==3){//Coulomb-kernel density //Gauss-kernel density for 2D
			//set up [mu-V]
			//     SetComplexField(data,data.ComplexField, data.DIM, n, data.muVec[s], 0.);
			//     CopyRealFieldToComplex(data,data.TmpComplexField, data.V[s], 0, data.DIM, n);
			//     AddFactorComplexField(data,data.ComplexField, data.TmpComplexField, data.DIM, n, -1., 0.);
			//set up [mu-V]_T
			//StartTiming(data); cout << "set up [mu-V]_T" << endl;
			GetTFinspireddensity(s, 2, data);

			//MultiplyField(data,data.Den[s], data.DIM, n, data.tauVec[s]);
    
			SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.);
			CopyRealFieldToComplex(data,data.ComplexField, data.Den[s], 0, data.DIM, n);
			//CheckNAN("GetDensity", data.ComplexField, data.DIM, data);
			//EndTiming(data);
			//FFT [mu-V]_T
			//StartTiming(data); cout << "FFT" << endl;
			fftParallel(FFTW_FORWARD,data);
			//EndTiming(data);
			//multiply in k-space
			//StartTiming(data); cout << "FieldAssignment" << endl;
			//double t2 = data.tVec[s]*data.tVec[s];
	
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<n;i++){
				int index;
				double kx = data.k0+(double)i*data.Deltak, ky, factor;
				//double k2, prefactor = 1./(2.*PI*data.tauVec[s]*t2);
				for(int j=0;j<n;j++){
					index = i*n+j;
					ky = data.k0+(double)j*data.Deltak;
					//k2 = kx*kx + ky*ky;
	
					//factor = prefactor*EXP(0.5*t2*k2);
					//factor = EXP(0.5*t2*k2);
					factor = sqrt(kx*kx + ky*ky);
	
					data.ComplexField[index][0] *= factor;
					data.ComplexField[index][1] *= factor;
				}
			}
			//CheckNAN("GetDensity", data.ComplexField, data.DIM, data);
			//EndTiming(data);
			//inverse FFT
			//StartTiming(data); cout << "invFFT + output" << endl;
			fftParallel(FFTW_BACKWARD,data);
			CopyComplexFieldToReal(data,data.Den[s],data.ComplexField,0,data.DIM,n);
			MultiplyField(data,data.Den[s], data.DIM, n, 1./(2.*PI*data.tVec[s]));
    
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<n;i++) for(int j=0;j<n;j++) data.Den[s][i*n+j] *= Heaviside(data.Den[s][i*n+j]);
			//EndTiming(data);
		}
		else if(data.DensityExpression==4) GetSmoothdensity(s,data);
		else if(data.DensityExpression==5) GetnAiT(0,s,data);
		else if(data.DensityExpression==6) Getn3pFFT(s,data);
		else if(data.DensityExpression==7 || data.DensityExpression==8) GetTFdensity(s,data.DensityExpression,data);
		else if(data.DensityExpression==9) Getn3pT(s,data);
		else if(data.DensityExpression==10) Getn3p(s,data);
		else if(data.DensityExpression==11) Getn7(s,data);
		else if(data.FLAGS.SCO){
			if(data.DensityExpression<1000 && data.DensityExpression>1010) PRINT("GetDensity: Warning!!! DensityExpression incompatible",data);
			GetSCOdensity(s,data);
		}
  
		//cout << Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame) << endl;
		Regularize(data.regularize,data.Den[s],data.RegularizationThreshold,data);
		//cout << Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame) << endl;
	}
  
	EndTimer("GetDensity",data);
}

void getVint(int InteractionType, int ss, datastruct &data){
	//write into TmpField2 !!! (TmpField1 and TmpField3 are used by the subroutines here)
	//omp-parallelized - careful about performance when called from within omp loop!!!
	int EL = data.EdgeLength;
  
	if(InteractionType==0) SetField(data,data.TmpField2, data.DIM, EL, 0.);//no interaction
	else if(InteractionType<0) getLIBXC(0, ss, InteractionType, data.libxcPolarization, data);
	else if(InteractionType==1){//mutually repulsive contact interaction of strength data.beta
		SetField(data,data.TmpField2, data.DIM, EL, 0.);
		for(int s=0;s<data.S;s++){
			if(s!=ss) AddField(data,data.TmpField2, data.Den[s], data.DIM, EL); 
		}
		MultiplyField(data,data.TmpField2, data.DIM, EL, data.beta);
	}
	else if(InteractionType==2) ResourceIntegrand(0, ss, data);//central resource term
	else if((InteractionType>=3 && InteractionType<=7) || InteractionType==16 || InteractionType==18) CompetitionIntegrand(0, InteractionType, ss, data);//various point-like interactions
	else if(InteractionType==8){//default intraspecific Hartree-type repulsion, with prefactor data.gammaH
		data.HartreeType = 0;
		HartreeRepulsion(0, ss, data);
	}
	else if(InteractionType==9) RenormalizedContact(0, ss, data);//renormalized contact interaction for two species with S-wave scattering length data.beta
	else if(InteractionType==10){//repulsive contact interaction of strength data.beta (\beta\int(\d\vec r)(n(\vec r))^2) for one species
		data.TmpField2 = data.Den[0];
		MultiplyField(data,data.TmpField2, data.DIM, EL, 2.*data.beta);
	}
	else if(InteractionType==11) DiracExchange(0, ss,data);
	else if(InteractionType==12) GombasCorrelation(0, ss,data);
	else if(InteractionType==13) VWNCorrelation(0, ss, data);
	else if(InteractionType==14) DeltaEkin2D3Dcrossover(0, ss, data);
	else if(data.S==1 && data.DIM==3 && InteractionType==15) DipolarInteraction(0, ss, data);
	else if(InteractionType==16){ /*nothing to be done here, see above */ }
	else if(InteractionType==17) DipolarInteractionPos(0, ss, data);
	else if(InteractionType==18){ /*nothing to be done here, see above */ }
	else if(InteractionType==19 || InteractionType==20 || InteractionType==21){
		if(InteractionType==19) data.HartreeType = 1;//part 1 of parasitic intraspecific Hartree-type repulsion, with prefactor data.gamma
		else if(InteractionType==20) data.HartreeType = 2;//part 2 of parasitic intraspecific Hartree-type repulsion, with prefactor data.gamma
		else data.HartreeType = -1;//intraspecific Hartree repulsion for 2D, with prefactor data.gamma
		SetField(data,data.TmpField3, data.DIM, EL, 0.);
		for(int i=0;i<data.S;i++){
			if(i!=ss){
				data.FocalSpecies = ss;
				HartreeRepulsion(0, i, data);
				AddField(data,data.TmpField3, data.TmpField2, data.DIM, EL);
			}
		}
		data.TmpField2 = data.TmpField3;
	}
	else if(InteractionType==22){//full Coulomb interaction
		//at present only used in 1pEx-calculations
	}  
	else if(InteractionType==23){//harmonic interaction, single species, spin-1/2, unpolarized
		//ToDo: Just modify the oscillator frequency of the noninteracting system
	}
	else if(InteractionType==24) Quasi2Dcontact(0, ss, data);
	else if(InteractionType==61) DFTeAlignment(0, ss, data);
	else if(InteractionType==62) EkinDynDFTe(0, ss, data);
	else if(InteractionType==103){//harmonic interaction, single species, spin-1/2, unpolarized
		MomentumContactInteractionFiniteT(ss,data);
	}
	else if(data.FLAGS.SCO){//SelfConsistentOptimization method, for data.S==1 and only one InteractionType at a time
		
		//set up gradients (derivatives of objective functions)
		GetSCOgradients(data.Den[ss],data.TmpField2,data);//new gradients
		
		if(data.SCcount==0) data.lambda = 1.;
		
		if(data.PrintSC) PRINT("\n getVint: SCO learning rate = " + to_string(data.lambda),data);
		
		//multiply objective function gradients with learning rate lambda
		MultiplyField(data, data.TmpField2, data.DIM, EL, data.lambda);
		
		//subtract E_aux derivative
		if(data.DensityExpression>=1000 && data.DensityExpression<=1001){
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1 && data.GridSize>1000)
			for(int i=0;i<data.GridSize;i++) data.TmpField2[i] -= data.Den[ss][i];//TF-type
		}
		
	}
}

void GetSCOgradients(vector<double> &infield,vector<double> &outfield, datastruct &data){
	if(data.Interactions[0]==1000){//quadratic
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1 && data.GridSize>1000)
		for(int i=0;i<data.GridSize;i++){
			outfield[i] = infield[i]-1./((double)(i+1));
		}
	}
	else if(data.Interactions[0]==1001){//quartic
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1 && data.GridSize>1000)
		for(int i=0;i<data.GridSize;i++){
			outfield[i] = POW(infield[i]-1./((double)(i+1)),3);
		}
	}
	else if(data.Interactions[0]==1002){//inverted Gaussian
		double Gaussian = 0.;
		for(int i=0;i<data.GridSize;i++) Gaussian += POW(infield[i]-1./((double)(i+1)),2);
		Gaussian = EXP(-0.5*Gaussian);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1 && data.GridSize>1000)
		for(int i=0;i<data.GridSize;i++){
			outfield[i] = Gaussian * (infield[i]-1./((double)(i+1)));
		}
	}
	else if(data.Interactions[0]==1003){//Quadratic Programming Problem 20240502
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<OPTSCO.D;i++){
			outfield[i] = OPTSCO.AuxVec[i];
			for(int j=0;j<OPTSCO.D;j++) outfield[i] += 2.*OPTSCO.AuxMat[i][j]*infield[j];
		}
	}	
	else{//default: finite-difference gradient
		double delta = 1.0e-6;
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			vector<double> vars(data.Den[0]);
			vars[i] += delta;
			double f2 = GetSCOEnergy(vars,data);
			vars[i] -= 2.*delta;
			double f1 = GetSCOEnergy(vars,data);
			outfield[i] = (f2-f1)/(2.*delta);
		}
	}
	
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1 && data.GridSize>1000)
	for(int i=0;i<data.GridSize;i++){
		if(!(std::isfinite(outfield[i]))) outfield[i] = 1.0e+100;
	}
}

void MomentumContactInteractionFiniteT(int ss, datastruct &data){//Alex
  double lam = 2.0*PI/data.degeneracy/data.TVec[ss];
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for (int i = 0; i < data.GridSize; i++){
    vector<double> integrand(data.GridSize);
    double Den1 = data.Den[ss][i];//POS(data.Den[ss][i]);
    for (int j = 0; j < data.GridSize; j++){
      double Den2 = data.Den[1-ss][j];//POS(data.Den[1-ss][j]);
      if (ABS(Den1 - Den2) < 1e-10) {
	integrand[j] = 0.5*(1 - exp(-lam*Den2));
      }
      else {
	integrand[j] = (exp(lam*Den2) - 1)*(exp(lam*Den2) + exp(lam*Den1)*(lam*Den1 - lam*Den2 - 1))/((exp(lam*Den1) - exp(lam*Den2))*(exp(lam*Den1) - exp(lam*Den2)) );
      }
    }
    data.TmpField2[i] = data.beta*data.degeneracy/(4.*PI*PI)*Integrate(1,data.method, data.DIM, integrand, data.frame);
    //data.TmpField2[i] = data.beta * POS(data.Den[1-ss][i]);
  }
}

void ResourceIntegrand(int EnergyQ, int ss, datastruct &data){//resource terms for 2D
  //EnergyQ==0: Vint->TMPField2[ss]
  //EnergyQ==1: integrand of resource energy -> TMPField2[ss]
  
  StartTimer("ResourceIntegrand",data);
  
  SetField(data,data.TMPField2[ss], data.DIM, data.EdgeLength, 0.);
  SetField(data,data.tauVecMatrix[ss], data.DIM, data.EdgeLength, 0.);
  
  int jLimit = data.EdgeLength; if(data.System==35 || data.System==36) jLimit = (data.EdgeLength+1)/2;
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.EdgeLength;i++) for(int j=0;j<jLimit;j++){
  //for(int index=0;index<data.GridSize;index++){
    int index = i*data.EdgeLength+j;
    double Rk, TotalRk = 0.;
    vector<vector<double>> Contribution(2+data.S); for(int c=0;c<Contribution.size();c++) Contribution[c].resize(data.K);
    //Get resource densities & energy- and derivative contributions
    for(int k=0;k<data.K;k++){
      if(data.Resources[k]>=0.) Rk = data.Resources[k]/data.area;//uniform resource densities
      else if(data.Resources[k]<-1. && data.Resources[k]>-2.) Rk = data.Den[k][index];//resource density k is prey density (species k)
      //else if(data.Resources[k]<-2. && data.Resources[k]>-3.) Rk = data.NonUniformResourceDensities[k][index];//nonuniform resource densities
      else if(data.Resources[k]<-2. && data.Resources[k]>-4.) Rk = data.NonUniformResourceDensities[k][index];//nonuniform resource densities
      TotalRk += Rk;
      double nuksns = data.Consumption[k][ss]*data.Den[ss][index];
      double X = 0.;
      for(int s=0;s<data.S;s++){
	X += data.Consumption[k][s]*data.Den[s][index];
	//if(X!=X){ cout << "ResourceIntegrand X " << k << " " << s << " " << data.Consumption[k][s] << " " << data.Den[s][index] << " " << Rk << endl; }
      }
      Contribution[0][k] = 2.*data.Consumption[k][ss]*(X-nuksns-Rk);//derivative contribution for env,res & int,res
      Contribution[1][k] = nuksns*(X-nuksns)-2.*nuksns*Rk+Rk*Rk/((double)data.S);//energy contribution for env,res & int,res
      //if(Contribution[0][k]!=Contribution[0][k] || Contribution[1][k]!=Contribution[1][k]){ cout << "ResourceIntegrand C " << data.Consumption[k][ss] << " " << X << " " << nuksns << " " << Rk << endl; usleep(1*sec); }
      for(int s=0;s<data.S;s++){
	if(ABS(data.Consumption[k][s])>MP) Contribution[2+s][k] = Rk/data.Consumption[k][s];//support limit lambda_ks for Den_s
	else Contribution[2+s][k] = -1.;//species s does not require resource k
      }
    }
    data.LimitingResource[ss][index] = 0;
    if(TotalRk>MP){
      //Get limiting carrying capacities & LimitingResourceDistributions
      vector<double> weight(data.K), MinLambda(data.S);
      for(int s=0;s<data.S;s++){
	for(int k=0;k<data.K;k++) if(Contribution[2+s][k]>0.){ MinLambda[s] = Contribution[2+s][k]; if(s==ss) data.LimitingResource[ss][index] = (double)k; break; }//get one candidate MinLambda
	for(int k=0;k<data.K;k++) if(Contribution[2+s][k]>0. && Contribution[2+s][k]<MinLambda[s]){ MinLambda[s] = Contribution[2+s][k]; if(s==ss) data.LimitingResource[ss][index] = (double)k; }
      } 
      //Get weights and ss-contributions to (i) tauVecMatrix for 'dis,res' and (ii) energies and derivatives from env,res & int,res 
      for(int k=0;k<data.K;k++){
	weight[k] = 0.;
	for(int s=0;s<data.S;s++) if(data.Consumption[k][s]>0.){
	  if(MinLambda[s]>MP) weight[k] += EXP(data.sigma*(Contribution[2+s][k]/MinLambda[s]-1.))/data.AverageResourceSquared[k];
	  else if(Contribution[2+s][k]<MP) weight[k] += 1./data.AverageResourceSquared[k];
	  //if(weight[k]!=weight[k] || Contribution[EnergyQ][k]!=Contribution[EnergyQ][k]){ cout << "ResourceIntegrand " << ss << " " << index << " " << weight[k] << " " << Contribution[EnergyQ][k] << endl; usleep(1*sec); }
	}
	data.tauVecMatrix[ss][index] += data.zeta*weight[k]*data.Consumption[k][ss]*data.Consumption[k][ss];//ss-contribution to 'dis,res'
//       if(EnergyQ==0 && data.Resources[k]<-1. && data.Resources[k]>-2. && ss==k){//prey add-on for ss-contribution to derivative, don't use it
// 	Contribution[0][k] += 2.*data.Den[ss][index]/((double)data.S);
// 	for(int s=0;s<data.S;s++) Contribution[0][k] -= 2.*data.Consumption[ss][s]*data.Den[s][index];
//       }
	data.TMPField2[ss][index] += weight[k]*Contribution[EnergyQ][k];//ss-contribution to env,res & int,res
      }
      data.tauVecMatrix[ss][index] += 0.5*data.tauVec[ss];
    }
  }
  
  //if(data.System==35 || data.System==36) for(int i=0;i<data.EdgeLength;i++) for(int j=(data.EdgeLength+1)/2;j<data.EdgeLength;j++) data.tauVecMatrix[ss][i*data.EdgeLength+j] = 0.;
  
  //for(int index=0;index<data.GridSize;index++) if(data.TMPField2[ss][index]!=data.TMPField2[ss][index]){ cout << "ResourceIntegrand: " << ss << " " << index << " " << data.TMPField2[ss][index] << " " << data.tauVecMatrix[ss][index] << endl; usleep(1*sec); }
  
  if(ABS(data.zeta-1.)>MP) MultiplyField(data,data.TMPField2[ss], data.DIM, data.EdgeLength, data.zeta);
  
  EndTimer("ResourceIntegrand",data);
}

void CompetitionIntegrand(int EnergyQ, int InteractionType, int ss, datastruct &data){//for 2D
  //EnergyQ==0: Vint->TMPField2[ss]
  //EnergyQ==1: integrand of Competition energy -> TMPField2[ss]  
  
  //InteractionType==3: direct competition using constant (hard-coded) fitness
  //InteractionType==4: direct competition using position-dependent fitness (generated by SP_fitness)
  //InteractionType==5: direct competition using constant (hard-coded) data.Competition matrix
  //InteractionType==6:         amensalism using constant (hard-coded) data.Amensalism matrix
  //InteractionType==7:         amensalism using position-dependent fitness (generated by SP_fitness)
  //InteractionType==16:        repulsion, combined with mutualism using constant (hard-coded) data.RepMut matrix
  //InteractionType==18:        repulsion using position-dependent fitness (generated by SP_fitness)
  
  StartTimer("CompetitionIntegrand",data);
  
  int EL = data.EdgeLength;
  SetField(data,data.TMPField2[ss], data.DIM, EL, 0.);
  
  //set up competition coefficients
  vector<vector<double>> Competition; Competition.resize(data.S*data.S);//gamma_{ss'}(\vec r)
  int imax = 1, jmax = 1;
  if(InteractionType==4 || InteractionType==7 || InteractionType==18){
    for(int i=0;i<Competition.size();i++) Competition[i].resize(data.GridSize);
    imax = EL; jmax = EL;
  }
  else for(int i=0;i<Competition.size();i++) Competition[i].resize(1);
  int Cindex = 0;
  if(InteractionType==4  || InteractionType==7 || InteractionType==18){
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) ...cannot be used when SP_fitness is updated
    for(int i=0;i<imax;i++){
      for(int j=0;j<jmax;j++){
	int index = i*EL+j;
	Cindex = index; SP_fitness(index, data);//update fitness @ index
	for(int s1=0;s1<data.S;s1++){
	  for(int s2=0;s2<data.S;s2++){
	    if(s2!=s1) Competition[s1*data.S+s2][Cindex] = POS(data.fitness[s1]/data.fitness[s2]-1.);
	    else Competition[s1*data.S+s2][Cindex] = 0.;
	  }
	}
      }
    }
  }
  else{
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    //for(int index=0;index<data.GridSize;index++){
      //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int s1=0;s1<data.S;s1++){
	for(int s2=0;s2<data.S;s2++){
	  if(InteractionType==5) Competition[s1*data.S+s2][Cindex] = data.Competition[s1*data.S+s2];
	  else if(InteractionType==16) Competition[s1*data.S+s2][Cindex] = data.RepMut[s1*data.S+s2];
	  else if(InteractionType==6) Competition[s1*data.S+s2][Cindex] = data.Amensalism[s1*data.S+s2];
	  else if(s2!=s1) Competition[s1*data.S+s2][Cindex] = POS(data.fitness[s1]/data.fitness[s2]-1.);
	  else Competition[s1*data.S+s2][Cindex] = 0.;
	}
      }
    //}    
  }
  
  int jLimit = EL; if(data.System==35 || data.System==36) jLimit = (EL+1)/2;

  for(int s=0;s<data.S;s++){
    if(ABS(Competition[ss*data.S+s][0])>MP || ABS(Competition[s*data.S+ss][0])>MP){
      //for(int index=0;index<data.GridSize;index++){
      for(int i=0;i<EL;i++) for(int j=0;j<jLimit;j++){
	int index = i*EL+j;
	int cindex = 0;
	if(InteractionType==4  || InteractionType==7 || InteractionType==18) cindex = index;
	if(InteractionType==6 || InteractionType==7){
	  if(data.Den[ss][index]>MP){
	    if(EnergyQ) data.TMPField2[ss][index] += Competition[ss*data.S+s][cindex] * data.Den[s][index];
	    else data.TMPField2[ss][index] += Competition[s*data.S+ss][cindex];
	  }
	}
	else if(InteractionType==16 || InteractionType==18){
	  if(EnergyQ) data.TMPField2[ss][index] += data.Den[ss][index] * Competition[ss*data.S+s][cindex] * data.Den[s][index];
	  else data.TMPField2[ss][index] += data.Den[s][index]*(Competition[s*data.S+ss][cindex]+Competition[ss*data.S+s][cindex]);	  
	}
	else if(InteractionType<=5){
	  if(EnergyQ) data.TMPField2[ss][index] += data.Den[ss][index] * Competition[ss*data.S+s][cindex] * data.Den[s][index] * data.Den[s][index];
	  else data.TMPField2[ss][index] += data.Den[s][index]*(Competition[ss*data.S+s][cindex]*data.Den[s][index]+2.*Competition[s*data.S+ss][cindex]*data.Den[ss][index]);
	}
      }
    }
  }   
    
//   vector<double> res(data.GridSize,0.);
//   #pragma omp parallel for schedule(dynamic) reduction(vec_double_plus : res) if(data.ompThreads>1)
//   for(int s=0;s<data.S;s++){
//     if(ABS(Competition[ss*data.S+s][0])>MP || ABS(Competition[s*data.S+ss][0])>MP){
//       for(int index=0;index<data.GridSize;index++){
// 	int cindex = 0;
// 	if(InteractionType==4  || InteractionType==7 || InteractionType==18) cindex = index;
// 	if(InteractionType==6 || InteractionType==7){
// 	  if(data.Den[ss][index]>MP){
// 	    if(EnergyQ) res[index] += Competition[ss*data.S+s][cindex] * data.Den[s][index];
// 	    else res[index] += Competition[s*data.S+ss][cindex];
// 	  }
// 	}
// 	else if(InteractionType==16 || InteractionType==18){
// 	  if(EnergyQ) res[index] += data.Den[ss][index] * Competition[ss*data.S+s][cindex] * data.Den[s][index];
// 	  else res[index] += data.Den[s][index]*(Competition[s*data.S+ss][cindex]+Competition[ss*data.S+s][cindex]);	  
// 	}
// 	else if(InteractionType<=5){
// 	  if(EnergyQ) res[index] += data.Den[ss][index] * Competition[ss*data.S+s][cindex] * data.Den[s][index] * data.Den[s][index];
// 	  else res[index] += data.Den[s][index]*(Competition[ss*data.S+s][cindex]*data.Den[s][index]+2.*Competition[s*data.S+ss][cindex]*data.Den[ss][index]);
// 	}
//       }
//     }
//   }
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int index=0;index<data.GridSize;index++) data.TMPField2[ss][index] = res[index];
  
  double prefactor = data.gamma;
  if(InteractionType==6 || InteractionType==7) prefactor = data.alpha;
  if(InteractionType==16 || InteractionType==18) prefactor = data.beta;
  if(ABS(prefactor-1.)>MP) MultiplyField(data,data.TMPField2[ss], data.DIM, EL, prefactor);
  
  EndTimer("CompetitionIntegrand",data);
  
}

void HartreeRepulsion(int EnergyQ, int ss, datastruct &data){
	//intraspecific Hartree-type repulsion, with prefactor data.gammaH or data.gamma, for 2D & 3D
	//EnergyQ==0: Vint->TmpField2;
	//EnergyQ==1: integrand energy -> TmpField2
  
	if(data.DIM==2 && data.HartreeType==-1){
    
    	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    	for(int i=0;i<data.GridSize;i++){
      		data.ComplexField[i][0] = data.Den[ss][i];
      		data.ComplexField[i][1] = 0.;
    	}
    	fftParallel(FFTW_FORWARD, data);
    	double n0tilde = data.ComplexField[data.CentreIndex][0];
    
    	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    	for(int i=0;i<data.GridSize;i++){
      		double k = Norm(data.kVecAt[i]);
      		if(k>0.00001*data.Deltak){
				data.ComplexField[i][0] = (data.ComplexField[i][0]-n0tilde*EXP(-k*k))/k;
				data.ComplexField[i][1] /= k;
      		}
      		else{//put k==0-contribution of discrete FFT to zero in 2D
				data.ComplexField[i][0] = 0.;
				data.ComplexField[i][1] = 0.;
      		}
    	}

    	fftParallel(FFTW_BACKWARD, data);
    
        double prefac = n0tilde/(4.*SQRTPI), prefac2 = 2.*PI*(0.5*data.gamma);
    	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    	for(int i=0;i<data.GridSize;i++){
      		double z = Norm(data.VecAt[i]), expbesselI0; if(z*z/8.<1.0e+300) expbesselI0 = EXP(-z*z/8.)*gsl_sf_bessel_I0(z*z/8.); else expbesselI0 = 1./sqrt(2.*PI*z*z/8.);
      		data.ComplexField[i][0] += prefac*expbesselI0;
      		data.TmpField2[i] = prefac2*data.ComplexField[i][0];
      		if(EnergyQ) data.TmpField2[i] *= data.Den[data.FocalSpecies][i];
        }


  	}
  	else{
    	double W = 0.;
    	if(data.HartreeType==0) W = data.gammaH;
    	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    	for(int i=0;i<data.GridSize;i++){
      		data.ComplexField[i][0] = data.Den[ss][i];
      		if(data.HartreeType==1) data.ComplexField[i][0] *= data.Den[ss][i];
      		data.ComplexField[i][1] = 0.;
    	}
    	fftParallel(FFTW_FORWARD, data);
    	double f0tilde = data.ComplexField[data.CentreIndex][0];

    	if(data.DIM==2){
      		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      		for(int i=0;i<data.EdgeLength;i++){
				double qi = data.k0+(double)i*data.Deltak, qi2 = qi*qi;
				for(int j=0;j<data.EdgeLength;j++){
	  				double qj = data.k0+(double)j*data.Deltak, qj2 = qj*qj;
	  				int index = i*data.EdgeLength+j;
	  				double k = sqrt(qi2+qj2);
	  				if(k>0.00001*data.Deltak){
	    				data.ComplexField[index][0] = (data.ComplexField[index][0]-f0tilde*EXP(-k*k))/k;
	    				data.ComplexField[index][1] /= k;
	  				}
	  				else{//put k==0-contribution of discrete FFT to zero in 2D
	    				data.ComplexField[index][0] = 0.;
	    				data.ComplexField[index][1] = 0.;
	  				}
				}
      		}
      		fftParallel(FFTW_BACKWARD, data);
            if(data.HartreeType==0){
				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
				for(int i=0;i<data.GridSize;i++){
	  				double z = Norm(data.VecAt[i]), expbesselI0;
                	if(z*z/8.<1.0e+300) expbesselI0 = EXP(-z*z/8.)*gsl_sf_bessel_I0(z*z/8.);
                	else expbesselI0 = 1./sqrt(2.*PI*z*z/8.);
	  				data.ComplexField[i][0] += f0tilde*expbesselI0/(4.*SQRTPI);
	  				data.TmpField2[i] = 2.*PI*W * data.ComplexField[i][0];
	  				if(EnergyQ) data.TmpField2[i] *= 0.5*data.Den[ss][i];
				}
      		}
      		else if(data.HartreeType==1 || data.HartreeType==2){
				for(int i=0;i<data.GridSize;i++){//cannot pragma omp parallel due to SP_fitness
	  				double z = Norm(data.VecAt[i]), expbesselI0; if(z*z/8.<300.) expbesselI0 = EXP(-z*z/8.)*gsl_sf_bessel_I0(z*z/8.); else expbesselI0 = 1./sqrt(2.*PI*z*z/8.);
	  				data.ComplexField[i][0] += f0tilde*expbesselI0/(4.*SQRTPI);
	  				SP_fitness(i, data);
	  				if(data.HartreeType==1 || EnergyQ) W = data.gamma*POS(data.fitness[data.FocalSpecies]/data.fitness[ss]-1.);
	  				else W = 2.*data.Den[data.FocalSpecies][i] * data.gamma*POS(data.fitness[ss]/data.fitness[data.FocalSpecies]-1.);
	  				data.TmpField2[i] = 2.*PI*W * data.ComplexField[i][0];
				}
      		}
    	}
    	else if(data.DIM==3){
          	double kzero = 0.000001*data.Deltak*data.Deltak;
      		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      		for(int i=0;i<data.EdgeLength;i++){
				double qi = data.k0+(double)i*data.Deltak, qi2 = qi*qi;
				for(int j=0;j<data.EdgeLength;j++){
	  				double qj = data.k0+(double)j*data.Deltak, qj2 = qj*qj;
                    for(int l=0;l<data.EdgeLength;l++){
	    				double ql = data.k0+(double)l*data.Deltak, ql2 = ql*ql;
	    				int k = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+l;
	    				double qk2 = qi2+qj2+ql2;
	    				if(qk2>kzero){
	      					if(data.RegularizeCoulombSingularity) data.ComplexField[k][0] = (data.ComplexField[k][0]-f0tilde*EXP(-qk2))/qk2;
                            else data.ComplexField[k][0] /= qk2;
	      					data.ComplexField[k][1] /= qk2;
	    				}
	    				else{//qk2==0 contribution is f0tilde in 3D
                          	if(data.RegularizeCoulombSingularity) data.ComplexField[k][0] = f0tilde;
                            else data.ComplexField[k][0] = 0.;
	      					data.ComplexField[k][1] = 0.;
	    				}
	  				}
				}
      		}
      		fftParallel(FFTW_BACKWARD, data);
      		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      		for(int i=0;i<data.GridSize;i++){
              	if(data.RegularizeCoulombSingularity){
					double z = Norm(data.VecAt[i]), zf;
					if(z>1.0e-10) zf = PI*gsl_sf_erf(z/2.)/z;
                	else zf = SQRTPI;
					data.ComplexField[i][0] += f0tilde*zf/(4.*PI*PI);
                }
				data.TmpField2[i] = 4.*PI*data.gammaH * data.ComplexField[i][0];
				if(EnergyQ) data.TmpField2[i] *= 0.5*data.Den[ss][i];
      		}
			//for(int i=0;i<data.GridSize;i++){ if(data.TmpField2[i]<data.muVec[ss]) cout << "Hartree pot too neg" << endl; }
      		//cout << "Hartree: " << Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame) << endl;
    	}
 	}
}

void DipolarInteractionPos(int EnergyQ, int ss, datastruct &data){//3D, 1 species, dipolar interaction (in TF approximation), units==1, position space
  //EnergyQ==0: Vint->TmpField2;
  //EnergyQ==1: integrand energy -> TmpField2
  
  if(EnergyQ) SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);//ToDo
  else{
    double n0tilde = data.Abundances[ss], qk2test = 0.01*data.Deltak*data.Deltak, mu0mu2 = data.alpha; //0.00267;
    
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      data.ComplexField[i][0] = data.Den[ss][i];
      data.ComplexField[i][1] = 0.;
    }
    fftParallel(FFTW_FORWARD, data);
    
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double qk2 = Norm2(data.kVecAt[i]), ql2 = data.kVecAt[i][2]*data.kVecAt[i][2];//z-direction singled out
      if(qk2>qk2test){
	data.ComplexField[i][0] = (data.ComplexField[i][0]-n0tilde*EXP(-qk2))*((ql2/qk2)-1./3.+(8.*PI)/3.);
	data.ComplexField[i][1] *= (ql2/qk2)-1./3.+(8.*PI)/3.;
      }
      else{//q==0 contribution is zero
	data.ComplexField[i][0] = 0.;
	data.ComplexField[i][1] = 0.;
      }
    }   

    fftParallel(FFTW_BACKWARD, data);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){//singularity compensation
      double z = Norm(data.VecAt[i]), zf, sqrtPI = SQRTPI;
      if(z>1.0e-12) zf = (PI/(z*z*z))*(EXP(-z*z/4.)*sqrtPI*z*(4.+z*z)-4.*PI*gsl_sf_erf(z/2.))+PI*sqrtPI*EXP(-z*z/4.)*(-1./3.+8.*PI/3.);
      else zf = PI*sqrtPI*(8.*PI/3.);
      data.ComplexField[i][0] += (n0tilde/(8.*PI*PI*PI))*zf;
      data.TmpField2[i] = mu0mu2*data.ComplexField[i][0];
    }
  }
}

void DiracExchange(int EnergyQ, int ss, datastruct &data){//EnergyQ=0: Vint->TmpField2, EnergyQ=1: integrand for energy->TmpField2
  if(data.DIM>1){
    if(EnergyQ){
      double cXD; /*if(data.DIM==1) cXD = 0.; else */cXD = 2.*sqrt(4.*PI)/((data.DDIM-1.)*PI)*pow(0.5*pow(0.75*SQRTPI,data.DDIM-2.),1./data.DDIM);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
      for(int i=0;i<data.GridSize;i++) data.TmpField2[i] = -data.gammaH*cXD*data.DDIM/(1.+data.DDIM)*pow(POS(data.Den[ss][i]),1.+1./data.DDIM);
    }
    else{
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++) data.TmpField2[i] = -data.gammaH*(2.*pow(POS(data.Den[ss][i])*data.DDIM/PI,1./data.DDIM))/(data.DDIM-1.);
    }
  }
  else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
}

void GombasCorrelation(int EnergyQ, int ss, datastruct &data){//Vint->TmpField2
  if(EnergyQ){
    double a1 = 0.0357, a2 = 0.0562, b1 = 0.0311, b2 = 2.39, cG = 0.52917721067;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
    for(int i=0;i<data.GridSize;i++){
      double n13 = pow(POS(data.Den[ss][i]),1./3.);
      double dn = a2+cG*n13;
      double lnarg = 1.+b2*cG*n13;
      data.TmpField2[i] = -data.gammaH * data.Den[ss][i] * (a1*n13/dn+(b1/cG)* log(lnarg));
    }    
  }
  else{
    if(data.DIM==3 && data.Units==2){
      double a1 = 0.0357, a2 = 0.0562, b1 = 0.0311, b2 = 2.39, c = 0.52917721067;
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++){
	double n13 = pow(POS(data.Den[ss][i]),1./3.), dn = a2+c*n13, lnarg = 1.+b2*c*n13;
	data.TmpField2[i] = -data.gammaH * ( a1*n13/dn+ (b1/c)* log(lnarg) + (n13/3.) * ( a1*a2/(dn*dn) + b1*b2/lnarg ) );    
      }
    }
    else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
  }
}

void VWNCorrelation(int EnergyQ, int ss, datastruct &data){//Vint->TmpField2
  if(EnergyQ){
    double A = 0.846021, b = 13.072, c = 42.7198, Q = 0.0448998, x0 = -0.409286, BohrAngConv = 0.52917721067*0.52917721067*0.52917721067, Xx0 = 37.53713;
    for(int i=0;i<data.GridSize;i++){
      double density = POS(data.Den[ss][i]);
      if(density>1.0e-15){
	double x = pow(3./(4.*PI*BohrAngConv*density),1./6.);
	double Xx = x*x+b*x+c;
	double Yx = log(x*x/Xx);
	double Zx = log((x-x0)*(x-x0)/Xx);
	double ATAN = atan(Q/(2.*x+b));
	data.TmpField2[i] = density*A*(Yx+2.*b/Q*ATAN-b*x0/Xx0*(Zx+2.*(b+2.*x0)*ATAN/Q));
      }
      else data.TmpField2[i] = 0.;
    }    
  }
  else{
    if(data.DIM==3 && data.Units==2){
      double A = 0.846021, b = 13.072, c = 42.7198, Q = 0.0448998, x0 = -0.409286, BohrAngConv = 0.52917721067*0.52917721067*0.52917721067, Xx0 = 37.53713, pow1 = pow(2.,0.6666666666666666), pow2 = pow(2.,0.3333333333333333), pow3 = pow(6.,0.6666666666666666), pow4 = pow(PI,0.3333333333333333), pow5 = pow(3./PI,0.16666666666666666), pow6 = pow(PI,0.16666666666666666), pow7 = pow(3.,0.8333333333333334), pow8 = pow(6./PI,0.3333333333333333), pow9 = pow(PI,-0.3333333333333333), pow10 = pow(6.,0.3333333333333333), pow11 = pow(3.*PI,0.16666666666666666);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++){
	double density = POS(data.Den[ss][i]);
	if(density>MP){
	  double rs = pow(BohrAngConv*density,-1./6.), rs2 = rs*rs;
	  data.TmpField2[i] = A/(Q*Xx0)*(2.*b*(-(x0*(b + 2.*x0)) + Xx0)*atan(Q/(b + rs*pow1*pow5)) + Q*(Xx0*log(3./(3. + c*pow3*pow4/rs2 + b*pow2*pow7*pow6/rs)) - b*x0*log((pow(-2.*x0 + rs*pow1*pow5,2.)/(2.*c + b*rs*pow1*pow5 + rs2*pow8))/2.)) + (Q*(-(Xx0*(b*rs*pow2*pow7*pow6 + 2.*c*pow3*pow4)/(b*rs*pow2*pow7*pow6 + c*pow3*pow4 + 3.*rs2)) + 2.*b*rs*x0*pow9*(b*rs*pow10 + 2.*rs*x0*pow10 + 2.*c*pow1*pow11 + b*x0*pow1*pow11)/(-2.*x0 + rs*pow1*pow5)/(2.*c + b*rs*pow1*pow5 + rs2*pow8) - 2.*b*rs*pow1*(b*x0 - Xx0 + 2.*x0*x0)*pow5/(b*b + Q*Q + 2.*b*rs*pow1*pow5 + 2.*rs2*pow8)))/6.);
	}
      }
    }
    else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);  
  }
}

void getLIBXC(int EnergyQ, int ss, int neg_func_id, int polarization, datastruct &data){//write integrand to TmpField2
  	//from xc_funcs.h:
	// #define  XC_LDA_X               1  /* Exchange                                                   */
	// #define  XC_LDA_C_RPA           3  /* Random Phase Approximation                                 */
	// #define  XC_LDA_C_VWN_1        28  /* Vosko, Wilk, & Nussair (1)                                 */
	// #define  XC_LDA_C_VWN_2        29  /* Vosko, Wilk, & Nussair (2)                                 */
	// #define  XC_LDA_C_VWN_3        30  /* Vosko, Wilk, & Nussair (3)                                 */
	// #define  XC_LDA_C_VWN_4        31  /* Vosko, Wilk, & Nussair (4)                                 */
	// #define  XC_GGA_X_PBE         101  /* Perdew, Burke & Ernzerhof exchange                         */
	// #define  XC_GGA_C_PBE         130  /* Perdew, Burke & Ernzerhof correlation                      */
  
  	//int vmajor, vminor; xc_version(&vmajor, &vminor); PRINT("Libxc version:" + to_string(vmajor) + "." + to_string(vminor),data);
  	double Hartree = 27.211386245988;//in eV
  	double Bohr = 0.529177210903/*in Angstrom*/;
    double AngstromToBohr = pow(Bohr,3.);
    double AngstromToBohrForGradSquared = pow(Bohr,8.);
    double BohrToAngstrom = 1./AngstromToBohr;
  
  	xc_func_type func;
  	xc_func_init(&func, -neg_func_id, polarization);
  
  	if(data.InitializationInProgress){
    	//for(int i=0;func.info->refs[i]!=NULL;i++){ string xcinf = func.info->refs[i]->ref; data.txtout << xcinf << "\\\\"; cout << xcinf << endl; }
    	data.InitializationInProgress = 0;
  	}
  
    if(func.info->family != XC_FAMILY_LDA) DelField(data.Den[ss], ss, data.DelFieldMethod, data);

    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
    for(int i=0;i<data.GridSize;i++){
      	if(data.Den[ss][i]>0.){//convert Den and GradSquared from [Angstrom] (in mpDPFT) to [Bohr] for LIBXC_rho and LIBXC_sigma
			data.LIBXC_rho[i] = AngstromToBohr*data.Den[ss][i];
			if(func.info->family != XC_FAMILY_LDA) data.LIBXC_sigma[i] = AngstromToBohrForGradSquared*data.GradSquared[ss][i];
      	}
      	else{//consider only regions with positive densities
			data.LIBXC_rho[i] = 0.;
			data.LIBXC_sigma[i] = 0.;
     	}
    }

    switch(func.info->family){
      	case XC_FAMILY_LDA:
      		if(EnergyQ) xc_lda_exc(&func, data.GridSize, data.LIBXC_rho, data.LIBXC_exc);
      		else xc_lda_vxc(&func, data.GridSize, data.LIBXC_rho, data.LIBXC_vxc);
      	break;
      	case XC_FAMILY_GGA:
      	case XC_FAMILY_HYB_GGA:
      		if(EnergyQ) xc_gga_exc(&func, data.GridSize, data.LIBXC_rho, data.LIBXC_sigma, data.LIBXC_exc);
      		else xc_gga_vxc(&func, data.GridSize, data.LIBXC_rho, data.LIBXC_sigma, data.LIBXC_vxc, data.LIBXC_vsigma);
      	break;
    }
  
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
    for(int i=0;i<data.GridSize;i++){//consider only positive densities, convert to [Hartree] from LIBXC to eV for mpDPFT
      	if(EnergyQ) data.TmpField2[i] = POS(data.Den[ss][i])*Hartree*data.LIBXC_exc[i];
      	else data.TmpField2[i] = Hartree*data.LIBXC_vxc[i];
    }

  	xc_func_end(&func);
}

void RenormalizedContact(int EnergyQ, int ss, datastruct &data){
  //for 2D, two species, data.beta is the 2D scattering length
  int ssp = 0; if(ss==0) ssp = 1;
  bool WarningActive = false;
  double prefactor = 4.*PI*data.beta*data.beta;
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    //tweak densities:
    double ssDen = POS(data.Den[ss][i])+1.0e-16, sspDen = POS(data.Den[ssp][i])+1.0e-16, argss = prefactor*ssDen, argssp = prefactor*sspDen, etass = -2./log(argss), etassp = -2./log(argssp);
    if(!WarningActive && (argss>1.-MP || argssp>1.-MP)){ WarningActive = true; data.warningCount++; PRINT("RenormalizedContact: Warning!!! eta is about to leave repulsive branch.",data); } 
    if(EnergyQ) data.TmpField2[i] = PI*ssDen*sspDen*(PiotrBeta(0,etass) + PiotrBeta(0,etassp));
    else data.TmpField2[i] = PI*sspDen*(PiotrBeta(0,etass) + PiotrBeta(0,etassp) + 0.5*PiotrBeta(1,etass)*etass*etass);
  }
}

void DeltaEkin2D3Dcrossover(int EnergyQ, int ss, datastruct &data){
  //for DIM==2: remainder DeltaEkin for 2D-3D crossover, hbar=m=1, transversal oscillator frequency omega_z = data.alpha*omega
  //treated as part of the interaction energy (Ekin[2DTF] has been subtracted explicitly, i.e., Ekin[2DTF] is the proper DispersalEnergy)
  if(data.Units==1){
    if(data.PrintSC) PRINT("DeltaEkin2D3Dcrossover: maximal l used = " + to_string(floor(-0.5+sqrt(0.25+4.*PI*data.DenMax[ss]/data.alpha))),data);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double density = data.Den[ss][i], l = floor(-0.5+sqrt(0.25+4.*PI*density/data.alpha));
      if(l>MP){
	double V0 = 0.5*l*data.alpha, V1 = -PI*l/(l+1.);
	//double En0 = l*(l+1.)*data.alpha*((2.+7.*l)*data.alpha-8*PI*(1.+2.*l))/(48.*PI);
	double En0 = -data.alpha*data.alpha*l*(l+1.)*(l+2.)/(48.*PI);
	if(EnergyQ) data.TmpField2[i] = En0 + density*V0 + density*density*V1;
	else data.TmpField2[i] = V0 + density*2.*V1;
      }
      else data.TmpField2[i] = 0.;
    }
  }
  else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
}

void Load2D3Deta(datastruct &data){
  string FileName = "TabFunc_2D3D_eta.dat";//see Gen_MIT.nb
  ifstream infile;
  infile.open(FileName.c_str());
  string line;
  int ival, EntriesPerline = 5, count = 0;
  double val;
  
  vector<double> eta123(3);
  data.eta.clear(); data.eta.resize(0);
  while(getline(infile,line)){
    istringstream iss(line);
    if(line != ""){
      count++;
      iss.str(line);
      iss >> ival; iss >> ival;
      for(int i=2;i<EntriesPerline;i++){ iss >> val; eta123[i-2] = val; }
      data.eta.push_back(eta123);//31^2 triplets in row-major order: eta(l_1,l_2) = eta(31*l_1 + l_2)
      //cout << "Load2D3Deta: " << vec_to_str(eta123) << endl;
    }
  }
  infile.close();
  //cout << "Load2D3Deta: " << (int)(sqrt((double)count)+0.5) << "^2 triples {eta1,eta2,eta3}(l_+,l_-) in row-major order" << endl;
}

void Quasi2Dcontact(int EnergyQ, int ss, datastruct &data){
  //for DIM==2, S=2, perpendicular oscillator frequency omega_z = data.alpha*omega
  if(data.Units==1 && abs(data.beta)>MP){
    if(data.PrintSC) PRINT("Quasi2Dcontact: maximal l" + to_string(ss+1) + " used = " + to_string(floor(-0.5+sqrt(0.25+4.*PI*data.DenMax[ss]/data.alpha))),data);
    int s1 = ss/*focal species*/, s2 = abs(s1-1)/*other species*/;
    double fac1=data.alpha/(2.*PI), fac2 = fac1*fac1, G = data.beta*sqrt(data.alpha);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double den1 = data.Den[s1][i], den2 = data.Den[s2][i];
      int l1 = max(0,(int)floor(-0.5+sqrt(0.25+4.*PI*den1/data.alpha))), l2 = max(0,(int)floor(-0.5+sqrt(0.25+4.*PI*den2/data.alpha)));
      if(EnergyQ) data.TmpField2[i] = G*(den1*den2*data.eta[l1*31+l2][0] + fac1*(den1*data.eta[l1*31+l2][1]+den2*data.eta[l2*31+l1][1]) + fac2*data.eta[l1*31+l2][2]);
      else data.TmpField2[i] = G*(den2*data.eta[l1*31+l2][0] + fac1*data.eta[l1*31+l2][1]);
    }
  }
  else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
}

void DipolarInteraction(int EnergyQ, int ss, datastruct &data){//3D, 1 species, dipolar interaction (in TF approximation), momentum space
  if(data.Units==1){
    double mu0mu2 = data.alpha, prefactor = mu0mu2/(2.*pow(2.*PI,3.));
    vector<double> UnitVectorDipoleMoment(3); UnitVectorDipoleMoment[0] = 0.; UnitVectorDipoleMoment[1] = 0.; UnitVectorDipoleMoment[2] = 1.;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double test, rhop = data.Den[ss][i];
      vector<double> q = data.VecAt[i], k(3), pk(3);//wavevector corresponding to pVec
      for(int j=0;j<data.GridSize;j++){
	k = data.VecAt[j];//wavevector corresponding to kVec
	pk = VecSum(q,k);
	if(Norm(pk)<0.5*data.edge){
	  test = data.Den[ss][IndexOfVec(pk,data)] - rhop;
	  if(test>0.) data.TmpField1[j] = (UnitVectorDipoleMoment[0]*k[0] + UnitVectorDipoleMoment[1]*k[1] + UnitVectorDipoleMoment[2]*k[2])/Norm2(k) - 1./3.;
	  else data.TmpField1[j] = 0.;
	}
	else data.TmpField1[j] = 0.;
      }
      data.TmpField2[i] = prefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
    }
  }
  else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);  
}

void DFTeAlignment(int EnergyQ, int ss, datastruct &data){
	int DenFac = 2;
	//for DIM==2, S=1, prefactor data.alpha	
	if(data.DIM==2 && data.S==1){
		int n = data.EdgeLength;
		double AlignmentThreshold = MP;//1.0e-3//
		for(int i=0;i<data.GridSize;i++){
			double cosi = 1., CosPrefactor = 1.;
			double Ng = Norm(TASK.DynDFTe.g[ss][i]), Nv = Norm(TASK.DynDFTe.v[ss][i]);
			if(Ng>AlignmentThreshold && Nv>AlignmentThreshold) cosi = ScalarProduct(TASK.DynDFTe.g[ss][i],TASK.DynDFTe.v[ss][i])/(Ng*Nv);
			if(EnergyQ){
				double weight = 1.; if(DenFac==1) weight = data.Den[ss][i]; if(DenFac==2) weight = 0.5*data.Den[ss][i]*data.Den[ss][i];
				data.TmpField2[i] = weight*data.alpha*(1.-CosPrefactor*cosi);
			}
			else{
				double gDelSum = 0.;
				vector<int> indices = NeighborhoodIndices(i,false,data);
				for(int ni=0;ni<indices.size();ni++){
					int k = indices[ni];
					if(k!=i){//uki is zero for k==i
						int a = -1, b = -1, c = -1, d = -1;
						int l = (int)((double)k/(double)n+MP);//k=(l,m)
						int m = k-l*n;
						if(l>0) a = (l-1)*n+m;//moving left is allowed
						if(l<n-1) b = (l+1)*n+m;//moving right is allowed
						if(m>0) c = l*n+m-1;//moving down is allowed
						if(m<n-1) d = l*n+m+1;//moving up is allowed
						vector<double> uki(2);
						uki[0] = Kd(b,i)-Kd(a,i);
						uki[1] = Kd(d,i)-Kd(c,i);
						double kContrib = 0.;
						if(Norm2(uki)>MP){
							Nv = Norm(TASK.DynDFTe.v[ss][k]);
							Ng = Norm(TASK.DynDFTe.g[ss][k]);
							if(Nv>AlignmentThreshold && Ng>AlignmentThreshold){ 
								double cosk = ScalarProduct(TASK.DynDFTe.g[ss][k],TASK.DynDFTe.v[ss][k])/(Ng*Nv);
								double weight = 1.; if(DenFac==1) weight = data.Den[ss][k]; if(DenFac==2) weight = 0.5*data.Den[ss][k]*data.Den[ss][k];
								kContrib = weight * CosPrefactor * (1./Nv) * (ScalarProduct(TASK.DynDFTe.v[ss][k],uki)/Nv) * cosk - ScalarProduct(TASK.DynDFTe.g[ss][k],uki)/Ng;
							}
						}
						gDelSum += kContrib;
					}
				}
				double weight = 0.; if(DenFac==1) weight = 1.; if(DenFac==2) weight = data.Den[ss][i];
				data.TmpField2[i] = data.alpha*(weight*(1.-CosPrefactor*cosi)+gDelSum);
			}
		}
	}
	else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
}

void EkinDynDFTe(int EnergyQ, int ss, datastruct &data){
	//for DIM==2, S=1, prefactor data.beta
	int DenFac = 0;//1;//
	if(data.DIM==2 && data.S==1){
		int n = data.EdgeLength;
		for(int i=0;i<data.GridSize;i++){
			if(EnergyQ){
				double weight = 1.; if(DenFac==1) weight = data.Den[ss][i];
				data.TmpField2[i] = 0.5*data.beta*Norm2(TASK.DynDFTe.v[ss][i])*weight;
			}
			else{
				double Weight = 0.; if(DenFac==1) Weight = 0.5*Norm2(TASK.DynDFTe.v[ss][i]);
				data.TmpField2[i] = 0.;
				vector<int> indices = NeighborhoodIndices(i,false,data);
				for(int ni=0;ni<indices.size();ni++){
					int k = indices[ni];
					if(k!=i){//uki is zero for k==i
						double weight = 1.; if(DenFac==1) weight = data.Den[ss][k];
						int a = -1, b = -1, c = -1, d = -1;
						int l = (int)((double)k/(double)n+MP);//k=(l,m)
						int m = k-l*n;
						if(l>0) a = (l-1)*n+m;//moving left is allowed
						if(l<n-1) b = (l+1)*n+m;//moving right is allowed
						if(m>0) c = l*n+m-1;//moving down is allowed
						if(m<n-1) d = l*n+m+1;//moving up is allowed
						vector<double> uki(2);
						uki[0] = Kd(b,i)-Kd(a,i);
						uki[1] = Kd(d,i)-Kd(c,i);
						data.TmpField2[i] += ScalarProduct(TASK.DynDFTe.v[ss][k],uki)*weight;
					}
					else data.TmpField2[i] += Weight;
				}
				data.TmpField2[i] *= data.beta;
			}
		}
	}
	else SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 0.);
}

int IndexOfVec(vector<double> &Vec, datastruct &data){
  vector<int> coorIndices(data.DIM);
  int indexofposition = 0;
  for(int i=0;i<data.DIM;i++){
    coorIndices[i] = (int)(0.5+(data.EdgeLength-1)*((Vec[i]-data.frame[2*i])/data.edge));
    //indexofposition += coorIndices[i] * (int)(0.5+pow((double)EdgeLength,(double)(dim-1)));
    indexofposition += coorIndices[i] * (int)(0.5+pow((double)data.EdgeLength,(double)(data.DIM-1-i))); 
  }
  return indexofposition;
}

// S P E C I A L   P R E P A R A T I O N S

void SP_fitness(int i, datastruct &data){
  if(data.System==1){//Kenkel1991, from homoheneous pot:
  //outdated:
//   double x = data.mpp[0];//Salinity
//   data.fitness[0] = 0.461502*EXP(-0.107343*x);
//   data.fitness[1] = 0.431514+0.0269622*x-0.0012015*x*x;
//   data.fitness[2] = 0.140403+0.0147439*x;
//   for(int s=0;s<data.S;s++) for(int sp=0;sp<data.S;sp++) if(sp!=s) data.Amensalism[s*data.S+sp] = data.fitness[sp]/data.fitness[s]+0.5;
  
    //Kenkel1991, see Ecology_DFT_25g_Kenkel.nb:
    double MaxS=14., Salinity = 0.;//constant salinity
    //Salinity = min(MaxS,MaxS*Norm(data.VecAt[i])/data.rW);//isotropic linearly increasing salinity, saturating beyond data.rW 
    //Salinity = min(MaxS,MaxS*abs(data.VecAt[i][0])/data.rW);//linearly increasing salinity along |x|-direction, saturating beyond data.rW
    //if(data.VecAt[i][0]>0.) Salinity = min(MaxS,MaxS*data.VecAt[i][0]/data.rW);//linearly increasing salinity along positive x-direction, saturating beyond data.rW
    //Salinity = POS(MaxS*(1.-abs(data.VecAt[i][0])/data.rW));//linearly decreasing salinity along |x|-direction, zero beyond data.rW
    //Salinity = MaxS*EXP(-5.*abs(data.VecAt[i][0])/data.rW);//exponentially decreasing salinity along |x|-direction
    //Salinity = MaxS*EXP(-10.*pow(abs(data.VecAt[i][0])/data.rW,2.));//Gaussian decreasing salinity along |x|-direction
    //Salinity = min(MaxS,MaxS*0.4*abs(data.VecAt[i][0])/data.rW);//slowly linearly increasing salinity along |x|-direction
    //Salinity = min(MaxS,MaxS*1.5*abs(data.VecAt[i][0])/data.rW);//strongly linearly increasing salinity along |x|-direction
    Salinity = data.SmoothRandomField[i];
    data.fitness[0] = pow(0.670*EXP(-0.0030*(Salinity+15.)*(Salinity+15.)),data.kappa);
    data.fitness[1] = pow(0.525*EXP(-0.0026*(Salinity-5.0)*(Salinity-5.0)),data.kappa);
    data.fitness[2] = pow(0.530*EXP(-0.0022*(Salinity-22.)*(Salinity-22.)),data.kappa);
  }
}

void LoadNuclei(datastruct &data){
  string FileName = findMatchingFile("TabFunc_Nuclei",".dat");//see PseudoPotentials_05.nb
  cout << " Load Nuclei (charge & position[Angstrom]) from " << FileName << endl;
  ifstream infile;
  infile.open(FileName.c_str());
  string line;
  int ival, EntriesPerline = 5;
  double val;
  
  data.NucleiPositions.resize(3);//xyz positions
  while(getline(infile,line)){
    istringstream iss(line);
    if(line != ""){
      iss.str(line);
      iss >> ival; data.NucleiTypes.push_back(ival);
      iss >> ival; data.ValenceCharges.push_back(ival);
      for(int i=2;i<EntriesPerline;i++){ iss >> val; data.NucleiPositions[i-2].push_back(val); }
    }
  }
  infile.close();
  data.NumberOfNuclei = data.NucleiTypes.size();
  
  for(int i=0;i<data.NumberOfNuclei;i++){
    //PRINT(to_string(i+1) + ":    nucleus type = " + to_string(data.NucleiTypes[i]),data);
    //PRINT(to_string(i+1) + ":  valence charge = " + to_string(data.ValenceCharges[i]),data);
    for(int j=0;j<3;j++) data.NucleiPositions[j][i] *= data.stretchfactor;
    //PRINT(to_string(i+1) + ":        position = " + to_string(data.NucleiPositions[0][i]) + " " + to_string(data.NucleiPositions[1][i]) + " " + to_string(data.NucleiPositions[2][i]),data);
  }
  
  for(int atom=1;atom<118;atom++) for(int i=0;i<data.NumberOfNuclei;i++) if(data.NucleiTypes[i]==atom){ data.NucleiTypes.push_back(atom); break; }
  cout << "Nuclei loaded" << endl;
  data.NucleiEnergy = GetNucleiEnergy(data);
  if(data.TVec[0]<0.){
    for(int s=0;s<data.S;s++){ if(data.NucleiEnergy>MP) data.TVec[s] = data.AutomaticTratio*data.NucleiEnergy; else data.TVec[s] = ABS(data.TVec[s]); }
    data.txtout << "AutoTemperature = " << data.TVec[0] << "\\\\";
  }
  LoadPseudoPotentials(data);
}

void LoadPseudoPotentials(datastruct &data){//load high-resolution RadialTotalPseudoPotential {eV,A}; see PseudoPotentials_05.nb
	for(int i=0;i<data.NumberOfNuclei;i++){
		for(int atom=1;atom<118;atom++){
			if(data.NucleiTypes[i]==atom){
				for(int valence=1;valence<=atom;valence++){
					if(data.ValenceCharges[i]==valence && data.ValenceCharges[i]<data.NucleiTypes[i]){
						bool ppLoaded = false;
						for(int pp=0;pp<data.PPlist.size();pp++){
							if(data.PPlist[pp].atom==atom && data.PPlist[pp].valence==valence){ ppLoaded = true; break; }
						}
						if(!ppLoaded){
							pseudopotstruct PPstruct;
							string FileName = "TabFunc_PP_" + to_string(atom) + "_" + to_string(valence) + ".dat";
							cout << "Load " << FileName << endl;
							ifstream infile;
							string line;
							infile.open(FileName.c_str());
							double tmp;
							int rQ = 1;
							vector<double> Radialr,Radialf;
							while(getline(infile,line)){
								istringstream LINE(line);
								if(line != ""){
									while(LINE>>tmp){
										if(rQ){ Radialr.push_back(tmp); rQ=0; PPstruct.rMax = tmp; }
										else{ Radialf.push_back(tmp); rQ=1; }
									}
								}
							}
							infile.close();
							real_1d_array radialr,radialf;
							spline1dinterpolant PPspline;
							radialr.setcontent(Radialr.size(), &(Radialr[0]));
							radialf.setcontent(Radialf.size(), &(Radialf[0]));
							spline1dbuildcubic(radialr, radialf, PPspline);
							PPstruct.atom = atom;
							PPstruct.valence = valence;
							PPstruct.PPspline = PPspline;
							data.PPlist.push_back(PPstruct);
							PRINT(" PseudoPotential " + FileName + " loaded",data);
						}
					}
				}
			}
		}
	}
	cout << "All PseudoPotentials loaded" << endl;
}

double AllElectronPP(int ZZ, double r, datastruct &data){
	double W = 14.3996483256249, Z = (double)ZZ;
  	if(data.AllElectronPP==0){//MinimalCoulombPP
   		if(r>MP) return -Z*W/r;
        else return -2.*Z*W/data.Deltax;
  	}
  	else if(data.AllElectronPP==1){//Paul Ayers' hydrogen-like pseudopotential in {eV,Angstrom}
      	double aep = data.AllElectronParam, ERF = aep*r;
      	if(ERF>10.) ERF=1.; else if(ERF<-10.) ERF=-1.; else ERF = gsl_sf_erf(ERF);
      	if(r>MP) return Z*W*(-(0.923 + 1.568*aep)*EXP(-pow((0.241 + 1.405*aep)*r,2.))-ERF/r);
      	else return -Z*W*((0.923 + 1.568*aep) + 2.*aep/SQRTPI);
    }
    else if(data.AllElectronPP==2){//Gygi-Lehtola PP
      r *= Z;
      double res = 0.;
      //double Gygi = 1.; double b = 3.6442293860*0.1;
      //double Gygi = 2.; double b = 1.9653418941*0.1;
      //double Gygi = 4.; double b = 1.0200558632*0.1;
      //double Gygi = 8.; double b = 5.1947028410*0.01;
      double Gygi = 12.; double b = 3.4841536898*0.01;

      double a = Gygi;
      double ERF = a*r;
      if(ERF>10.) ERF=1.; else if(ERF<-10.) ERF=-1.; else ERF = gsl_sf_erf(ERF);
      if(r>MP){
        	double a2 = a*a, r2 = r*r;
        	res =  0.5 - (2.*a2*POW(-1.+a*b*SQRTPI,2)*r2)*EXP(-2.*a2*r2)/PI - ERF/r - 0.5*ERF*ERF + ( a*(-4.+a*(2.*a*r2+b*SQRTPI*(3.-2.*a2*r2)) + 2.*(-1.+a*b*SQRTPI)*r*ERF) ) * EXP(-a2*r2)/SQRTPI;
      }
      else res = 0.5+3.*a*a*b-6.*a/SQRTPI;
      return W*Z*Z*res;
    }
    else return 0.;
}

void GetGuidingVectorField(int s, datastruct &data, taskstruct &task){
	double mixing = 1.;
	//(moving average of) current (raw) velocity vector field
	int n = data.EdgeLength;
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int GridIndex=0;GridIndex<data.GridSize;GridIndex++){
		for(int d=0;d<data.DIM;d++)	task.DynDFTe.v[s][GridIndex][d] *= (1.-mixing);
		int i = (int)((double)GridIndex/(double)n+MP);
		int j = GridIndex-i*n;
		int down = 0, up = 0, left = 0, right = 0;
		if(i>0) left = 1;//moving left is allowed
		if(i<n-1) right = 1;//moving right is allowed
		if(j>0) down = 1;//moving down is allowed
		if(j<n-1) up = 1;//moving up is allowed
		for(int h=-left;h<=right;h++) if(h!=0) task.DynDFTe.v[s][GridIndex][0] += mixing*(double)h*(data.Den[s][(i+h)*n+j]-data.OldDen[s][(i+h)*n+j]);
		for(int v=-down;v<=up;v++) if(v!=0) task.DynDFTe.v[s][GridIndex][1] += mixing*(double)v*(data.Den[s][i*n+(j+v)]-data.OldDen[s][i*n+(j+v)]);
// 		if(data.SCcount==20 && GridIndex==115){
// 			cout << "GridIndex=" << GridIndex << "=(" << i << "," << j << ") r=" << vec_to_str(data.VecAt[GridIndex]) << " (->line(file)=" << GridIndex+1 << "): v=" << vec_to_str(task.DynDFTe.v[s][GridIndex]) << " Delta nb = " << (data.Den[s][(i+1)*n+j]-data.OldDen[s][(i+1)*n+j]) << " Delta na = " << (data.Den[s][(i-1)*n+j]-data.OldDen[s][(i-1)*n+j]) << endl;
// 			usleep(10*sec);
// 		}
	}
	//currently active guiding vector field
	double focalweight = 1.;//for intertia of focal pixel
	int multiAveraging = 1;//0;//
	for(int mA=0;mA<=multiAveraging;mA++){
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int GridIndex=0;GridIndex<data.GridSize;GridIndex++){
			vector<int> indices = NeighborhoodIndices(GridIndex,true,data);
			for(int d=0;d<data.DIM;d++) task.DynDFTe.g[s][GridIndex][d] = 0.;
			for(int j=0;j<indices.size();j++){
				int x = indices[j];
				if(x==GridIndex) for(int d=0;d<data.DIM;d++) task.DynDFTe.g[s][GridIndex][d] += focalweight*task.DynDFTe.v[s][x][d]*data.Den[s][x];
				else for(int d=0;d<data.DIM;d++) task.DynDFTe.g[s][GridIndex][d] += task.DynDFTe.v[s][x][d]*data.Den[s][x];
			}
			for(int d=0;d<data.DIM;d++)	if(ABS(task.DynDFTe.g[s][GridIndex][d])<MP) task.DynDFTe.g[s][GridIndex][d] = 0.;
		}
		if(multiAveraging>0 && mA==0) task.DynDFTe.v[s] = task.DynDFTe.g[s];
	}
}

vector<int> NeighborhoodIndices(int GridIndex, bool NNN, datastruct &data){//indices of nearest(& next-to-nearest, if NNN==true) 2Dneighborhood
	vector<int> NI(0);
	int n = data.EdgeLength;
	int i = (int)((double)GridIndex/(double)n+MP);
    int j = GridIndex-i*n;
	int down = 0, up = 0, left = 0, right = 0;
	if(i>0) left = 1;//moving left is allowed
	if(i<n-1) right = 1;//moving right is allowed
	if(j>0) down = 1;//moving down is allowed
	if(j<n-1) up = 1;//moving up is allowed
	for(int h=-left;h<=right;h++){
		for(int v=-down;v<=up;v++){
			if((i+h)*n+(j+v)<0 || (i+h)*n+(j+v)>=data.GridSize) PRINT("NeighborhoodIndices: Error !!! GridIndex=" + to_string(GridIndex) + " i=" + to_string(i) + " j=" + to_string(j) + " NeighborIndex=" + to_string((i+h)*n+(j+v)),data);
			if(NNN || (!NNN && abs(h)+abs(v)<2)) NI.push_back((i+h)*n+(j+v));
		}
	}
	return NI;
}


// S E C O N D A R Y    R O U T I N E S

void GetDensities(taskstruct &task, datastruct &data){
	if(data.TaskType==100 && TASK.ex.System>=100) GetOptOccNum(task, data);
	for(int s=0;s<data.S;s++){
		GetDensity(s,data);
		data.TmpAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
	}
	StartTimer("EnforceConstraints",data);
	EnforceConstraints(data);
	EndTimer("EnforceConstraints",data);
	if(data.TaskType==61 && TASK.DynDFTe.mode==1) TASK.DynDFTe.InitializationPhase = false;//initialization done if densities computed once in DynDFTe.mode==1
}

void GetTFinspireddensity(int s, int DenExpr, datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<data.GridSize;i++){
		double test = data.muVec[s]-data.V[s][i], prefactor;
		if(data.System==1 || data.System==2 || data.FLAGS.ForestGeo){ if(data.tauVecMatrix[s][i]>MP) prefactor = 1./(2.*data.tauVecMatrix[s][i]); else prefactor = 0.; } else prefactor = 1./data.tauVec[s];
		test *= prefactor;
		if(abs(prefactor)<1.0e-100){ prefactor = 0.; test = 0.; }
		else if(prefactor!=0.){
			if(!(std::isfinite(test)) || !(std::isfinite(prefactor))){
				cout << "GetTFinspireddensity: " << s << " " << i << " " << prefactor << " " << test << " " << data.muVec[s] << " " << data.V[s][i] << endl;
				test = 0.;
			}
		}
		else test = 0.;
		if(DenExpr==1){
			if(test>0. && std::isfinite(test)){
				if(data.DIM==1) data.Den[s][i] = sqrt(prefactor*test);
				else if(data.DIM==2) data.Den[s][i] = test;
				else if(data.DIM==3) data.Den[s][i] = test*sqrt(test)/sqrt(prefactor);
			}
			else data.Den[s][i] = 0.;
		}
		else{
			double test2 = test/data.TVec[s];
			if(ABS(test2)<40. && test2>-40.) data.Den[s][i] = prefactor*data.TVec[s]*log(1.+EXP(test2));
			else if(test2<=-40. || test2!=test2) data.Den[s][i] = 0.;
			else data.Den[s][i] = prefactor*test;
		} 
	}
}

void GetSmoothdensity(int s, datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  //for 2D
  DelField(data.V[s], s, 1, data);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.EdgeLength;i++){
    int index; double smoothening;
    for(int j=0;j<data.EdgeLength;j++){
      index = i*data.EdgeLength+j;
      smoothening = 0.01*data.edge*sqrt(data.Dx[s][index]*data.Dx[s][index]+data.Dy[s][index]*data.Dy[s][index]);
      data.Den[s][index] = (data.TVec[s]/data.tauVec[s])*log(1.+EXP((data.muVec[s]-(data.V[s][index]+smoothening))/data.TVec[s]));
    }
  }  
}

double getDenEnAt(int EnergyQ, int s, int i, datastruct &data){
    //USERINPUT
    double tweak = 1.;//default=1.;
      
    if(data.DensityExpression==5){
      int maxIter = (int)(data.MppVec[s][1]+0.5);//maximum number of bisections //18;
      double alpha; if(data.Units==1) alpha = 1.; else if(data.Units==2) alpha = 1./0.1312342130759153;//alpha=hbar^2/(me*Angstrom^2*eV);
      double res = 0., tau = 1./data.TVec[s], beta0 = -data.degeneracy/pow(2.*PI,1.5), beta1 = pow(alpha*tau,-1.5);
      
      if(data.DIM==2 || data.DIM==3){//ToDo: 1D
	for(int a=0;a<data.nAiTevals[s][i].size();a++) data.nAiTevals[s][i][a] = 0;
	double U = data.V[s][i]-data.muVec[s], ar = 0.5*pow(alpha*tweak*data.GradSquared[s][i],1./3.);
	double beta2 = -data.Laplacian[s][i]/(12.*sqrt(alpha*tau));
	if(ar<ABS(data.MppVec[s][2])){//when to treat Nabla(V) as zero //1.0e-6;  
	  double arg = -tau*U;
	  if(data.DIM==1) res = 0.;
	  else if(data.DIM==2){
	    double zeta = EXP(-arg), LOGterm;
	    if(arg<-100.) LOGterm = 1./zeta; else if(arg>100.) LOGterm = arg; else LOGterm = log(1.+1./zeta);	    
	    res = data.degeneracy/(2.*PI)*( LOGterm/tau - tau*tweak*data.Laplacian[s][i]/12.*zeta/((1.+zeta)*(1.+zeta)) );
	  }
	  else if(data.DIM==3){
	    if(EnergyQ) res = -beta0*data.TVec[s]*(beta1*polylog("3half",arg,data)+beta2*polylog("m1half",arg,data));
	    else res = beta0*(beta1*polylog("3half",arg,data)+beta2*polylog("m1half",arg,data));
	  }
	}
	else{
	  if(data.DIM==1) res = 0.;
	  else if(data.DIM==2){
	    double RTPNA = data.InternalAcc;//min(1.0e-10,data.RelAcc);
	    //double prefactor = data.degeneracy/(2.*PI); double prefactor2 = alpha*tau*tau/12.;
	    double prefactor = data.degeneracy/(2.*PI); double prefactor2 = tau*tweak*data.Laplacian[s][i]/12.;
	    double x = -1., upperx = 100., prevres = 0.;
	    res = AdaptiveBoole_nAiT(EnergyQ, s, maxIter, x, upperx, 0., 0, RTPNA, data.AverageDensity[s], prefactor,U,ar,i,prefactor2,data);
	    bool success = false;
	    while(!success){
	      prevres = res; upperx = x; x *= 2.; res += AdaptiveBoole_nAiT(EnergyQ, s, maxIter, x, upperx, 0., 0, RTPNA, data.AverageDensity[s], prefactor,U,ar,i,prefactor2,data);
	      if(x<-64. && ABS(prevres-res)<RTPNA) success = true;
	      else if(x<-1.0e+8){ success = true; /*if(data.PrintSC==1)*/ PRINT("GetnAiT: Warning !!! Lower integration limit not found at r = (" + vec_to_str(data.VecAt[i]) + "): prevres = " + to_string_with_precision(prevres,16) + " res = " + to_string_with_precision(res,16),data); }
	    }
	        
/*	    double test = 1., x = -1., arg = -tau*(U-x*ar);
	    while(ABS(test)>MP){//find lower integration limit:
	      x *= 2.; arg = -tau*(U-x*ar); data.nAiTevals[s][i][0]++;
	      test = -data.degeneracy*EXP(arg)*(data.Laplacian[s][i]-12.)/(24.*PI*tau);//RemainderEstimate
	      if(x<-1.0e+8){
		data.warningCount++;
		//if(data.PrintSC==1) PRINT("GetnAiT: Warning !!! Lower integration limit not found: RemainderEstimate(x=" + to_string(x) + ") = " + to_string_with_precision(test,16),data);
		test = 0.;
		break;
	      }
	    }
	    //bisection on real axes:
	    res = test;
	    res += AdaptiveBoole_nAiT(EnergyQ, s, maxIter, x, 100., 0., 0, RTPNA, data.AverageDensity[s], prefactor,U,ar,i,prefactor2,data);*/	   	    
	  }
	  else if(data.DIM==3){
	    double RTPNA = data.InternalAcc;//data.RelAcc;//
	    double prefactor = beta0*ar*tau*beta1;
	    double prefactor2 = beta2/beta1;
	    double test = 1., x = -1., arg = -tau*(U-x*ar);
	    while(ABS(test)>MP){//find lower integration limit:
	      x *= 2.; arg = -tau*(U-x*ar); data.nAiTevals[s][i][0]++;
	      test = beta0*CurlyA(x,data)*ar*tau*(beta1*polylog("1half",arg,data)+beta2*polylog("m3half",arg,data));
	      if(x<-1.0e+8){
		if(ABS(test)>MP){
		  data.warningCount++;
		  //if(data.PrintSC==1) PRINT("GetnAiT: Warning !!! Lower integration limit not found: Integrand(x=" + to_string(x) + ") = " + to_string_with_precision(test,16),data);
		}
		break;
	      }
	    }
	    double RemainderEstimate = AdaptiveBoole_nAiT(EnergyQ, s, maxIter, 2.*x, x, 0., 0, RTPNA, data.AverageDensity[s], prefactor,U,ar,i,prefactor2,data);
	    //if(ABS(test)>MP) if(data.PrintSC==1) PRINT("GetnAiT: RemainderEstimate(x=" + to_string(x) + ") = " + to_string_with_precision(RemainderEstimate,16),data);
	    //bisection on real axes:
	    res = RemainderEstimate;
	    res += AdaptiveBoole_nAiT(EnergyQ, s, maxIter, x, 100., 0., 0, RTPNA, data.AverageDensity[s], prefactor,U,ar,i,prefactor2,data);    
	  }
	}
	
	if(res!=res){
	  //cout <<"i:" << i << " res = " << res << " @ r = " << Norm(data.VecAt[i]) << " => res:=0" << endl;
	  res = 0.;
	}
	return res;
      }
    }
    
    //default:
    return 0.;  
}

void GetnAiT(int EnergyQ, int s, datastruct &data){//in HO units for 2D || in {eV,A} for 3D
  DelField(data.V[s], s, data.DelFieldMethod, data);

  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    if(data.SymmetryMask[i] || (data.Symmetry==3 && EnergyQ)){
      if(EnergyQ) data.TmpField1[i] = getDenEnAt(1, s, i, data);
      else data.Den[s][i] = getDenEnAt(0, s, i, data);
    }
  }
  
  data.FocalSpecies = s;
  if(EnergyQ){
      Regularize({{2}},data.TmpField1,data.RegularizationThreshold,data);
      ExpandSymmetry(0,data.TmpField1,data);
  }
  else ExpandSymmetry(1,data.Den[s],data);
  //cout << "data.Den[s][data.CentreIndex] = " << data.Den[s][data.CentreIndex] << endl;
}

void AdaptnAiT(datastruct &data){
  int nAiTmonitorSize = data.nAiTevals[0][0].size();
  for(int s=0;s<data.S;s++){
    vector<double> Accumulate(nAiTmonitorSize), nAiTmonitor(nAiTmonitorSize); fill(Accumulate.begin(), Accumulate.end(), 0.); fill(nAiTmonitor.begin(), nAiTmonitor.end(), 0.);
    for(int a=0;a<nAiTmonitorSize;a++){
      for(int i=0;i<data.GridSize;i++) nAiTmonitor[a] += (double)data.nAiTevals[s][i][a];
      Accumulate[a] += nAiTmonitor[a];
    }
    vector<string> nAiTmonitorString(nAiTmonitorSize);
    for(int a=0;a<nAiTmonitorSize;a++) nAiTmonitorString[a] = to_string((int)(nAiTmonitor[a]/((double)(data.GridSize))));
    if(data.PrintSC==1){
      PRINT(" AdaptnAiT[s=" + to_string(s) + "]: Averages:",data);
      PRINT("                #iterations (LowerLimitSearch) = " + nAiTmonitorString[0],data);
      PRINT("                 #evaluations (nAiT integrand) = " + nAiTmonitorString[1],data);
      PRINT("                              #relAccReachedQ = " + nAiTmonitorString[5],data);
      PRINT("                                    #iterMaxQ = " + nAiTmonitorString[2],data);
      PRINT("                        #NegligibleMagnitudeQ = " + nAiTmonitorString[4],data);    
      PRINT("                                      #DeltaQ = " + nAiTmonitorString[3] + "\n",data);
    }
  
    if(data.PrintSC==1){
      PRINT("#nAiT integrand evaluations (of last density evaluation) = " + to_string(Accumulate[1]),data);
      PRINT("Accumulated {#relAccReachedQ,#NegligibleMagnitudeQ,#iterMaxQ} = {" + to_string(Accumulate[5]) + "," + to_string(Accumulate[4]) + "," + to_string(Accumulate[2]) + "}",data);
    }
    
//     if(data.PrintSC==1){
//       PRINT("#nAiT(old) = " + to_string(data.OldnAiTEVALS[s]),data);
//       PRINT("#nAiT      = " + to_string(Accumulate[1]),data);
//       PRINT("BPreduction= " + to_string(data.BPreductionInProgress),data);
//     }
//     
//     if(Accumulate[2]>0.01*Accumulate[5]){//if maximal number of bisections reached in more than 1% of cases
//       data.MppVec[s][0] += 1.;
//       data.MppVec[s][1] += 1.; 
//     }
//     else if(Accumulate[1]>data.OldnAiTEVALS[s]){//#nAiT has increased compared with previous SC-iteration...
//       if(data.BPreductionInProgress){//...despite reduced number of BoolePairs
// 	data.MppVec[s][0] += 1.;//increase number of BoolePairs
// 	data.BPreductionInProgress = false;
//       }
//       else{//...along with increased number of BoolePairs
// 	data.MppVec[s][0] -= 1.;//reduce number of BoolePairs if possible; data.Mpp[0] is the minimum number of BoolePairs
// 	data.BPreductionInProgress = true;
//       }      
//     }
//     else{//#nAiT has decreased compared with previous SC-iteration...
//       if(data.BPreductionInProgress){//...along with reduced number of BoolePairs
// 	data.MppVec[s][0] -= 1.;//continue to reduce number of BoolePairs if possible; 
// 	data.BPreductionInProgress = true;	
//       }
//       else{//...despite increased number of BoolePairs
// 	data.MppVec[s][0] += 1.;//continue to increase number of BoolePairs
// 	data.BPreductionInProgress = false;
//       }        
//     }
//     
//     if(data.PrintSC==1) PRINT("BPreduction->" + to_string(data.BPreductionInProgress),data);
//     
//     data.MppVec[s][0] = min(data.MppVec[s][0],50.);//at most 50 BoolePairs
//     data.MppVec[s][0] = max(data.MppVec[s][0]-1.,data.Mpp[0]);// data.Mpp[0] is the minimum number of BoolePairs

    if(Accumulate[2]>0.01*Accumulate[5]){//if maximal number of bisections reached in more than 1% of cases
      data.MppVec[s][0] += 1.;
      data.MppVec[s][1] += 1.; 
    }
//     else if(Accumulate[1]>data.OldnAiTEVALS[s]){//if too many evaluations (nAiT integrand)
//       data.MppVec[s][0] = max(data.MppVec[s][0]-1.,data.Mpp[0]);//reduce number of BoolePairs if possible; data.Mpp[0] is the minimum number of BoolePairs
//     }
//     else data.MppVec[s][0] += 1.;
    
    data.MppVec[s][0] = min(data.MppVec[s][0],50.);//at most 50 BoolePairs
    
    data.OldnAiTEVALS[s] = Accumulate[1];
    
    if(data.PrintSC==1) PRINT(" update Mpp -> " + vec_to_str(data.MppVec[s]),data);
  }
}

double AdaptiveBoole_nAiT(int EnergyQ, int s, int maxIter, double a, double b, double prevInt, int iter, double RTPNA, double Compare, double prefactor,double U,double ar,int i,double prefactor2,datastruct &data){
  
  int BP = (int)data.MppVec[s][0], n = 4*BP+1;
  double x, halfinterval = 0.5*(b-a), Delta = 0.25*halfinterval/((double)BP), Inta = 0., Intb = 0.;
  
  vector<double> IntaVec(n), IntbVec(n);
  for(int yi=0;yi<n;yi++){
    x = a + (double)yi*Delta;
    IntaVec[yi] = nAiTintegrand(EnergyQ, s, x,prefactor,U,ar,i,prefactor2,data);
    IntbVec[yi] = nAiTintegrand(EnergyQ, s, x + halfinterval,prefactor,U,ar,i,prefactor2,data);
  }
  Inta = BooleRule1D(data.ompThreads, IntaVec,n,a,a+halfinterval,BP);
  Intb = BooleRule1D(data.ompThreads, IntbVec,n,a+halfinterval,b,BP);
  data.nAiTevals[s][i][1] += 2*n;
  double Int = Inta + Intb;
  
  //Compare = max(Compare,Int);
  
  if(iter>1 && ABS(Int)>0. && ABS((Int-prevInt)/Int)<RTPNA){ data.nAiTevals[s][i][5]++; return Int; }
  else if(iter>maxIter){ data.nAiTevals[s][i][2]++; return Int; }
  //else if(NegligibleMagnitudeQ(Compare,Int,-log10(RTPNA))){ data.nAiTevals[s][i][4]++; return Int; }
  else if(NegligibleMagnitudeQ(Compare,pow(2.,(double)iter)*Int,-log10(RTPNA))){ data.nAiTevals[s][i][4]++; return Int; }
  //if(iter>10 && pow(2.,(double)iter)*ABS(Int)<MP){ data.nAiTevals[s][i][4]++; return Int; }
  else if(Delta<MP){ data.nAiTevals[s][i][3]++; return Int; }
  else return AdaptiveBoole_nAiT(EnergyQ, s, maxIter, a, a+halfinterval, Inta, iter+1, RTPNA, Compare, prefactor,U,ar,i,prefactor2,data) + AdaptiveBoole_nAiT(EnergyQ, s, maxIter, a+halfinterval, b, Intb, iter+1, RTPNA, Compare, prefactor,U,ar,i,prefactor2,data);
}

double nAiTintegrand(int EnergyQ, int s, double x,double prefactor,double U,double ar,int i,double prefactor2,datastruct &data){
  //ensure that x<=100.
    if(data.DIM==1){
      //ToDo
    }
    else if(data.DIM==2){
      double zeta = EXP(U-ar*x)/data.TVec[s], LO = prefactor*data.TVec[s]*log(1.+1./zeta), NLO = prefactor*prefactor2*zeta/((1.+zeta)*(1.+zeta));
      if(LO!=LO){ LO = 0.; /*if(data.PrintSC==1)*/ PRINT("nAiTintegrand: Warning !!! LO!=LO at r=(" + vec_to_str(data.VecAt[i]) + ")",data); }
      if(NLO!=NLO){ NLO = 0.; /*if(data.PrintSC==1)*/ PRINT("nAiTintegrand: Warning !!! NLO!=NLO at r=(" + vec_to_str(data.VecAt[i]) + ")",data); }
      double AI; if(x>100.) AI = 0.; else if(x>-300.) AI = gsl_sf_airy_Ai(x,GSL_PREC_DOUBLE); else AI = cos(2.*(-x)*sqrt(-x)/3.-0.25*PI)/(SQRTPI*pow(-x,0.25));
      double result = AI*(LO + NLO);
      if(result!=result){ result = 0.; /*if(data.PrintSC==1)*/ PRINT("nAiTintegrand: Warning !!! result!=result at r=(" + vec_to_str(data.VecAt[i]) + ")",data); }
      return result;
      
//       double test, logzeta = (U-ar*x)/data.TVec[s], atilde = prefactor*CurlyA(x,data)*ar, logABSatilde = log(ABS(atilde)), b = 1.-data.Laplacian[s][i]/12., c = 2+data.Laplacian[s][i]/12.;
//       if(logzeta>37.){
// 	test = EXP(logABSatilde-3.*logzeta);
// 	if(ABS(b)>0.) test += Sign(b)*EXP(logABSatilde+log(ABS(b))-logzeta);
// 	if(ABS(c)>0.) test += Sign(c)*EXP(logABSatilde+log(ABS(c))-2.*logzeta);
// 	test *= Sign(atilde);
//       }
//       else if(logzeta<-37.){
// 	test = atilde;
// 	if(ABS(b)>0.) test += Sign(b)*EXP(logABSatilde+log(ABS(b))+2.*logzeta);
// 	if(ABS(c)>0.) test += Sign(c)*EXP(logABSatilde+log(ABS(c))+logzeta);
// 	test *= Sign(atilde);
//       }
//       else{
// 	double zeta = EXP(logzeta);
// 	test = atilde*(1.+b*zeta*zeta+c*zeta)/((1.+zeta)*(1.+zeta)*(1.+zeta));
//       }
//       return test;
      
      
    }
    else if(data.DIM==3){
      double arg = -(U-x*ar)/data.TVec[s];
      if(EnergyQ) return -prefactor*data.TVec[s]*CurlyA(x,data)*(polylog("3half",arg,data)+prefactor2*polylog("m1half",arg,data));
      else return prefactor*CurlyA(x,data)*(polylog("1half",arg,data)+prefactor2*polylog("m3half",arg,data));
    }
    
    //default
    return 0.;
}

void Getn3pFFT(int s, datastruct &data){//in HO units with hbar=m=omega=1 for 1D & 2D || in {eV,A} for 3D
  
  double UnitPrefactor; if(data.Units==1) UnitPrefactor = 1.; else if(data.Units==2) UnitPrefactor = 0.131163;
  
  int n = data.steps+1, n2=n*n;
  double x0 = -data.edge/2., Deltax=data.edge/(double)(data.steps), k0 = -PI/Deltax, Deltak=2.*PI/data.edge, Prefactor = pow(Deltax,data.DDIM) * data.degeneracy * AngularFactor(data.DIM) * pow(8.,data.DDIM/2.)/(pow(2.*PI,data.DDIM)*data.DDIM*pow(2.,data.DDIM));

  SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<n;i++){
    vector<double> qVec(data.DIM), rVec(data.DIM); qVec[0] = k0+(double)i*Deltak;
    double q2term,test,arg; int index;
    if(data.DIM==1){
      index = i;
      q2term = 0.125*qVec[0]*qVec[0];
      for(int xi=0;xi<n;xi++){
	rVec[0] = x0+(double)xi*Deltax;
	test = UnitPrefactor*(data.muVec[s]-data.V[s][xi])-q2term;
	if(test>0.){
	  test = sqrt(test);
	  arg = -qVec[0]*rVec[0];
	  data.ComplexField.at(index).at(0) += cos(arg)*test;
	  data.ComplexField.at(index).at(1) += sin(arg)*test;
	}
      }      
    }
    else for(int j=0;j<n;j++){
      qVec[1] = k0+(double)j*Deltak;
      if(data.DIM==2){
	index = i*n+j;
	q2term = 0.125*(qVec[0]*qVec[0]+qVec[1]*qVec[1]);
	for(int xi=0;xi<n;xi++){
	  rVec[0] = x0+(double)xi*Deltax;
	  for(int yj=0;yj<n;yj++){
	    test = UnitPrefactor*(data.muVec[s]-data.V[s][xi*n+yj])-q2term;
	    if(test>0.){
	      rVec[1] = x0+(double)yj*Deltax;
	      arg = -qVec[0]*rVec[0]-qVec[1]*rVec[1];
	      data.ComplexField.at(index).at(0) += cos(arg)*test;
	      data.ComplexField.at(index).at(1) += sin(arg)*test;
	    }
	  }
	}	
      }
      else if(data.DIM==3){
	for(int k=0;k<n;k++){
	  qVec[2] = k0+(double)k*Deltak;
	  index = i*n*n+j*n+k;
	  q2term = 0.125*(qVec[0]*qVec[0]+qVec[1]*qVec[1]+qVec[2]*qVec[2]);
	  for(int xi=0;xi<n;xi++){
	    rVec[0] = x0+(double)xi*Deltax;
	    for(int yj=0;yj<n;yj++){
	      rVec[1] = x0+(double)yj*Deltax;
	      for(int zk=0;zk<n;zk++){
		test = UnitPrefactor*(data.muVec[s]-data.V[s][xi*n*n+yj*n+zk])-q2term;
		if(test>0.){
		  rVec[2] = x0+(double)zk*Deltax;
		  test = test*sqrt(test);
		  arg = -qVec[0]*rVec[0]-qVec[1]*rVec[1]-qVec[2]*rVec[2];
		  data.ComplexField.at(index).at(0) += cos(arg)*test;
		  data.ComplexField.at(index).at(1) += sin(arg)*test;
		}
	      }
	    }
	  }	  
	}
      }
    }
    
  }
  MultiplyComplexField(data,data.ComplexField, data.DIM, n, Prefactor);
  fftParallel(FFTW_BACKWARD,data);
  
  CopyComplexFieldToReal(data,data.Den[s], data.ComplexField, 0, data.DIM, n);
  
}

void Getn3p(int s, datastruct &data){//original n3', all D, HO units & {eV,Angstrom}
  double mass = 1.; if(data.Units==2) mass = 0.131163;
  vector<double> Gammafactor(3); Gammafactor[0] = 0.5; Gammafactor[1] = 0.125; Gammafactor[2] = 0.02083333333333333;
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int a=0;a<data.IndexList.size();a++){
    int i = data.IndexList[a];
    vector<double> SpatialIntegrand(data.GridSize);
    for(int j=0;j<data.GridSize;j++){
      double PV = data.muVec[s]-data.V[s][j];
      if(PV>0.){
	double rPrime = 0.;
	PV = sqrt(2.*mass*PV);
	for(int DD=0;DD<data.DIM;DD++) rPrime += (data.VecAt[j][DD]-data.VecAt[i][DD])*(data.VecAt[j][DD]-data.VecAt[i][DD]);
	if(rPrime>MachinePrecision){
	  rPrime = sqrt(rPrime);
	  double z = 2.*rPrime*PV, Bessel; if(z>2.0e+12) Bessel = sqrt(2./(PI*z))*cos(z-0.5*data.DDIM*PI-0.25*PI); else Bessel = gsl_sf_bessel_Jn(data.DIM,z);
	  SpatialIntegrand[j] = pow(PV/(2.*PI*rPrime),data.DDIM)*Bessel;
	}
	else SpatialIntegrand[j] = pow(PV*PV/PI,data.DDIM)*Gammafactor[data.DIM-1];
      }
      else SpatialIntegrand[j] = 0.;
    }
    data.Den[s][i] = data.degeneracy*Integrate(data.ompThreads,data.method,data.DIM,SpatialIntegrand,data.frame);
  }
  data.FocalSpecies = s;
  ExpandSymmetry(1,data.Den[s],data);
}



void Getn3pT(int s, datastruct &data){//finite-temperature expression for n3'
  data.ySteps = (int)data.Mpp[0];/*4*/
  data.yStepsMax = 8*data.ySteps;/*256*/;
  
  data.Stride = data.Mpp[1];
  data.StrideMax = 8.*data.Stride;
  
  data.yStepsOriginal = data.ySteps;
  data.StrideOriginal = data.Stride;
  data.yCount = 0;  
  
  //switch to positive mu
  bool switchQ = false;
  double MuShift = 0.;
  if(data.muVec[s]<0.){
    switchQ = true;
    MuShift = 2.*ABS(data.muVec[s]);
    data.muVec[s] += MuShift;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++) data.V[s][i] += MuShift;
  }

  data.FocalSpecies = s;
  int n = data.EdgeLength, BisectionCount = 1;
  double yStart = data.Mpp[5];//1.0e-299;//0.;
  double AccumulatedPartNum = 1.0e-16, beta = 1.; if(data.Units==2) beta = 0.131163;
  double crit = data.InternalAcc;
  double gamma; if(data.DIM==1) gamma = SQRTPI; else if(data.DIM==2) gamma = 1.; else gamma = SQRTPI/2.;
  double TmpPartNumOld = 2., y, ymax = 40., PreviousIntegral = -1.23456789;
  double test, VMIN = data.V[s][data.CentreIndex]; for(int i=0;i<data.GridSize;i++){ test = data.V[s][i]; if(test<VMIN) VMIN = test; }
  double tau = data.muVec[s]/data.TVec[s], abssigmamax = ABS(tau*(1.-VMIN/data.muVec[s])), yEpsilon = max(1.0e-250,abssigmamax*EXP(-abssigmamax)/100.);
	
  SetField(data,data.Den[s],data.DIM,n,0.);
  double prefactor = data.degeneracy*pow(beta*data.muVec[s]/(2.*PI*tau),data.DDIM/2.)/gamma;
  n3pxyFFTPARALLEL(data.Den[s], prefactor, yEpsilon, BisectionCount, PreviousIntegral, AccumulatedPartNum, yStart, ymax, data.ySteps, crit, data);
  MultiplyField(data,data.Den[s],data.DIM,n,prefactor);
  
  //switch back to original mu
  if(switchQ){
    data.muVec[s] -= MuShift;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++) data.V[s][i] -= MuShift;
  }
  
  //Adaptn3pT(data);

}

void Adaptn3pT(datastruct &data){
  PRINT("ySteps = " + to_string(data.ySteps) + " Stride = " + to_string(data.Stride) + " yCount = " + to_string(data.yCount),data);
}

void n3pxyFFTPARALLEL(vector<double> &NewField, double prefactor, double yEpsilon, int BisectionCount, double PreviousIntegral, double &AccumulatedPartNum, double yStart, double yEnd, int ySteps, double crit, datastruct &data){
  
  int s = data.FocalSpecies, size = data.GridSize, n = data.EdgeLength, n2 = n*n, yEdge = ySteps+1;
  double a,b,c,d,Order0, Order0plus1, BisectionRatio = 1./pow(2.,(double)BisectionCount);
  double yStart1 = yStart;
  double yEnd1; if(yStart1<1.0e-300) yEnd1 = yEnd/data.Stride; else yEnd1 = yStart1+(yEnd-yStart)/data.Stride;
  double yStart2 = yEnd1, yEnd2 = yEnd;
  data.yCount += 2*yEdge;
  
  double Deltay1 = (yEnd1-yStart1)/((double)ySteps), Deltay1Half = Deltay1/2., Deltay2 = (yEnd2-yStart2)/((double)ySteps), Deltay2Half = Deltay2/2.;
  int yEpsilonFlag = 0;
  if(yStart1<1.0e-300){
    //cout << "GetEpsilonNewField...";
    yEpsilonFlag = 1; if(yEnd1<1.0e-290) yEpsilonFlag = 2;
    Order0 = GetEpsilonNewField(0,-1.23456789,yEnd1,crit,data.V[s],data);//writing to data.TmpField1
    Order0plus1 = prefactor*GetEpsilonNewField(1,-1.23456789,yEnd1,crit,data.V[s],data);//internally added to data.TmpField1 after Order0 result available
    if(ABS(Order0plus1)<1.0e-300) yEpsilonFlag = 3;
    a = Order0plus1;
    //cout << "...done" << endl;
  }
  else{
    SetComplexField(data,data.TmpComplexField, data.DIM, n, 0., 0.);
    for(int yi=0;yi<yEdge;yi++){
      data.y = yStart1+(double)yi*Deltay1;
      double BC = BooleCoefficient(yi,yEdge,Deltay1);
      FillComplexField(data.V[data.FocalSpecies],data);
      //BooleRule
      AddFactorComplexField(data,data.TmpComplexField, data.ComplexField, data.DIM, n, BC, BC);
      //RiemannRule:
//       if(yi==0 || yi==ySteps) AddFactorComplexField(data,AuxParameters.TmpComplexField, AuxParameters.ComplexField, data.DIM, n, Deltay1Half, Deltay1Half);
//       else AddFactorComplexField(data,AuxParameters.TmpComplexField, AuxParameters.ComplexField, data.DIM, n, Deltay1, Deltay1);
    }
    CopyComplexField(data,data.ComplexField, data.TmpComplexField, data.DIM, n);
    fftParallel(FFTW_BACKWARD, data);
    CopyComplexFieldToReal(data,data.TmpField1, data.ComplexField, 0, data.DIM, n);
    a = prefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  }
  
  SetComplexField(data,data.TmpComplexField, data.DIM, n, 0., 0.);
  for(int yi=0;yi<yEdge;yi++){
    data.y = yStart2+(double)yi*Deltay2;
    double BC = BooleCoefficient(yi,yEdge,Deltay2);
    FillComplexField(data.V[data.FocalSpecies],data);
    //BooleRule:
    AddFactorComplexField(data,data.TmpComplexField, data.ComplexField, data.DIM, n, BC, BC);
//     //RiemannRule:
/*    if(yi==0 || yi==ySteps) AddFactorComplexField(data,AuxParameters.TmpComplexField, AuxParameters.ComplexField, data.DIM, n, Deltay2Half, Deltay2Half);
    else AddFactorComplexField(data,AuxParameters.TmpComplexField, AuxParameters.ComplexField, data.DIM, n, Deltay2, Deltay2); */   
  }
  CopyComplexField(data,data.ComplexField, data.TmpComplexField, data.DIM, n);
  fftParallel(FFTW_BACKWARD, data);
  CopyComplexFieldToReal(data,data.TmpField2, data.ComplexField, 0, data.DIM, n);
  b = prefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  
  bool nanFlag = false;
  if(a!=a || b!=b){
    nanFlag = true;
    PRINT("a=" + to_string_with_precision(a,16) + "; b=" + to_string_with_precision(b,16) + "(" + to_string_with_precision(PreviousIntegral,16) + ") @ [" + to_string_with_precision(yStart1,16) + "," + to_string_with_precision(yEnd1,16) + "] and " + "[" + to_string_with_precision(yStart2,16) + "," + to_string_with_precision(yEnd2,16) + "] with ySteps = " + to_string(ySteps),data);
    //try again with 2*ySteps:
    ySteps *= 2;
    if(ySteps<data.yStepsMax) n3pxyFFTPARALLEL(NewField, prefactor, yEpsilon, BisectionCount, PreviousIntegral, AccumulatedPartNum, yStart, yEnd, ySteps, crit, data);
    PRINT("a,b OK for ySteps = " + to_string(ySteps),data);
  }
  else ySteps = data.yStepsOriginal;  
  
  //both a+b and PreviousIntegral should be positive, and may be very small negative due to numerical errors only
  c = max(ABS(a+b+PreviousIntegral),ABS(a+b)+ABS(PreviousIntegral));//c = ABS(a+b+PreviousIntegral);
  d = pow(2.,(double)BisectionCount);
  int CHECKok = 0, PrintFlag = (int)data.Mpp[4];
  double NegAcc = max(1.0e-10,data.RelAcc);
  if(yEpsilonFlag==2){ CHECKok = 1; if(PrintFlag>0) PRINT("n3pxyFFTPARALLEL: yEnd1 too small! (further bisectioning useless)",data); }
  else if(yEpsilonFlag==3 && yEnd1<yEpsilon){ CHECKok = 2; if(PrintFlag>0) PRINT("n3pxyFFTPARALLEL:  proper yEpsilon regime reached and epsilon-integral is zero @ yEnd1=" + to_string_with_precision(yStart1,16) + " (further bisectioning unnecessary)",data); }
  else if(ABS(a+b)<NegAcc*data.Abundances[data.FocalSpecies]){ CHECKok = 3; if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL: absolute contribution negligible:",data); }
  else if(ABS(AccumulatedPartNum)>0. && (d*ABS((a+b)/AccumulatedPartNum)<crit || ABS((a+b)/AccumulatedPartNum)<NegAcc) ){ CHECKok = 4; if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL: relative contribution negligible",data); }
  else if(ABS((ABS(a+b)-ABS(PreviousIntegral)))<crit*ABS(c)){ CHECKok = 5; if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL: accuracy criterion satisfied:",data); }
  //else if(ABS((ABS(a+b-PreviousIntegral)))<crit*ABS(c)){ CHECKok = 5; if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL: accuracy criterion satisfied:",data); }

  int BisectionCountMinimum = (int)data.Mpp[2]; /*3;*/
  int BisectionCountLimit = (int)data.Mpp[3]; /*400;*/
  if( (BisectionCount>BisectionCountMinimum && CHECKok>0) || BisectionCount==BisectionCountLimit){
    if(BisectionCount<BisectionCountLimit){
      if(yEpsilonFlag==0 || yEpsilonFlag==3) AddField(data,NewField, data.TmpField1, data.DIM, n); else a=0.;
      AddField(data,NewField, data.TmpField2, data.DIM, n);
      AccumulatedPartNum += a+b;
    }
    if(PrintFlag>1 || (yStart1>0. && yStart1<1.0e-300) || BisectionCount==BisectionCountLimit){
      PRINT("n3pxyFFTPARALLEL(Bisection=" + to_string(BisectionCount) + ",y=[" + to_string_with_precision(yStart,16) + "," + to_string_with_precision(yEnd,16) + "," + to_string_with_precision(yEnd-yStart,16) + "],ySteps=" + to_string(ySteps) + ",stride=" + to_string((int)data.Stride) + "): **** " + to_string_with_precision((a+b),16) + "(" + to_string_with_precision(PreviousIntegral,16) + "[N(accumulated)=" + to_string_with_precision(AccumulatedPartNum,16) + "]) ****",data);
      if(BisectionCount==BisectionCountLimit){
	PRINT("abort sub-n3pxyFFTPARALLEL: BisectionCount==" + to_string(BisectionCountLimit),data);
	data.warningCount++; PRINT("n3pxyFFTPARALLEL: Warning!!! BisectionCountLimit reached.",data);
      }
    }
  }
  else{
    if(BisectionCount>1){ data.Stride = min(data.Stride*2.,data.StrideMax); /*ySteps = min(ySteps+4,data.yStepsMax);*/ }
    if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL(a): BisectionCount = " + to_string(BisectionCount) + " ySteps = " + to_string(data.ySteps) + " Stride = " + to_string(data.Stride) + " yCount = " + to_string(data.yCount),data);
    n3pxyFFTPARALLEL(NewField, prefactor, yEpsilon, BisectionCount+1, a, AccumulatedPartNum, yStart1, yEnd1, ySteps, crit, data);
    if(BisectionCount>1){ data.Stride = max(data.Stride/2.,2.); /*ySteps = max(ySteps-4,4);*/ }
    if(PrintFlag==2) PRINT("n3pxyFFTPARALLEL(b): BisectionCount = " + to_string(BisectionCount) + " ySteps = " + to_string(data.ySteps) + " Stride = " + to_string(data.Stride) + " yCount = " + to_string(data.yCount),data);
    n3pxyFFTPARALLEL(NewField, prefactor, yEpsilon, BisectionCount+1, b, AccumulatedPartNum, yStart2, yEnd2, ySteps, crit, data);    
  }
  
}

double GetEpsilonNewField(int order, double PreviousResult,double yEpsilon,double crit, vector<double> &InputField, datastruct &data){
  int n = data.EdgeLength, n2=n*n;
  double mu = data.muVec[data.FocalSpecies], tau = mu/data.TVec[data.FocalSpecies];
  double beta = 1.; if(data.Units==2) beta = 0.131163;
  double kappat = 1./(8.*beta*data.TVec[data.FocalSpecies]);
  
  double OrderFactor = tau; if(order==1) OrderFactor = 2.*tau;

  //compute FFT of f_yEpsilon
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    double tauvar = OrderFactor*(1.-InputField[i]/mu);
    data.ComplexField[i][0] = EXP(tauvar);//EXP(tauvar-yEpsilon*EXP(tauvar));
    data.ComplexField[i][1] = 0.;
  }
  fftParallel(FFTW_FORWARD, data);

  //produce new kappa-spline for given dimension and given yEpsilon
  double maxFk = MaxAbsOfComplexField(data,data.ComplexField, data.DIM, n);
  spline1dinterpolant gfuncspline = gfuncSpline(order, yEpsilon, crit, maxFk, data);
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    double GFUNC = spline1dcalc(gfuncspline,kappat*data.Norm2kVecAt[i]);
    data.ComplexField[i][0] *= GFUNC;
    data.ComplexField[i][1] *= GFUNC;
  }
  fftParallel(FFTW_BACKWARD, data);

  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    if(order==0) data.TmpField1[i] = data.ComplexField[i][0];
    else data.TmpField1[i] -= data.ComplexField[i][0];
  }
  
  double test = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  
  if(test==test) return test;
  else return 0.;
}

spline1dinterpolant gfuncSpline(int order, double yEpsilon, double crit, double maxFk, datastruct &data){
  
  vector<double> Radialkappa(data.steps/2+1), RadialgInt(data.steps/2+1);
  double beta = 1.; if(data.Units==2) beta = 0.131163;
  double prefactor = 1./(8.*beta*data.TVec[data.FocalSpecies]), xStart=0.; if(data.DIM==1) xStart = 1.0e-15;
  int BP = 1, BisectionCount=0;
  
  //produce \int\dy G_y(k) for all k on the halfdiagonal in k-space
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int hdi=0;hdi<data.HDI.size();hdi++){
    Radialkappa[hdi] = prefactor*data.Norm2kVecAt[data.HDI[hdi]];
    if(data.y<1.0e-300) RadialgInt[hdi] = 0.;
    else{//perform x-integral to within accuracy of crit
      RadialgInt[hdi] = GetRadialgInt(order, yEpsilon,Radialkappa[hdi],maxFk,BisectionCount,xStart,700.,-1.23456789,EXP(-700.),BP,crit,data);
      if(ABS(maxFk*RadialgInt[hdi])<MP) RadialgInt[hdi] = 0.;
    }    
  }
  
//   //produce \int\dy G_y(k) for all k on the halfdiagonal in k-space
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)  
//   for(int rk=0;rk<data.steps/2+1;rk++){
//     double k2 = sqrt(data.DDIM)*data.Deltak*(double)rk; k2 *= k2; Radialkappa[rk] = k2/(8.*beta*data.TVec[data.FocalSpecies]); 
//     if(data.y<1.0e-300) RadialgInt[rk] = 0.;
//     else{//perform x-integral to within accuracy of crit
//       size_t NumPoints = 10;
//       double result, error;
//       int BP = 32, BisectionCount=0;
//       double x, xStart=0., xEnd=700., PreviousIntegral = -1.23456789, AccumulatedIntegral = EXP(-700.), POW, arg; if(data.DIM==1) xStart = 1.0e-15;
// 
//       RadialgInt[rk] = GetRadialgInt(order, yEpsilon,Radialkappa[rk],maxFk,BisectionCount,xStart,xEnd,PreviousIntegral,AccumulatedIntegral,BP,crit,data);
//       if(ABS(maxFk*RadialgInt[rk])<MP) RadialgInt[rk] = 0.;
//   
//     }
//   }
  
  //produce interpolation
  real_1d_array radialr,radialf;
  spline1dinterpolant RadialSpline;
  radialr.setcontent(Radialkappa.size(), &(Radialkappa[0]));
  radialf.setcontent(Radialkappa.size(), &(RadialgInt[0]));
  //spline1dbuildcubic(radialr, radialf, RadialSpline);
  spline1dbuildmonotone(radialr, radialf, RadialSpline);
  
  return RadialSpline;
}

double GetRadialgInt(int order, double yEpsilon, double kappa, double maxFk, int BisectionCount,double xStart, double xEnd, double PreviousIntegral, double AccumulatedIntegral, int BP, double crit, datastruct &data){
  double a,b,c,x,POW,arg,OrderFactor = 1.;
  int n = 4*BP+1; double xsteps = ((double)(n-1));
  
  if(BisectionCount==0){
    double deltax = (xEnd-xStart)/xsteps;
    vector<double> f(n);
    for(int i=0;i<n;i++){
      x = xStart+(double)i*deltax;
      if(data.DIM==3) POW = sqrt(x); else if(data.DIM==2) POW = 1.; else POW = 1./sqrt(x);
      arg = -yEpsilon*EXP(x+kappa);
      if(order==1) OrderFactor = (yEpsilon+EXP(-x-kappa));
      f[i] = POW*EXP(-x-kappa)*(1.-EXP(arg))*OrderFactor;
    }
    PreviousIntegral = BooleRule1D(data.ompThreads, f,n,xStart,xEnd,BP);
  }
  
  BisectionCount++;
  
  double xStart1 = xStart, xEnd1 = xStart1+0.5*(xEnd-xStart), deltax1 = (xEnd1-xStart1)/xsteps, xStart2 = xEnd1, xEnd2 = xEnd, deltax2 = (xEnd2-xStart2)/xsteps;
  vector<double> fa(n),fb(n);
  for(int i=0;i<n;i++){
    x = xStart1+(double)i*deltax1;
    if(data.DIM==3) POW = sqrt(x); else if(data.DIM==2) POW = 1.; else POW = 1./sqrt(x);
    arg = -yEpsilon*EXP(x+kappa);
    if(order==1) OrderFactor = (yEpsilon+EXP(-x-kappa));
    fa[i] = POW*EXP(-x-kappa)*(1.-EXP(arg))*OrderFactor;
    x = xStart2+(double)i*deltax2;
    if(data.DIM==3) POW = sqrt(x); else if(data.DIM==2) POW = 1.; else POW = 1./sqrt(x);
    arg = -yEpsilon*EXP(x+kappa);
    if(order==1) OrderFactor = (yEpsilon+EXP(-x-kappa));
    fb[i] = POW*EXP(-x-kappa)*(1.-EXP(arg))*OrderFactor;
  }
  a = BooleRule1D(data.ompThreads, fa,n,xStart1,xEnd1,BP);
  b = BooleRule1D(data.ompThreads, fb,n,xStart2,xEnd2,BP);

  double res = a+b;
  c = ABS(res+PreviousIntegral);

  int smalldeltaxFlag = 0, weightedabscontribFlag = 0, relcontribFlag = 0, abFlag = 0, resFlag = 0, printFlag = 0;
  double weight = pow(2.,(double)BisectionCount), weightedabscontrib = maxFk*ABS(weight*res), relcontrib = 2.*crit;
  if(AccumulatedIntegral>0.) relcontrib = ABS(weight*res/AccumulatedIntegral);
  
  if(deltax1<1.0e-13 || deltax2<1.0e-13){ smalldeltaxFlag = 1; if(printFlag) PRINT("GetRadialgInt: W A R N I N G! deltax too small",data); }
  if(weightedabscontrib<1.0e-16){ weightedabscontribFlag = 1; if(printFlag) PRINT("GetRadialgInt: absolute contribution negligible",data); }
  if(relcontrib<crit){ relcontribFlag = 1; if(printFlag) PRINT("GetRadialgInt: relative contribution negligible",data); }
  if(ABS(b)>0.) if(ABS(a/b)>1.0e+16 || ABS(a/b)<1.0e-16){ abFlag=1; if(printFlag) PRINT("GetRadialgInt: W A R N I N G! MachinePrecision reached",data); }
  if(ABS(a)>0.) if(ABS(b/a)>1.0e+16 || ABS(b/a)<1.0e-16){ abFlag=1; if(printFlag) PRINT("GetRadialgInt: W A R N I N G! MachinePrecision reached",data); }
  if(AccumulatedIntegral>0. && ABS(res)<1.0e-15*AccumulatedIntegral){ resFlag = 1; if(printFlag) PRINT("GetRadialgInt: W A R N I N G! res too small for accumulation",data); }
  
  if( (ABS(c)>1.0e-290 && ABS((res-PreviousIntegral)/c)<crit) || smalldeltaxFlag || weightedabscontribFlag || relcontribFlag || abFlag || resFlag ){
    //if(printFlag) PRINT("GetRadialgInt(Bisection=" + to_string(BisectionCount) + ",BP=" + to_string(BP) + ",x=[" + to_string_with_precision(xStart) + "," + to_string_with_precision(xEnd) + "],weightedabscontrib=" + to_string_with_precision(weightedabscontrib) + ",relcontrib=" + to_string_with_precision(relcontrib) + "): **** " + to_string_with_precision(a+b) + "(" + to_string_with_precision(PreviousIntegral) + "[" + to_string_with_precision(AccumulatedIntegral) + "])" + ",maxFk=" + to_string_with_precision(maxFk) + "  ****",data);
    return res;
  }
  else{
    //if(printFlag) PRINT("GetRadialgInt(Bisection=" + to_string(BisectionCount) + ",BP=" + to_string(BP) + ",x=[" + to_string_with_precision(xStart) + "," + to_string_with_precision(xEnd) + "],weightedabscontrib=" + to_string_with_precision(weightedabscontrib) + ",relcontrib=" + to_string_with_precision(relcontrib) + "):      " + to_string_with_precision(a+b) + "(" + to_string_with_precision(PreviousIntegral) + ")",data);
    BP = min(2*BP,512);
    AccumulatedIntegral += GetRadialgInt(order,yEpsilon,kappa,maxFk,BisectionCount,xStart1,xEnd1,a,AccumulatedIntegral,BP,crit,data);
    //BP = max(BP/2,32);
    AccumulatedIntegral += GetRadialgInt(order,yEpsilon,kappa,maxFk,BisectionCount,xStart2,xEnd2,b,AccumulatedIntegral,BP,crit,data);
    return AccumulatedIntegral;
  }
  
}

void FillComplexField(vector<double> &InputField, datastruct &data){

  //cout << "parallel FieldAssignment" << endl;
  //for(int tcount = 0;tcount<7;tcount++){
    double tau = data.muVec[data.FocalSpecies]/data.TVec[data.FocalSpecies];
    double threshold = 100.;/*700.-(double)tcount*100.;*/
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double tauvar = tau*(1.-InputField[i]/data.muVec[data.FocalSpecies]);
      data.ComplexField[i][0] = EXPt(tauvar-data.y*EXP(tauvar),threshold);//not: EXP(tauvar-data.y*EXP(tauvar)); ?
      data.ComplexField[i][1] = 0.;
    }
    fftParallel(FFTW_FORWARD, data);
    
//     int FieldOkQ = 1;
//     #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int i=0;i<data.GridSize;i++) if(data.ComplexField[i][0]!=data.ComplexField[i][0] || data.ComplexField[i][1]!=data.ComplexField[i][1]) FieldOkQ = 0;
//     if(FieldOkQ) break;
//     else{ data.warningCount++; PRINT("FillComplexField: Warning!!! ComplexField not OK !!! ",data); }
  //}
      
  double beta = 1.; if(data.Units==2) beta = 0.131163;
  double alpha = 1./(8.*beta*data.TVec[data.FocalSpecies]), logy = log(data.y);
  //for(int tcount = 0;tcount<7;tcount++){
    //double threshold = 700.-(double)tcount*100.;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double kappa = alpha*data.Norm2kVecAt[i], kappatilde = logy+kappa, GFUNC;
      if(kappatilde<7. && kappatilde >-threshold){
	if(data.DIM==1) GFUNC = spline1dcalc(data.Gykspline1D,kappatilde); 
	//else if(data.DIM==2) GFUNC = -std::tr1::expint(-max(1.0e-300,min(1.0e+300,EXP(kappa)*data.y)));
	//else if(data.DIM==2) boost::math::expint(1,max(1.0e-300,min(1.0e+300,EXP(kappa)*data.y)));
	else if(data.DIM==2) GFUNC = spline1dcalc(data.Gykspline2D,kappatilde);
	else if(data.DIM==3) GFUNC = spline1dcalc(data.Gykspline3D,kappatilde); 
      }
      else GFUNC = 0.;
      data.ComplexField[i][0] *= GFUNC;
      data.ComplexField[i][1] *= GFUNC;
    }
    
//     int FieldOkQ = 1;
//     #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int i=0;i<data.GridSize;i++) if(data.ComplexField[i][0]!=data.ComplexField[i][0] || data.ComplexField[i][1]!=data.ComplexField[i][1]) FieldOkQ = 0;
//     if(FieldOkQ) break;
//     else{ data.warningCount++; PRINT("FillComplexField: Warning!!! ComplexField not OK !!! ",data); }
  //}

}

double getn7AIintegral(double a, double b, double STEPS, double prevres, int iter, int MAXITER, int &NumMAXITER, double dist, double LO, double NLO, datastruct &data){
  if(NumMAXITER>0 || a<-1.0e+6) return 0.;
  else if(iter>MAXITER) return prevres;
  else{
    double m = 0.5*(a+b), deltax = 0.5*(b-a)/((double)STEPS);
    vector<double> aFrame = {{a},{m}}, bFrame = {{m},{b}}, aField(STEPS+1), bField(STEPS+1);
    for(int k=0;k<STEPS+1;k++){
      double ax = a + deltax*(double)k, bx = m + deltax*(double)k;
      double ak7 = LO-ax*NLO, bk7 = LO-bx*NLO;
      if(ak7>MP){
	ak7 = sqrt(ak7);
	double AI; if(ax>100.) AI = 0.; else if(ax>-300.) AI = gsl_sf_airy_Ai(ax,GSL_PREC_DOUBLE); else AI = cos(2.*(-ax)*sqrt(-ax)/3.-0.25*PI)/(SQRTPI*pow(-ax,0.25));
	aField[k] = AI*ak7*gsl_sf_bessel_Jn(1,2.*dist*ak7);
      }
      else aField[k] = 0.;
      if(bk7>MP){
	bk7 = sqrt(bk7);
	double AI; if(bx>100.) AI = 0.; else if(bx>-300.) AI = gsl_sf_airy_Ai(bx,GSL_PREC_DOUBLE); else AI = cos(2.*(-bx)*sqrt(-bx)/3.-0.25*PI)/(SQRTPI*pow(-bx,0.25));
	bField[k] = AI*bk7*gsl_sf_bessel_Jn(1,2.*dist*bk7);
      }
      else bField[k] = 0.;      
    } 
    double resa = Integrate(data.ompThreads,data.method, 1, aField, aFrame)/(2.*PI*dist), resb = Integrate(data.ompThreads,data.method, 1, bField, bFrame)/(2.*PI*dist);
    
    double res = resa+resb, asym = ABS(ASYM(res,prevres));
    if(iter==MAXITER && asym>sqrt(data.InternalAcc)){
      NumMAXITER++;
      cout << "MAXITER reached: a = " << a << " b = " << b << " res = " << res << " prevres = " << prevres << " ASYM = " << asym << endl;
      return 0.;
    }
    double Avtmpfieldj = data.AverageDensity[0]/(data.area*data.degeneracy), test = 1.0e-4*Avtmpfieldj;
    if( iter > 2 && (asym<data.InternalAcc || deltax<MachinePrecision*(b-a)*(pow(2.,(double)iter)*(double)STEPS) || (ABS(res)<test && ABS(prevres)<test) ) ) return res;
    else return getn7AIintegral(a,m,STEPS,resa,iter+1,MAXITER,NumMAXITER,dist,LO,NLO,data) + getn7AIintegral(m,b,STEPS,resb,iter+1,MAXITER,NumMAXITER,dist,LO,NLO,data);
  }
}

double GetID(const double tauThreshold, const int s, const int i, const double r, const int ip, const double rp, datastruct &data){
	if(ABS(rp-r)<MP){//CASE rp==r
		double tau = pow(data.GradSquared[s][ip]/(3.*data.U*data.U), 1./3.);
        double nu = 2.*(data.muVec[s] - (data.V[s][i] + 2.*data.V[s][ip])/3.)/data.U;
        double Dfactorial = data.DDIM; if(data.DIM==3) Dfactorial = 6.;
        double prefactor = POW(1./(2.*PI),data.DIM)/Dfactorial;
        if(tau<tauThreshold*ABS(nu)){//CASE tau==0
          	if(nu>0.) return prefactor * POW(nu,data.DIM);
            else return 0.;
        }
        else{//CASE tau>0, see KD.nb
          	//version1:
          	double res = 0.;
            double sigma = nu/tau;
            double A = 1.-CurlyA(sigma,data);
            double ai = tau*tau*gsl_sf_airy_Ai(sigma,GSL_PREC_DOUBLE);
            double aip = tau*gsl_sf_airy_Ai_deriv(sigma,GSL_PREC_DOUBLE);
          	if(data.DIM==1) res = nu*A-aip;
          	else if(data.DIM==2) res = nu*nu*A-nu*aip-ai;
          	else if(data.DIM==3) res = nu*nu*nu*A-nu*nu*aip-nu*ai;
            return prefactor*res;

          	//version2:
     //      	double pow316 = pow(3.,1./6.);
     //      	double pow313 = pow316*pow316;
     //      	double pow323 = pow313*pow313;
     //        double Gammam43 = gsl_sf_gamma(-4./3.);
     //        double Gamma23 = gsl_sf_gamma(2./3.), Gammam23 = gsl_sf_gamma(-2./3.);
     //        double Gamma13 = gsl_sf_gamma(1./3.), Gammam13 = gsl_sf_gamma(-1./3.);
     //        double Gamma113 = gsl_sf_gamma(11./3.);
     //        double sigma = nu/tau;
     //        double arg = POW(sigma,3)/9.;
     //      	if(data.DIM==1){//ToDo
     //          	double HG1F2a = HG1F2(1./3.,2./3.,4./3,arg);
     //            double HG1F2b = HG1F2(2./3.,4./3.,5./3,arg);
     //          	double AIP = gsl_sf_airy_Ai_deriv(sigma,GSL_PREC_DOUBLE);
     //          	return (2.*nu/3.-tau*AIP+nu*sigma*HG1F2a/(pow323*Gamma23)-nu*sigma*sigma*pow316*Gamma23*HG1F2b/(4.*PI)) / (2.*PI);
     //        }
     //        else if(data.DIM==2){//ToDo
     //          	return 0.;
     //        }
     //        else if(data.DIM==3){//ToDo (boost seems to get it right, check TestMat in KD.nb, but something is wrong here nonetheless)
     //                double InfiniteAiryIntegral = 2.*nu*nu*nu/3. + pow323*nu*nu*tau/Gamma13 + pow323*pow323*nu*tau*tau/Gammam13 - 4.*tau*tau*tau/3.;
     //
     //                //option1 (see KD.nb)
     //                double HG1F2a = HG1F2(-1./3.,4./3.,5./3,arg);
     //                double HG1F2b = HG1F2(-2./3.,2./3.,4./3,arg);
					// double FiniteAiryIntegral = (nu*tau/pow323) * ( 2.*nu*Gammam43/(pow316*PI) * (HG1F2a-1.) - 3.*tau/Gamma23 * (HG1F2b-1.) );
     //
     //                //option2 (see KD.nb)
     //                // double AI = gsl_sf_airy_Ai(sigma,GSL_PREC_DOUBLE);
     //                // double HG1F2a = HG1F2(1./3.,2./3.,4./3,arg);
     //                // double HG1F2b = HG1F2(4./3.,2./3.,7./3,arg);
     //                // double HG1F2c = HG1F2(2./3.,4./3.,5./3,arg);
     //                // double HG1F2d = HG1F2(5./3.,4./3.,8./3,arg);
     //                // double FiniteAiryIntegral = (1./4800.*PI) * ( -14400.*PI*nu*tau*tau*AI - 2430.*pow316*nu*nu*tau*Gamma113 + 200.*pow323*pow316*Gamma13*(4*nu*nu*nu*sigma*HG1F2a+tau*tau*(12.*nu-POW(sigma,4)*tau*HG1F2b)) + 81.*pow316*sigma*sigma*Gamma113*(-5.*nu*nu*nu*HG1F2c+2.*nu*nu*nu*HG1F2d) );
     //
     //          		return prefactor * ( InfiniteAiryIntegral + FiniteAiryIntegral );
     //        }
        }
    }
    else{//ToDo? //CASE rp!=r
      	return 0.;
    }

    return 0.;
}

double HG1F2(const double a1,const double b1,const double b2,const double x){
  double res;
  mp_type mpHG1F2;
  try {
    mpHG1F2 = boost::math::hypergeometric_pFq_precision( {mp_type(a1)}, {mp_type(b1),mp_type(b2)}, mp_type(x), pFq_NumDigitsPrecision, pFq_TimeoutSeconds, my_pFq_policy );
    res = static_cast<double>(mpHG1F2);
    //data.TestMat.push_back({arg,res});
  } catch (const boost::math::evaluation_error& e) {
    std::cerr << "Error in pFq: " << e.what() << std::endl;
    res = 0.;
  }
  return res;
}



void Getn7(int s, datastruct &data){
  	struct KDintegrationParams KDip;
	//BEGIN USER INPUT
	int METHOD = 3;//0: brute-force integration --- 1: bisection --- 2: accumulative bisection --- 3: Plugin_KD (Alex & Tri) --- 4: new bisection (MIT)
	//for METHOD==2:
	double TargetAcc = sqrt(data.InternalAcc), Avtmpfieldj = data.AverageDensity[0]/(data.area*data.degeneracy), TEST = 1.0e-4*Avtmpfieldj;
	//for METHOD<3:
	double ZEROGRAD2AT = 1.0e-6, OMEGA = 100.;
	int MAXITER = 10, MAXCOUNT = 10, STEPS = 8192, MAXFIELDSTEPS = 200000000/*maximum: 2.0e+8*/;
	//for METHOD==3:
	//check 'double KD()' in Plugin_KD.cpp
	//for METHOD==4:
	KDparams KDparameters;
	KDparameters.RelCrit = 1.0e-2;//2.0e-3;//
	KDparameters.UpperLimit = 1.0e+3;//1.0e+3;//
	KDparameters.IntLimitCount = 1000;
	KDparameters.relCrit = 2.0e-3;//1.0e-3;//
	KDparameters.absCrit = 1.0e-5;//1.0e-6;//
	KDparameters.minIter = 3;
	KDparameters.maxIter = 100;//
	KDparameters.betaThreshold = MachinePrecision;//1.0e-6;//
	//KDparameters.epsTweeze = 1.0e-12;
	KDparameters.epsIntLimits = 1.0e-2;
	KDparameters.Threads = 1;
	KDparameters.prefactor = 2./PI;
	KDparameters.BaseScale = 1.;
	KDparameters.bStepScale = 1.23456789;
	KDparameters.minDelta = 1.0e-16;
	KDparameters.distinguishabilityThreshold = 1.0e-14;
	KDparameters.printQ = 0;
	KDparameters.CompareKD = true;
    KDip.contourQ = true;
	//data.Print = 1;
    double tauThreshold = 1.0e-10;
	//END USER INPUT
	
	vector<vector<struct ComputeKD>> COMPUTEKD;

	DelField(data.V[s], s, data.DelFieldMethod, data);
    cout << "Getn7 derivatives computed" << endl;

    double ABSERR;

	if( data.Symmetry==1 || METHOD<3 ){
      	data.TestMat.resize(0);
        if(data.Symmetry==1){//CASE isotropic

            vector<int> pf = primeFactors((int)omp_get_max_threads(),0);
            cout << IntVec_to_CommaSeparatedString(pf) << endl;
            vector<int> tn = GetNestedThreadNumbers(pf,0);
            cout << "Getn7 thread numbers: Outer(" << tn[0] << ") Inner(" << tn[1] << ")" << endl;

           	KDip.num_outer_threads = tn[0];
            KDip.num_inner_threads = tn[1];

            // Outer parallel region
            #pragma omp parallel num_threads(KDip.num_outer_threads)
            {
              	// Outer loop with static scheduling
              	#pragma omp for schedule(dynamic)
    	    	for(int k=0;k<data.HDI.size();k++){
            		int i = data.HDI[k];
            		double r = Norm(data.VecAt[data.HDI[k]]);
               		vector<double> isoframe(2);
               		vector<double> tmpfield(data.HDI.size(),0.);
               		for(int j = 0; j < data.HDI.size(); j++){//fill integrand of g*\int_0^\infty(dr')
               			double rp = Norm(data.VecAt[data.HDI[j]]);
               			if(ABS(rp-r)<MP) tmpfield[j] = 0.;// 4.*PI*rp*rp*GetID(tauThreshold,s,i,r,data.HDI[j],rp,data);//wrong//ToDo
               			else{//CASE rp!=r
                    		if(data.DIM==1){//ToDo

                    		}
                    		else if(data.DIM==2){//ToDo

                    		}
                   			else if(data.DIM==3){//CASE D==3
                    			double p2 = (rp+r)*(rp+r);
                    			double m2 = (rp-r)*(rp-r);
                    			double alpha = 8.*(data.muVec[s] - (data.V[s][i] + 2.*data.V[s][data.HDI[j]])/3.)/data.U;
                    			double beta = 4.*pow(data.GradSquared[s][data.HDI[j]]/(3.*data.U*data.U), 1./3.);
                    			double Aplus = p2*alpha;
                    			double Aminus = m2*alpha;
                    			double Bplus = p2*beta;
                    			double Bminus = m2*beta;
                    			if(r<MP){//CASE r==0
                       				tmpfield[j] = KD(3, Aplus, Bplus, ABSERR, data.InternalAcc, KDip)/(16.*PI*PI*POW(rp,4));
                    			}
                    			else{//CASE r>0
                      				if(rp<MP){//CASE rp==0
	                   					tmpfield[j] = 0.;
    	               				}
                        			else{//CASE rp>0
                          				tmpfield[j] = rp/(r*32.*PI*PI) * (KD(2, Aminus, Bminus, ABSERR, data.InternalAcc, KDip)/(m2*m2) - KD(2, Aplus, Bplus, ABSERR, data.InternalAcc, KDip)/(p2*p2));
                        			}
                      			}
                        	}
                    	}
                	}
            		isoframe[0] = 0.;
                	isoframe[1] = sqrt(data.DDIM)*data.frame[1];
                	data.Den[s][i] = data.degeneracy*Integrate(data.ompThreads, data.method, 1, tmpfield, isoframe);
                	cout << "density[i=" << i << "] = " << data.Den[s][i] << endl;
          		}
            }


		}
// 		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
// 		for(int i=0;i<data.GridSize;i++){
// 			if(data.SymmetryMask[i]){
//
//                 if(data.Symmetry==1){
//                   	double r = Norm(data.VecAt[i]);
//
//                 }
// 				else if(METHOD<3){
//                   	int maxFieldSteps = 100, NumNLOthreshold = 0;
//                   	double gamma = gsl_sf_gamma(1.+data.DDIM), partialSum = 0., maxdelta = 0., minNLO = 1.0e+300;
// 					vector<double> tmpfield(data.GridSize);
// 					for(int j=0;j<data.GridSize;j++){
// 						double LO = 2./data.U*(data.muVec[s]-data.V[s][i]/3.-2.*data.V[s][j]/3.), NLO = pow(data.GradSquared[s][j]/(3.*data.U*data.U),1./3.), dist2 = Norm2(VecDiff(data.VecAt[i],data.VecAt[j])), dist = sqrt(dist2), alpha = 4*dist2*NLO, prefactor = 1./(8.*dist2*NLO*NLO);
// 						if(METHOD<3){//with Bessel, for 1D
// 							if(NLO<minNLO) minNLO = NLO;
// 							if(NLO<ZEROGRAD2AT){
// 								NumNLOthreshold++;
// 								double k7zero = sqrt(POS(LO));
// 								if(j==i) tmpfield[j] = k7zero*k7zero/(2.*PI*gamma);
// 								else tmpfield[j] = k7zero*gsl_sf_bessel_Jn(1,2.*dist*k7zero)/(2.*PI*dist);
// 							}
// 							else{
// 								double x1 = LO/NLO, AIprime = 0.; if(x1<-150.) AIprime = 0.; else if(x1<100.) AIprime = gsl_sf_airy_Ai_deriv(x1,GSL_PREC_DOUBLE);
// 								if(j==i) tmpfield[j] = NLO*(x1*(1.-CurlyA(x1,data))-AIprime)/(2.*PI*gamma);
// 								else{
// 									tmpfield[j] = 0.;
// 									if(METHOD==0 || METHOD==1){//brute-force integration
// 										double x0 = min(-100.,x1-2.*OMEGA*dist*sqrt(NLO))-100.;
// 										double prevres = 1.;
// 										int xFieldSteps = 100, count = 0;
// 										tmpfield[j] = -1.;
// 										while( count < 2 || ABS(ASYM(tmpfield[j],prevres))>data.InternalAcc ){
// 											count++; prevres = tmpfield[j]; xFieldSteps *= 2; if(xFieldSteps>maxFieldSteps) maxFieldSteps = xFieldSteps;
// 											double deltax = (x1-x0)/((double)xFieldSteps); if((x1-x0)>maxdelta) maxdelta = x1-x0;
// 											if(METHOD==0){
// 												vector<double> xFrame = {{x0},{x1}}; vector<double> xField(xFieldSteps+1);
// 												for(int k=0;k<xFieldSteps+1;k++){
// 													double x = x0 + deltax*(double)k;
// 													double k7 = LO-x*NLO;
// 													if(k7>MP){
// 														k7 = sqrt(k7);
// 														double AI;
// 														if(x>100.) AI = 0.;
// 														else if(x>-300.) AI = gsl_sf_airy_Ai(x,GSL_PREC_DOUBLE);
// 														else AI = cos(2.*(-x)*sqrt(-x)/3.-0.25*PI)/(SQRTPI*pow(-x,0.25));
// 														xField[k] = AI*k7*gsl_sf_bessel_Jn(1,2.*dist*k7);
// 													}
// 													else xField[k] = 0.;
// 												}
// 												tmpfield[j] = Integrate(data.ompThreads,data.method, 1, xField, xFrame)/(2.*PI*dist);
// 											}
// 											else if(METHOD==1){
// 												int NumMAXITER = 0;
// 												tmpfield[j] = getn7AIintegral(x0,x1,STEPS,-1.0e-300,0,MAXITER,NumMAXITER,dist,LO,NLO,data);
// 											}
// 											if(xFieldSteps>MAXFIELDSTEPS/2) cout << "i = " << i << ": " << xFieldSteps << " x0=" << x0 << " x1=" << x1 << " tmpfield[" << j << "] = " << tmpfield[j] << " partialSum = " << partialSum << endl;
// 											if( xFieldSteps>MAXFIELDSTEPS /*|| (count>10 && (ABS(tmpfield[j])<0.001*data.AverageDensity[s]/((double)data.GridSize)) )*/ ) break;
// 										}
// 									}
// 									else if(METHOD==2){//integration with adaptive bisection
// 										//double z0 = 1.25*PI;
// 										double x0 = min(-100.,x1-2.*OMEGA*dist*sqrt(NLO))-100.;
// 										double z0 = ceil(2.*dist*sqrt(NLO*(x1-x0))/(2.*PI))*2.*PI+1.25*PI;
// 										double a = x1-z0*z0/alpha, b = x1, prevres;//to put z0 (eventually) onto a zero of the cosine that appears in the asymptotic form of BesselJ
// 										int count = 0, successcount = 0;
// 										while(successcount<3){
// 											count++; prevres = tmpfield[j]; int NumMAXITER = 0;
// 											tmpfield[j] += getn7AIintegral(a,b,STEPS,-1.0e-300,0,MAXITER,NumMAXITER,dist,LO,NLO,data);
// 											z0 += 2.*PI/**(double)count*//**(int)(sqrt((double)count)+0.5)*/; b = a; a = x1-z0*z0/alpha;
// 											if(ABS(ASYM(tmpfield[j],prevres))<TargetAcc || (ABS(tmpfield[j])<TEST && ABS(prevres)<TEST)) successcount++; else successcount = 0;
// 											if(count==MAXCOUNT && ABS(ASYM(tmpfield[j],prevres))>TargetAcc){
// 												cout << "MAXCOUNT insufficient @ i = " << i << ": LO=" << LO << " NLO=" << NLO << " a=" << a << " b=" << b << " Bessel@x0=" << gsl_sf_bessel_Jn(1,2.*dist*sqrt(POS(LO-a*NLO))) << " NumMAXITER(accumulated) = " << NumMAXITER << " successcount = " << successcount << " tmpfield[" << j << "] = " << tmpfield[j] << "(" << prevres << ")" << " asym=" << ABS(ASYM(tmpfield[j],prevres)) << endl;
// 												//tmpfield[j] = 0.;
// 											}
// 											if(NumMAXITER>0){
// 												//cout << "MAXITER insufficient @ i = " << i << ": LO=" << LO << " NLO=" << NLO << " a=" << a << " b=" << b << " Bessel@x0=" << gsl_sf_bessel_Jn(1,2.*dist*sqrt(POS(LO-a*NLO))) << " NumMAXITER(accumulated) = " << NumMAXITER << " successcount = " << successcount << " tmpfield[" << j << "] = " << tmpfield[j] << "(" << prevres << ")" << " asym=" << ABS(ASYM(tmpfield[j],prevres)) << endl;
// 												//tmpfield[j] = 0.; break;
// 											}
// 											if(count==MAXCOUNT) break;
// 										}
// 										//cout << "final tmpfield(i,j)=(" << i << "," << j << ") == " << tmpfield[j] << " count = " << count << "/" << MAXCOUNT << endl;
// 									}
// 								}
// 							}
// 						}
// 					}
// 					data.Den[s][i] = data.degeneracy*Integrate(data.ompThreads,data.method, data.DIM, tmpfield, data.frame);
// 					cout << "density[" << i << "] = " << data.Den[s][i] << endl;
// 				}
// 			}
// 		}
		MatrixToFile(data.TestMat,"TestMat.dat",16);
	}
	else if(METHOD>=3){//with KD
      	double prefactorA = 8./data.U, prefactorB = 4./pow(3.*data.U*data.U,1./3.);
		COMPUTEKD.clear(); COMPUTEKD.resize(data.KD.CoarseGridSize);//points where the density is calculated
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int c=0;c<data.KD.CoarseGridSize;c++){
			int FocalIndex = data.KD.CoarseIndices[c];
			double ABSERR2, AverageRelERR = 0.;
			vector<double> tmpfield(data.GridSize);
            vector<double> rVec(data.VecAt[FocalIndex]);
			for(int j=0;j<data.GridSize;j++){
				COMPUTEKD[c].resize(data.GridSize);//grid over which KD is integrated
				double dist2 = Norm2(VecDiff(rVec,data.VecAt[j]));
				double A = prefactorA*dist2*(data.muVec[s]-data.V[s][FocalIndex]/3.-2.*data.V[s][j]/3.);
				double B = prefactorB*dist2*pow(data.GradSquared[s][j],1./3.);
				if(j!=FocalIndex){
					double KDval;
					if(METHOD==3){
						if(data.KD.UseTriangulation){
							COMPUTEKD[c][j] = GetTriangulatedFuncVal(data.DIM, {A,B}, data.KD.GoodTriangles, data.InternalAcc, 1., data.KD.Vertextree);
							KDval = COMPUTEKD[c][j].ABres[2];
							if(COMPUTEKD[c][j].ABres.size()!=3){ PRINT("Getn7: GetTriangulatedFuncVal error: COMPUTEKD[c][j].ABres.size()!=3",data); usleep(10000000); }
						}
						else KDval = KD(data.DIM,A,B,ABSERR2,data.InternalAcc,KDip);
                        if(!std::isfinite(KDval)){ PRINT("Getn7: KDval error: !std::isfinite(KDval) " + to_string(c) + " " + to_string(j),data); usleep(10000000); }
					}
					else if(METHOD==4){
						KDparams kdparameters = KDparameters;
						GetKD(1,A,B,kdparameters,data,TASK);
						KDval = kdparameters.Result;
					}
					tmpfield[j] = KDval/POW(4.*PI*dist2,data.DIM);
					if(ABS(tmpfield[j])>MP) AverageRelERR += ABSERR2/ABS(tmpfield[j]);
				}
				else tmpfield[j] = GetID(tauThreshold,s,FocalIndex,Norm(rVec),FocalIndex,Norm(rVec),data);
			}
			data.Den[s][FocalIndex] = data.degeneracy*Integrate(data.ompThreads,data.method, data.DIM, tmpfield, data.frame);
 			//PRINT("density[" + to_string(FocalIndex) + "] = " + to_string(data.Den[s][FocalIndex]),data);
		}
	}
  
	data.FocalSpecies = s;
	ExpandSymmetry(1,data.Den[s],data);
	
	if(data.KD.UseTriangulation){
		int NumAccessesGoodTriangles = 0, NumSuccessfulValidations = 0, NumFailedValidations = 0;
		data.KD.NewPoints.clear();
		bool InitPhase = false;
		int NumKDApproximationsUsed = 0;
		for(int i=0;i<data.KD.CoarseGridSize;i++){
			for(int j=0;j<data.GridSize;j++){
				if(/*ToDo: Condition to be removed eventually*/j!=i){
					if(COMPUTEKD[i][j].goodTriangleQ==1) NumAccessesGoodTriangles++;
					else if(COMPUTEKD[i][j].goodTriangleQ==2) NumSuccessfulValidations++;
					else if(COMPUTEKD[i][j].goodTriangleQ==-1) NumFailedValidations++;
					else if(COMPUTEKD[i][j].abserr>=0.){
						if(data.KD.UpdateTriangulation) data.KD.NewPoints.push_back(COMPUTEKD[i][j].ABres);
					}
					else NumKDApproximationsUsed++;//don't store NewPoint if (fast) approximation has been used (ToDo: Or should we store nonetheless?)
				}
			}
		}
		PRINT("Getn7: NumKDApproximationsUsed = " + to_string(NumKDApproximationsUsed) + "/" + to_string(data.KD.CoarseGridSize*data.GridSize),data);
		PRINT("Getn7: NumNewPoints = " + to_string(data.KD.NewPoints.size()),data);
		PRINT("Getn7: NumExistingGoodTriangles = " + to_string(data.KD.GoodTriangles.size()),data);
		PRINT("Getn7: NumAccessesGoodTriangles = " + to_string(NumAccessesGoodTriangles),data);
		PRINT("Getn7: NumSuccessfulValidations = " + to_string(NumSuccessfulValidations) + " (expected: ~ " + to_string((int)(0.01*(double)NumAccessesGoodTriangles)) + ")",data);
		PRINT("Getn7: NumFailedValidations = " + to_string(NumFailedValidations) + " (expected: << " + to_string((int)(0.01*(double)NumAccessesGoodTriangles)) + ")",data);
		UpdateTriangulation(data.DIM, data.KD.NewPoints, data.KD.Vertices, data.KD.tags, data.KD.GoodTriangles, data.KD.MinArea, data.InternalAcc, 1., false, data.KD.NumChecks, data.KD.pointsArray, data.KD.TriangulationReport, data.KD.Vertextree);
        PRINT(data.KD.TriangulationReport,data);
        if(data.KD.IntermediateCleanUp) CleanUpTriangulation(data);
	}
}

void GetSCOdensity(int s, datastruct &data){
	if(data.DensityExpression>=1000 && data.DensityExpression<=1001){
		int iUpper = data.GridSize;
		if(data.Interactions[0]>1002){
			iUpper = OPTSCO.D;
			if(iUpper!=data.GridSize && data.SCcount==0) PRINT("GetSCOdensity: Info !!! OPTSCO.D = " + to_string(OPTSCO.D) + " data.GridSize = " + to_string(data.GridSize),data);
		}
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			double test = data.muVec[s]-data.V[s][i];
			if(data.DensityExpression==1001 && test<0.) test = 0.;
			if(i<iUpper) data.Den[s][i] = test;
			else data.Den[s][i] = 0.;
		}
	}
	//for(int i=0;i<data.GridSize;i++) cout << data.V[s][i] << " " << data.Den[s][i] << endl;
}

void GetTFdensity(int s, int DenExpr, datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  double prefactor;//ToDo
  if(data.DIM==1){
    if(data.Units==1) prefactor = sqrt(2.)*data.degeneracy/PI;
    else{ data.errorCount++; PRINT("GetTFdensity: to be implemented",data); }
  }
  else if(data.DIM==2){
    if(data.Units==1) prefactor = data.degeneracy/(2.*PI);
    else{ data.errorCount++; PRINT("GetTFdensity: to be implemented",data); }
  }
  else if(data.DIM==3){
    if(data.Units==1) prefactor = sqrt(2.)*data.degeneracy/(3.*PI*PI);
    else if(data.Units==2) prefactor = 0.0475412512*sqrt(2.)*data.degeneracy/(3.*PI*PI);
  }
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    //if(data.SymmetryMask[i]){//not expedient for TF
    double test = data.muVec[s]-data.V[s][i];
    if(DenExpr==7){
      if(test>0.){
	if(data.DIM==1) data.Den[s][i] = prefactor*sqrt(test);
	else if(data.DIM==2) data.Den[s][i] = prefactor*test;
	else if(data.DIM==3) data.Den[s][i] = prefactor*test*sqrt(test);
      }
      else data.Den[s][i] = 0.;
    }
    else if(DenExpr==8){//ToDo
      double polylog = -EXP(test/data.TVec[s]);
      if(test/data.TVec[s]>699.) polylog = -4./(3.*SQRTPI)*pow(test/data.TVec[s],1.5); else if(test/data.TVec[s]>-699.) polylog = spline1dcalc(data.PolyLog3half,test/data.TVec[s]);
      if(data.Units==1 && data.DIM==3) data.Den[s][i] = -data.degeneracy*pow(data.TVec[s]/(2.*PI),1.5)*polylog;
      else data.Den[s][i] = 0.;
//       double test2 = test/data.TVec[s];
//       if(ABS(test2)<40. && test2>-40.) data.Den[s][i] = prefactor*data.TVec[s]*log(1.+EXP(test2));
//       else if(test2<=-40. || test2!=test2) data.Den[s][i] = 0.;
//       else data.Den[s][i] = prefactor*test;
    } 
//     }
  }
  //cout << Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame) << endl;
  //data.FocalSpecies = s;
  //ExpandSymmetry(1,data.Den[s],data);
}

double PiotrBeta(int derivative, double x){
  //new
  double x2=x*x, x4 = x2*x2;
  if(derivative==0){//betaQMC(x)
    if(x<0.) return 100000.;
    else if(x>1.55) return -0.4633747986519757/x + 1.443565409987811 - 0.20465388562309392/x2; 
    else return -0.0004147527851895318 + 1.0062109233358525*x - 0.0584541517808854*x2 - 0.3073866986724988*x4 + 0.25331504986348463*x4*x - 0.06139783376450124*x4*x2;
  }
  else{//betaQMCder(x)
    if(x<0.) return 100000.;
    else if(x>1.55) return -0.0011588876670877157 + 0.0663320388441727/x + 1.745208782122636/(x2*x) - 1.25878015302857/x4;
    else{
      return 1 - 0.028100195459653712*x - 0.3963015574364809*x2 - 0.3445575747110073*x2*x - 0.0987982521417292*x4 + 1.2591046941895094*x4*x - 1.2654959902790368*x4*x2 + 0.5086340329349458*x4*x2*x - 0.07525628170004087*x4*x4;  
    }
  }  
//   //old
//   double eta2=eta*eta;
//   if(derivative==0){//beta(eta)
//     if(eta<0.) return 100000.;
//     else if(eta>1.55) return -0.7884382584922027/eta+1.5384587993751975+0.07319471906147554/eta2;
//     else return -0.0004147527851895318+1.0062109233358525*eta-0.0584541517808854*eta2-0.3073866986724988*eta2*eta2+0.25331504986348463*eta2*eta2*eta-0.06139783376450124*eta2*eta2*eta2;
//   }
//   else{//beta'(eta)
//     if(eta<0.) return 0.;
//     else if(eta>1.55) return (1.5384587993751977*(-0.5124857804527525+2.*eta))/eta2-(3.0769175987503954*(0.04757665209572174-0.5124857804527525*eta+eta2))/(eta2*eta);
//     else{
//       double eta4 = eta2*eta2;
//       return 0.9993095646410276-0.0784270387937806*eta-0.09719258795425846*eta2-4.984690223105141*eta4+13.354121636652842*eta4*eta-15.658562412284539*eta4*eta2 +9.84350626945261*eta4*eta2*eta-3.274958222075786*eta4*eta4+0.45726632371788417*eta4*eta4*eta;  
//     }
//   }
}

void ImposeNoise(int fields, datastruct &data){
  bool scheduleQ = false; for(int i=0;i<data.Schedule.size();i++) if(data.Schedule[i]>0) scheduleQ = true;
  if( data.Noise>0. && (data.InterpolVQ==0 || fields==2 || scheduleQ) ){
    //if(data.SCcount==0) PRINT("Impose Noise",data);
    for(int s=0;s<data.S;s++){
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++){
	if(fields==0) data.Env[s][i] *= 1. + data.Noise * data.RN(data.MTGEN);
	else if(fields==1) data.Den[s][i] *= 1. + data.Noise * data.RN(data.MTGEN);
	else if(fields==2) data.Den[s][i] *= 1. + 1.0e-14 * data.RN(data.MTGEN);
      }
    }  
  }
  else if(data.SCcount==0 && data.Print>=0) PRINT("Skip Noise",data);
}

void PreparationsForSCdensity(datastruct &data){
    
  if(data.DensityExpression==1){//handle NaN and inf of data.V
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int s=0;s<data.S;s++){
      for(int i=0;i<data.GridSize;i++){
	if(!std::isfinite(data.V[s][i])){
	  if(data.Print>-1){
		  PRINT("PreparationsForSCdensity: data.V[s=" + to_string(s) + "][i=" + to_string(i) + "] not finite: " + to_string(data.V[s][i]) + " -> data.V[" + to_string(s) + "][" + to_string(i) + "]=0",data);
		data.Print = -1;
	  }
	  data.V[s][i] = 0.;
	}
      }
    }
  }
  
  for(int s=0;s<data.S;s++){//update TVec if default TVec(=-1.) chosen in mpDPFT.input
    if(data.TVec[s]<0.){
      data.TVec[s] = max(ABS(data.muVec[s]-data.VMin[s]),ABS(data.muVec[s]-data.VMax[s]))/50.;
      PRINT("PreparationsForSCdensity: T[s=" + to_string(s) + "] = " + to_string(data.TVec[s]),data);
    }
  }
}

void GetFieldInterpolations(datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  data.FieldInterpolations.resize(data.S);
  int n = data.EdgeLength;
  
  vector<double> XX(n);
  for(int i=0;i<n;i++) XX[i] = -data.edge/2.+(double)i*data.edge/((double)data.steps);
  real_1d_array x;
  x.setcontent(XX.size(), &(XX[0]));
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int s=0;s<data.S;s++){
    int index;
    vector<double> F(n*n);
    data.FieldInterpolations[s].resize(3);
    for(int p=0;p<3;p++){
      for(int i=0;i<n;i++){
	for(int j=0;j<n;j++){
	  if(data.DIM==2) index = i*n+j;
	  else if(data.DIM==3) index = i*n*n+j*n+data.steps/2;//x-y-plane
	  if(p==0) F[i*n+j] = data.Den[s][index];
	  else if(p==1) F[i*n+j] = data.Env[s][index];
	  else if(p==2) F[i*n+j] = data.V[s][index];
      //if(index==data.CentreIndex) cout << "GetFieldInterpolations... " << data.Env[s][index] << endl;
	  if(F[i*n+j]!=F[i*n+j]) cout << "GetFieldInterpolations[p=" << p << "]: " << F[i*n+j] << endl;
	}
      }
      
      spline2dinterpolant spline;
      real_1d_array f;
      f.setcontent(F.size(), &(F[0]));
      //spline2dbuildbicubicv(x, n, x, n, f, 1, spline);//don't use, creates spurious overshootings (cf. Akima spline)
      spline2dbuildbilinearv(x, n, x, n, f, 1, spline);
      
      data.FieldInterpolations[s][p] = spline;
    }
  }
}

void EnforceConstraints(datastruct &data){
  if(data.Print>0) data.PrintSC = 1;
  data.TrappedQ = false;
  bool abortQ = false;
//   int EnforceQ = 0;
//   for(int s=0;s<data.S;s++){
//     double diff0 = ABS(data.TmpAbundances[s]-data.Abundances[s]);
//     if(diff0/ABS(2.*data.Abundances[s]) > data.RelAcc) EnforceQ = 1;
//   }
  if(data.RelAcc>MP && data.TaskType<100){
    //data.PrintSC=1;
    //if(data.SCcount==0 || data.PrintSC==1) PRINT(" ***** enforce constraints ****",data);
    
    data.PreviousmuVec = data.muVec; if(Norm(data.PreviousmuVec)<MP) for(int s=0;s<data.S;s++) data.PreviousmuVec[s] = Sign(data.PreviousmuVec[s])*MP;
    double MinDeltamu = ABS(data.muVec[0]);
    int muVecRangeCount = 0, muVecSearchCount = 0, MAXmuVecRangeCount = 20;
    
    vector<double> muVecMIN = data.muVec;
    vector<double> muVecMAX = data.muVec; 
    vector<double> OldmuVec(data.S); fill(OldmuVec.begin(), OldmuVec.end(), 0.);
    vector<double> Deltamu = VecDiff(data.muVec,OldmuVec);
    vector<bool> muVecMINfound(data.S),muVecMAXfound(data.S); for(int s=0;s<data.S;s++){ muVecMINfound[s] = false; muVecMAXfound[s] = false; }
    //GetDensity(0,data); cout << "GetDensity: " << Integrate(data.ompThreads,data.method, data.DIM, data.Den[0], data.frame) << endl;
    
    //GetTmpAbundances(data); //cout << "GetTmpAbundances: " << Integrate(data.ompThreads,data.method, data.DIM, data.Den[0], data.frame) << endl;
    
    if(data.PrintSC==1) PRINT("    Start with muVec = " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "]",data);
    for(int s=0;s<data.S;s++){
      double PrevioustestAbundance, diff0 = ABS(data.TmpAbundances[s]-data.Abundances[s]);
      if(diff0/ABS(2.*data.Abundances[s]) > data.RelAcc){
	muVecRangeCount = 0;
	PrevioustestAbundance = data.TmpAbundances[s];
	while(data.TmpAbundances[s]>data.Abundances[s] && muVecRangeCount<MAXmuVecRangeCount){
	  muVecMAXfound[s] = true; if(data.muVec[s]<muVecMAX[s]) muVecMAX[s] = data.muVec[s];
	  Deltamu[s] = EXP(data.DeltamuModifier) * (1.+(double)muVecRangeCount) * ABS(data.PreviousmuVec[s]);
	  OldmuVec[s] = data.muVec[s];
	  data.muVec[s] -= Deltamu[s];
	  if(ABS(data.muVec[s])>1.0e+20){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! ABS(mu) likely too big " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); data.muVec[s]=/*1.0e+20*/data.PreviousmuVec[s]; break; /*cout << "sleep..." << endl; usleep(1*sec);*/ }
	  GetDensity(s,data);
	  double testAbundance = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
	  //if(muVecRangeCount==0 || testAbundance<PrevioustestAbundance || data.RelAcc<MP)){
	    data.TmpAbundances[s] = testAbundance;
	    muVecRangeCount++; if(muVecRangeCount>data.S*10000){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! muVecRange not found " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); break;}
	    muVecSearchCount++;
	    if(muVecRangeCount==1){ double diff1 = ABS(data.TmpAbundances[s]-data.Abundances[s]), sgn01 = Sign(diff1-diff0); data.DeltamuModifier -= sgn01*0.1*ABS(data.DeltamuModifier); }
	    muVecMIN[s] = data.muVec[s];
	    if(data.PrintSC==1) PRINT("   Find muVecMIN[s=" + to_string(s) + "]: " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "] ",data);
	    if(data.RelAcc<MP) data.RelAcc = data.RelAccOriginal;
// 	  }
// 	  else{//try again with higher precision
// 	    data.RelAcc *= 0.1;
// 	    data.muVec[s] += Deltamu[s];
// 	    if(data.PrintSC==1) PRINT("   Find muVecMIN[s=" + to_string(s) + "]: " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "] data.RelAcc = " + to_string_with_precision(data.RelAcc,16),data);
// 	    //if(testAbundance>=PrevioustestAbundance) break;
// 	  }
	}
	if(data.TmpAbundances[s]<=data.Abundances[s]) muVecMINfound[s] = true;
	muVecRangeCount = 0;
	if(!muVecMAXfound[s]){
	  PrevioustestAbundance = data.TmpAbundances[s];
	  while(data.TmpAbundances[s]<data.Abundances[s] && muVecRangeCount<MAXmuVecRangeCount){
	    muVecMINfound[s] = true;
	    if(ABS(data.muVec[s])>1.0e+100){ data.errorCount++; PRINT("EnforceConstraints: Error!!! ABS(mu) too big " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); data.muVec[s]=/*1.0e+20*/data.PreviousmuVec[s]; break; }
	    if(data.muVec[s]>muVecMIN[s]) muVecMIN[s] = data.muVec[s];
	    Deltamu[s] = EXP(data.DeltamuModifier) * (1.+(double)muVecRangeCount) * ABS(data.PreviousmuVec[s]);
	    OldmuVec[s] = data.muVec[s];
	    data.muVec[s] += Deltamu[s];
	    if(ABS(data.muVec[s])>1.0e+20){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! ABS(mu) likely too big " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); data.muVec[s]=/*1.0e+20*/data.PreviousmuVec[s]; break; /*cout << "sleep..." << endl; usleep(1*sec);*/ }
	    GetDensity(s,data);
	    double testAbundance = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
	    //if(muVecRangeCount==0 || testAbundance>PrevioustestAbundance || data.RelAcc<MP){
	      data.TmpAbundances[s] = testAbundance;
	      muVecRangeCount++; if(muVecRangeCount>data.S*10000){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! muVecRange not found " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); break;}
	      muVecSearchCount++;
	      if(muVecRangeCount==1){ double diff1 = ABS(data.TmpAbundances[s]-data.Abundances[s]), sgn01 = Sign(diff1-diff0); data.DeltamuModifier -= sgn01*0.1*ABS(data.DeltamuModifier); }
	      muVecMAX[s] = data.muVec[s];
	      if(data.PrintSC==1) PRINT("   Find muVecMAX[s=" + to_string(s) + "]: " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "] ",data);
	      if(data.RelAcc<MP) data.RelAcc = data.RelAccOriginal;
/*	    }
	    else{//try again with higher precision
	      data.RelAcc *= 0.1;
	      data.muVec[s] -= Deltamu[s];
	      if(data.PrintSC==1) PRINT("   Find muVecMAX[s=" + to_string(s) + "]: " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "] data.RelAcc = " + to_string_with_precision(data.RelAcc,16),data);
	      //if(testAbundance<=PrevioustestAbundance) break;
	    }	*/    
	  }
	}
	if(data.TmpAbundances[s]>=data.Abundances[s]) muVecMAXfound[s] = true;
	if(!muVecMAXfound[s] || !muVecMINfound[s]){
	  data.muVec[s] = data.PreviousmuVec[s];
	  abortQ = true;
	  if(data.Print>=0) cout << "mu-ranges not found for s=" << s << " --- reset muVec to PreviousmuVec" << endl;
	  break;
	}
      }
    }
    
    if(!abortQ){
      if(data.PrintSC==1){
	PRINT("   [mu-ranges found] muVecMIN = " + vec_to_str(muVecMIN),data);
	PRINT("                     muVecMAX = " + vec_to_str(muVecMAX),data);
	//PRINT("                                      Deltamu  = " + vec_to_str(Deltamu),data);
      }
    
      //start bisection algorithm with midpoint of muVec-range
      Deltamu = VecDiff(muVecMAX,muVecMIN);
      Deltamu = VecFact(Deltamu,0.5);
      data.muVec = VecSum(muVecMIN,muVecMAX);
      data.muVec = VecFact(data.muVec,0.5);
      vector<vector<double>> history(data.S);
      for(int s=0;s<data.S;s++){
	double diff0 = ABS(data.TmpAbundances[s]-data.Abundances[s]);
	if(diff0/ABS(2.*data.Abundances[s]) > data.RelAccOriginal){//test with data.RelAccOriginal   
	  bool success = false;
	  bool DeltamuTooSmall = false;
	  while(!success && !DeltamuTooSmall){
	    success = true;
	    GetDensity(s,data);//operate with data.RelAcc
	    data.TmpAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
	    if(data.TmpAbundances[s]>0.) history[s].push_back(data.TmpAbundances[s]);
	    int trapcount = 0;
	    if(history[s].size()>5){
	      for(int h=history[s].size()-5;h<history[s].size()-1;h++){
		for(int hp=history[s].size()-5;hp<history[s].size()-1;hp++){
		  if(h>=0 && hp>h && ABS(history[s][h]-history[s][hp])<data.RelAccOriginal*data.Abundances[s]){
		    trapcount++;
		    /*if(!data.FLAGS.ForestGeo) */PRINT(" Bisectional search for s=" + to_string(s) + " are trapped: TmpAbundances [rel.dev.] = " + to_string(data.TmpAbundances[s]) + " [" + to_string_with_precision(100.*(data.TmpAbundances[s]-data.Abundances[s])/data.Abundances[s],2) + "%]. Abort search.",data);
		    data.TrappedQ = true;
		    break;
		  }
		}
		if(trapcount>0) break;
	      }
	    }
	    if(trapcount>0) break;
	    if(data.PrintSC==1) PRINT(" Bisectional search for muVec[s=" + to_string(s) + "]: " + vec_to_str(data.muVec) + " -> TmpAbundances = " + vec_to_str(data.TmpAbundances) + " [Abundances = " + vec_to_str(data.Abundances) + "]",data);
	    for(int sign=-1;sign<2;sign+=2){
	      if((double)sign*(data.TmpAbundances[s]-data.Abundances[s])/ABS(2.*data.Abundances[s]) > data.RelAccOriginal){
		OldmuVec[s] = data.muVec[s];
		data.muVec[s] -= (double)sign * 0.5 * Deltamu[s]; //cout << vec_to_str(data.muVec) << endl;
		if(ABS(data.muVec[s])>1.0e+20){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! ABS(mu) likely too big " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); data.muVec[s]=/*1.0e+20*/data.PreviousmuVec[s]; break; }
		Deltamu[s] = ABS(data.muVec[s] - OldmuVec[s]); if(Deltamu[s]<MinDeltamu) MinDeltamu = Deltamu[s];
		if(ABS(Deltamu[s]) < MP){ data.warningCount++; if(data.Print>=0) PRINT("EnforceConstraints: Warning!!! Deltamu too small " + vec_to_str(Deltamu) + vec_to_str(data.muVec) + vec_to_str(data.TmpAbundances),data); DeltamuTooSmall = true; break; }
		success = false;
		muVecSearchCount++;
	      }
	    }
	  }
	}
      }
    
      if(data.PrintSC==1){
	PRINT("EnforceConstraints: [Success] DeltamuModifier  = " + to_string(data.DeltamuModifier),data);
	PRINT("                             muVecSearchCount  = " + to_string(muVecSearchCount),data);      
	PRINT("                              TmpAbundances    = " + vec_to_str(data.TmpAbundances),data);
	PRINT("                              TargetAbundances = " + vec_to_str(data.Abundances),data);
	if(muVecSearchCount>0) PRINT("                              Minimal Deltamu  = " + to_string_with_precision(MinDeltamu,16),data);
	PRINT("                              Previous muVec   = " + vec_to_str(data.PreviousmuVec),data);
	PRINT("                              Final muVec      = " + vec_to_str(data.muVec),data);
      }        
    
      data.RelAcc = data.RelAccOriginal;

      //check FFTs
      if(data.MonitorFlag){ data.MONITOR = true; GetTmpAbundances(data); data.MONITOR = false; }
    
    }
  }
  else{
    if(data.PrintSC==1) PRINT("EnforceConstraints unnecessary, TmpAbundances = " + vec_to_str(data.TmpAbundances),data);
  }
  
  for(int s=0;s<data.S;s++) if(!std::isfinite(data.muVec[s])) data.muVec[s] = data.PreviousmuVec[s];
  
}

void GetTmpAbundances(datastruct &data){
  for(int s=0;s<data.S;s++){
    if(data.Abundances[s]<MP){
      SetField(data,data.Den[s], data.DIM, data.EdgeLength, 0.);
      data.TmpAbundances[s] = 0.;
    }
    else{
      GetDensity(s,data);
      data.TmpAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
    }
  }
  //UpdateApVec(0,data);
}

void CheckConvergence(datastruct &data){
  StartTimer("CheckConvergence",data);
  double mp = 100.*MachinePrecision;
  if(data.SCcount>0){
    GetFieldStats(0,0,data);
    data.CC = 0.;
    double totalAbundances = 0.;
    for(int s=0;s<data.S;s++) totalAbundances += data.TmpAbundances[s];
    double maxtest = 0.;
    for(int s=0;s<data.S;s++){ double test = RelDev(s,data.Den[s],data.OldDen[s],data)*data.TmpAbundances[s]/totalAbundances; if(test>maxtest){ maxtest = test; data.CC = maxtest; } }
    int steadyQ = 0; if(data.CC<mp && Norm(VecDiff(data.TmpAbundances,data.OldAbundances))<mp && Norm(VecDiff(data.muVec,data.OldmuVec))<mp) steadyQ = 1;
    if( (totalAbundances>mp || (totalAbundances<=mp && (double)data.SCcount*VecAv(data.thetaVec)>1.)) && data.SCcriterion>=0. && (data.CC<data.SCcriterion || steadyQ)){
      data.ExitSCcount++;
//       int test = (int)(-1./log10(1.-VecMin(data.thetaVec)));//(int)(-3./log10(1.-VecMin(data.thetaVec)));
//       if(data.SCcount<test && data.PrintSC==1) PRINT(" Convergence ok, but SCcount still too small w.r.t. thetaVec; SCcount > " + to_string(test) + " required.",data);
//       else data.ExitSCcount++;
    }
    else data.ExitSCcount=0;
    if( data.ExitSCcount>=3 ) data.ExitSCloop = 1;
  }
  else data.CC = 1.2345678901234567890;
  data.CChistory.push_back(data.CC);
  EndTimer("CheckConvergence",data);
}

double RelDev(int s, vector<double> &den1, vector<double> &den2, datastruct &data){
//   double MeanRelDev = 0., threshold = 1.0e-3;
//   if(data.RelAcc>MP) threshold = data.RelAcc;
//   int index, sumcount = 0;
//   vector<double> a(data.HDI.size()), b(data.HDI.size());
//   for(int i=0;i<data.HDI.size();i++){
// //     index = data.HDI[i];
// //     if(ABS(den1[index]+den2[index])>threshold*(data.DenMax[s]-data.DenMin[s])){ MeanRelDev += ABS(ASYM(den1[index],den2[index])); sumcount++; }
//     a[i] = den1[data.HDI[i]];
//     b[i] = den2[data.HDI[i]];
//   }
//   //return MeanRelDev/((double)sumcount);
//   return 1.-2.*VecMult(a,b,data)/(Norm2(a)+Norm2(b));
  
  double a = VecMult(den1,den1,data), b = VecMult(den2,den2,data);
  if(a+b>MP) return 1.-2.*VecMult(den1,den2,data)/(a+b);
  else return 0.;
  
}

void StoreCurrentIteration(datastruct &data){
  StartTimer("StoreCurrentIteration",data);
  data.OldDen = data.Den;
  data.OldV = data.V;
  data.OldmuVec = data.muVec;
  data.OldAbundances = data.TmpAbundances;
  data.OldCC = data.CC;
  if( data.TaskType==61 && TASK.DynDFTe.mode==1 && IntegerElementQ(data.SCcount,TASK.DynDFTe.SnapshotID) ) StoreSnapshotData(data);
  EndTimer("StoreCurrentIteration",data);
}

void UpdateV(datastruct &data){
  StartTimer("UpdateV",data);
  
  if(data.System==2 || data.FLAGS.ForestGeo){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int s=0;s<data.S;s++){
      GetVint(s,data);
      if(data.incrementalV<1.) MultiplyField(data,data.V[s], data.DIM, data.EdgeLength, data.incrementalV);
      AddField(data,data.V[s], data.Env[s], data.DIM, data.EdgeLength);
    }
    //for(int s=0;s<data.S;s++) for(int i=0;i<data.GridSize;i++) if(data.V[s][i]!=data.V[s][i]) cout << "UpdateV: " << s << " " << i << " " << data.V[s][i] << endl;
  }
  else for(int s=0;s<data.S;s++){
    GetVint(s,data);
    if(data.incrementalV<1.) MultiplyField(data,data.V[s], data.DIM, data.EdgeLength, data.incrementalV);
    AddField(data,data.V[s], data.Env[s], data.DIM, data.EdgeLength);
  }
  
  if(data.SCcount>0) data.incrementalV += data.incrementalVOriginal; data.incrementalV = min(1.,data.incrementalV);
  if(data.PrintSC==1) PRINT("UpdateV: incrementalV = " + to_string(data.incrementalV) + "/1.0",data);  
  if(data.PrintSC==1) for(int s=0;s<data.S;s++) PRINT("Species" + to_string(s) + ", central values: InitV = " + to_string(data.Env[s][data.CentreIndex]) + "; PreviousV = " + to_string(data.OldV[s][data.CentreIndex]) + "; CurrentV = " + to_string(data.V[s][data.CentreIndex]),data);
  
  EndTimer("UpdateV",data);
}

void GetVint(int s, datastruct &data){
  SetField(data,data.V[s], data.DIM, data.EdgeLength, 0.);
  for(int a=0;a<data.Interactions.size();a++){
    //cout << "**getVint**" << endl;
    getVint(data.Interactions[a],s,data);
    if(data.System==2 || data.FLAGS.ForestGeo) AddField(data,data.V[s], data.TMPField2[s], data.DIM, data.EdgeLength);
    else AddField(data,data.V[s], data.TmpField2, data.DIM, data.EdgeLength);
    //for(int i=0;i<data.GridSize;i++) if(data.V[s][i]!=data.V[s][i]){ cout << "GetVint: " << s << " " << data.Interactions[a] << " " << i << " " << data.V[s][i] << endl; usleep(1*sec); }
    //cout << "**TmpField2 added**" << endl;
  }
}

void Intercept(datastruct &data){
  StartTimer("Intercept",data);
  //including snapshot, manual break, re-reading of input, waiting for userinput when ExitSCcount==3 (in case calculation should continue with manually changed parameters), etc...
  
  abortQ(data);
  
  time(&data.Timing2); 
  
  if(data.DensityExpression==5 && data.Mpp[2]>0.) AdaptnAiT(data);
  
  if(data.PrintSC==1) PRINT(" ***** SCcount = " + to_string(data.SCcount) + "(" + to_string(data.ExitSCcount) + "/3);  Timing = " + to_string(difftime(data.Timing2,data.Timing1)) + ";  CC = " + to_string_with_precision(data.CC,8/*2+(int)(-log10(data.SCcriterion))*/) + " *****\n\n",data);
  data.Timing1 = data.Timing2;
  int breakCondition = 0; if(data.Print>=0) breakCondition = ManualSCloopBreakQ(data);
  if(breakCondition>0 || data.SCcount==data.maxSCcount) data.ExitSCloop = 1;
  if(data.PrintSC==1 && data.ExitSCcount==3){
    PRINT("\n ***** exit SCloop;         muVec = " + vec_to_str(data.muVec) + " *****",data);
    PRINT(" *****               NVec(Target) = " + vec_to_str(data.Abundances) + " *****",data);
    PRINT(" *****                       NVec = " + vec_to_str(data.TmpAbundances) + " *****",data);
	if(data.DensityExpression==5) data.CalculateEnergyQ = true;
  }
  if(breakCondition==2) TASK.ABORT = true;
  
  if(data.PrintSC==1 && data.CalculateEnergyQ){ GetEnergy(data); PRINT(" Intercept: Etot = " + to_string_with_precision(data.Etot,16) + "\n",data); }
  
  UpdateSymmetryMask(data);
  
  EndTimer("Intercept",data);
}

void UpdateSymmetryMask(datastruct &data){
  if(data.Symmetry==3){
    vector<int> OldSymmetryMask(data.SymmetryMask);
    double OldCalcFraction = 0.; for(int i=0;i<data.GridSize;i++) OldCalcFraction += (double)data.SymmetryMask[i];
    
    if(data.DensityExpression==5 && data.S==1) DelField(data.V[0], 0, data.DelFieldMethod, data);
    else{ data.errorCount++; PRINT("UpdateSymmetryMask: Error!!!",data); }
  
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      if(!data.SymmetryMask[i] && data.RNpos(data.MTGEN)<data.UpdateSymMaskFraction){//Den[s][i] came from interpolation...
	for(int s=0;s<data.S;s++) if(ABS(ASYM(data.Den[s][i],getDenEnAt(0,s,i,data)))>data.InterpolAcc) data.SymmetryMask[i] = 1;//...but the interpolation is insufficient
      }
      else if(data.SymmetryMask[i]){//Den[s][i] came from proper calculation...
	int sufficientQ = 0;
	for(int s=0;s<data.S;s++){
	  double interpol;
	  if(data.DIM==1) interpol = spline1dcalc(data.TmpSpline1D[s],data.VecAt[i][0]);
	  else if(data.DIM==2) interpol = spline2dcalc(data.TmpSpline2D[s],data.VecAt[i][0],data.VecAt[i][1]);
	  else if(data.DIM==3) interpol = spline3dcalc(data.TmpSpline3D[s],data.VecAt[i][0],data.VecAt[i][1],data.VecAt[i][2]);
	  if(ABS(ASYM(data.Den[s][i],interpol))<data.InterpolAcc || ABS(data.Den[s][i])<0.001*data.AverageDensity[s]) sufficientQ++;//...but the interpolation is sufficient for all species...
	}
	if(sufficientQ==data.S) data.SymmetryMask[i] = 0;//...so, use interpolation next time
      }
    }
    
    double NewCalcFraction = 0., NewInterpolr = 0., NewCalcr = 0.; int NewCalcPoints = 0, NewInterpolPoints = 0;
    for(int i=0;i<data.GridSize;i++){
      NewCalcFraction += (double)data.SymmetryMask[i];
      if(OldSymmetryMask[i]==1 && data.SymmetryMask[i]==0){ NewInterpolPoints++; NewInterpolr += Norm(data.VecAt[i]); }
      else if(OldSymmetryMask[i]==0 && data.SymmetryMask[i]==1){ NewCalcPoints++; NewCalcr += Norm(data.VecAt[i]); }
    }
    if(data.PrintSC) PRINT("UpdateSymmetryMask: CalcFraction(Old)=" + to_string_with_precision(OldCalcFraction/((double)data.GridSize),2) + " @StrideLevel=" + to_string(data.StrideLevel) + " #NewCalcPoints=" + to_string(NewCalcPoints) + " @<r>=" + to_string(NewCalcr/((double)NewCalcPoints)),data);
    if(data.StrideLevel>1 && NewCalcFraction>OldCalcFraction) data.StrideLevel--;
    //else if(data.StrideLevel<data.MaxStrideLevel && NewCalcFraction<OldCalcFraction) data.StrideLevel++;
    if(data.PrintSC) PRINT("UpdateSymmetryMask: CalcFraction(New)=" + to_string_with_precision(NewCalcFraction/((double)data.GridSize),2) + " @StrideLevel=" + to_string(data.StrideLevel) + " #NewInterpolPoints=" + to_string(NewInterpolPoints) + " @<r>=" + to_string(NewInterpolr/((double)NewInterpolPoints)),data);
    
  }
}

void AdmixDensities(datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
	StartTimer("AdmixDensities",data);
	
	//BEGIN USER INPUT
	double LinAdmix = -1.;//0.8;//negative LinAdmix if NONE of the Pulay steps (except the 1st) should be replaced by linear steps
	double CCcritForPulayActivation = 1.0e+300;//sqrt(data.SCcriterion);//large value (e.g. 1.0e+300) if NONE of the Pulay steps (except the 1st) should be replaced by linear steps
	//END USER INPUT
	
	double thetaVecModifier = 1., RelDiff = Norm(VecDiff(data.Abundances,data.TmpAbundances))/Norm(data.Abundances);
	if(RelDiff>MP) thetaVecModifier = min(1.,pow(ABS(1.-RelDiff),30.));
	if(data.TrappedQ){
		vecFact(data.thetaVec,thetaVecModifier);//lower weight for suboptimal densities
		if(data.PrintSC && !data.FLAGS.ForestGeo) PRINT("AdmixDensities: data.thetaVec -> " + vec_to_str(data.thetaVec),data);
	}
  
	int PulayQ = 0;
	if(data.Pulay.mixer>0 && data.SCcount>0 && data.RNpos(data.MTGEN)>LinAdmix && data.CC<CCcritForPulayActivation) PulayQ = PulayMixer(data);
	
	if(PulayQ>0){
		if(data.PrintSC==1){
			string space = "position";
			if(data.Pulay.MomentumSpaceQ) space = "momentum";
			PRINT("Pulay mixing (in " + space + " space) enabled",data);
		}
		data.PulMix++;
		if(data.Pulay.mixer==1){//COMBINED Pulay mixing of INDIVIDUAL densities for single- and multi-species systems
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){
				for(int s=0;s<data.S;s++){
					data.Den[s][i] = 0.;
					for(int p=0;p<data.Pulay.coeffVec.size();p++){
						data.Den[s][i] += data.Pulay.coeffVec[p] * ( (1.-data.thetaVec[s])*data.Pulay.ninVec[p][s][i] + data.thetaVec[s]*data.Pulay.noutVec[p][s][i] );
					}
				}
			}
		}
		else if(data.Pulay.mixer>1){
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){
				for(int s=0;s<data.S;s++){
					if(data.Pulay.mixer==2){//SEPARATE Pulay mixing of density DIFFERENCES for multi-species systems
						data.Den[s][i] = 0.;
						for(int ss=0;ss<=s;ss++){
							//compute Pulay-mixed density DIFFERENCE m for index ss ...
							double m = 0;
							for(int p=0;p<data.Pulay.CoeffVec[ss].size();p++){
								m += data.Pulay.CoeffVec[ss][p] * ( (1.-data.thetaVec[ss])*data.Pulay.ninVec[p][ss][i] + data.thetaVec[ss]*data.Pulay.noutVec[p][ss][i] );
							}
							//... and add it to Den (A^-1*m==n)
							data.Den[s][i] += m;
						}
					}
					else if(data.Pulay.mixer==3){//SEPARATE Pulay mixing of INDIVIDUAL densities for multi-species systems
						data.Den[s][i] = 0.;
						for(int p=0;p<data.Pulay.CoeffVec[s].size();p++) data.Den[s][i] += data.Pulay.CoeffVec[s][p] * ( (1.-data.thetaVec[s])*data.Pulay.ninVec[p][s][i] + data.thetaVec[s]*data.Pulay.noutVec[p][s][i] );
					}
				}
			}
		}
	}
	else{//linear mixing
		if(data.PrintSC==1) PRINT("Linear mixing enabled",data);
		data.LinMix++;
		//AdaptLinearMixer(data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++) for(int s=0;s<data.S;s++) data.Den[s][i] = data.thetaVec[s]*data.Den[s][i] + (1.-data.thetaVec[s])*data.OldDen[s][i];
	}
  
	if(data.TrappedQ) vecFact(data.thetaVec,1./thetaVecModifier);//restore for next iteration
  
	EndTimer("AdmixDensities",data);
}

void AdmixPotentials(datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int s=0;s<data.S;s++){
    MultiplyField(data,data.OldV[s], data.DIM, data.EdgeLength, 1.-data.thetaVec[s]);
    MultiplyField(data,data.V[s], data.DIM, data.EdgeLength, data.thetaVec[s]);
    AddField(data,data.V[s], data.OldV[s], data.DIM, data.EdgeLength);
    //restore OldV
    MultiplyField(data,data.OldV[s], data.DIM, data.EdgeLength, 1./(1.-data.thetaVec[s]));
  }
}

int PulayMixer(datastruct &data){
	
	data.Pulay.ninVec.push_back(data.OldDen);
	data.Pulay.noutVec.push_back(data.Den);
	int I = data.Pulay.noutVec.size();
	
	if(data.Pulay.mixer==2){//operate on {n(s=1),n(s=2)-n(s=1),n(3)-n(2),...,n(S-1)-n(S)}, i.e., n(s=1) & all density differences
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			for(int s=1;s<data.S;s++){
				data.Pulay.ninVec[I-1][s][i] = data.OldDen[s][i]-data.OldDen[s-1][i];
				data.Pulay.noutVec[I-1][s][i] = data.Den[s][i]-data.Den[s-1][i];
			}
		}
	}
  
	//compute residuals...
	data.Pulay.residualVec.clear(); data.Pulay.residualVec.resize(0);
	data.Pulay.residualVecFFT.clear(); data.Pulay.residualVecFFT.resize(0);
	for(int s=0;s<data.S;s++){
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){//position-space residuals
			data.TmpField1[i] = data.Pulay.noutVec[I-1][s][i] - data.Pulay.ninVec[I-1][s][i];
		}
		if(data.Pulay.MomentumSpaceQ){
			SetComplexField(data, data.ComplexField, data.DIM, data.EdgeLength, 0., 0.);
			CopyRealFieldToComplex(data, data.ComplexField, data.TmpField1, 0, data.DIM, data.EdgeLength);
			fftParallel(FFTW_FORWARD,data);
			data.Pulay.residualVecFFT.push_back(data.ComplexField);//...in momentum space
		}
		else data.Pulay.residualVec.push_back(data.TmpField1);//...in position space
	}
	if(data.Pulay.MomentumSpaceQ) data.Pulay.ResidualVecFFT.push_back(data.Pulay.residualVecFFT);
	else data.Pulay.ResidualVec.push_back(data.Pulay.residualVec);

	
	int shift = 0;
	//AdaptPulayMixer(data);
	if(I>data.Pulay.scope){//take into account only the data.Pulay.scope most recent iterations
		if(data.Pulay.MomentumSpaceQ) data.Pulay.ResidualVecFFT.erase(data.Pulay.ResidualVecFFT.begin());
		else data.Pulay.ResidualVec.erase(data.Pulay.ResidualVec.begin());
		data.Pulay.ninVec.erase(data.Pulay.ninVec.begin());
		data.Pulay.noutVec.erase(data.Pulay.noutVec.begin());
		I--;
		shift = 1;
	}
	
// 	if(I==data.Pulay.scope){//monitor residual norms, ... doesn't seem to do any good.
// 		double TotalCurrentResidualNorm = 0., AverageHistoryResidualNorm = 0.;
// 		for(int s=0;s<data.S;s++){
// 			for(int i=0;i<I-1;i++) AverageHistoryResidualNorm += Norm(data.Pulay.ResidualVec[i][s]) / (double)(I-1);
// 			TotalCurrentResidualNorm += Norm(data.Pulay.ResidualVec[I-1][s]);
// 		}
// 		if(TotalCurrentResidualNorm>AverageHistoryResidualNorm){
// 			PRINT("\n PulayMixer: Warning !!! ResidualNorm = " + to_string(TotalCurrentResidualNorm) + " is stalling (previous average = " + to_string(AverageHistoryResidualNorm) + ")",data);
// 			//PRINT("\n -> switch to linear mixing",data);
// 			//return 0;
// 		}
// 	}
  
	ae_int_t success = 1;
	MatrixXd ResMat(I,I);
  
	if(data.Pulay.mixer==1){
		//compute residual matrix
		vector<vector<double>> ResidualMatrixCopy(data.Pulay.ResidualMatrix);
		data.Pulay.coeffVec.clear(); data.Pulay.coeffVec.resize(I);
		data.Pulay.ResidualMatrix.clear();
		data.Pulay.ResidualMatrix.resize(I);
		//cout << "ResMat" << endl;
		for(int i=0;i<I;i++){
			data.Pulay.ResidualMatrix[i].resize(I);
			for(int j=0;j<I;j++){
				if(i==I-1 || j==I-1){
					data.Pulay.ResidualMatrix[i][j] = 0.;
					for(int s=0;s<data.S;s++) data.Pulay.ResidualMatrix[i][j] += PulayScalarProduct(s,i,j,data);
				}
				else data.Pulay.ResidualMatrix[i][j] = ResidualMatrixCopy[i+shift][j+shift];
				ResMat(i,j) = data.Pulay.ResidualMatrix[i][j];
				//cout << ResMat(i,j) << " ";
			}
			//cout << endl;
		}
		//compute mixing coefficents
		FullPivHouseholderQR<MatrixXd> qr(ResMat);
		if(qr.isInvertible()){
			MatrixXd ResMat_inv = qr.inverse();
			MatrixXd TestMat = ResMat_inv * ResMat - MatrixXd::Identity(I,I);
			double TestVal = TestMat.squaredNorm(), threshold = 1.0e-8;
			//if(data.Pulay.EagerExecutionQ) threshold = 1.0e-3;
			if(TestVal>threshold){
				PRINT("PulayMixer validation of ResMat_inv failed: TestVal = " + to_string_with_precision(TestVal,16) + " (not 0)",data);
				success = 0;
			}
			if(success>0){
				VectorXd bVec = VectorXd::Constant(I,1.0);
				VectorXd xVec =  ResMat_inv * bVec;
				double A = 0.;
				for(int i=0;i<I;i++) A += xVec(i);
				for(int i=0;i<I;i++) data.Pulay.coeffVec[i] = xVec(i)/A;
			}
		}
		else success = 0;
// 		MatrixXd ResMat_inv = ResMat.inverse();
// 		MatrixXd TestMat = ResMat_inv * ResMat - MatrixXd::Identity(I,I);
// 		double TestVal = TestMat.squaredNorm();
// 		if(TestVal>1.0e-8){
// 			PRINT("PulayMixer validation of ResMat_inv failed: TestVal = " + to_string_with_precision(TestVal,16) + " (not 0)",data);
// 			success = 0;
// 		}
// 		if(success>0){
// 			VectorXd bVec = VectorXd::Constant(I,1.0);
// 			VectorXd xVec =  ResMat_inv * bVec;
// 			double A = 0.;
// 			for(int i=0;i<I;i++) A += xVec(i);
// 			for(int i=0;i<I;i++) data.Pulay.coeffVec[i] = xVec(i)/A;
// 		}
	}
	else if(data.Pulay.mixer>1){
		data.Pulay.CoeffVec.clear(); data.Pulay.CoeffVec.resize(data.S);
		for(int s=0;s<data.S;s++){
			data.Pulay.CoeffVec[s].resize(I);
			//compute residual matrix
			vector<vector<double>> ResidualMatrixCopy(data.Pulay.ResidualMatrixSep[s]);
			data.Pulay.ResidualMatrixSep[s].clear();
			data.Pulay.ResidualMatrixSep[s].resize(I);
			for(int i=0;i<I;i++){
				data.Pulay.ResidualMatrixSep[s][i].resize(I);
				for(int j=0;j<I;j++){
					if(i==I-1 || j==I-1) data.Pulay.ResidualMatrixSep[s][i][j] = PulayScalarProduct(s,i,j,data);
					else data.Pulay.ResidualMatrixSep[s][i][j] = ResidualMatrixCopy[i+shift][j+shift];
					ResMat(i,j) = data.Pulay.ResidualMatrixSep[s][i][j];
				}
			}
			//compute mixing coefficents
			FullPivHouseholderQR<MatrixXd> qr(ResMat);
			if(qr.isInvertible()){
				MatrixXd ResMat_inv = qr.inverse();
				MatrixXd TestMat = ResMat_inv * ResMat - MatrixXd::Identity(I,I);
				double TestVal = TestMat.squaredNorm();
				if(TestVal>1.0e-8){
					PRINT("PulayMixer validation of ResMat_inv failed: TestVal = " + to_string_with_precision(TestVal,16) + " (not 0)",data);
					success = 0;
				}
				if(success>0){
					VectorXd bVec = VectorXd::Constant(I,1.0);
					VectorXd xVec =  ResMat_inv * bVec;
					double A = 0.;
					for(int i=0;i<I;i++) A += xVec(i);
					for(int i=0;i<I;i++) data.Pulay.CoeffVec[s][i] = xVec(i)/A;
				}
			}
			else success = 0;
		}
	}

	if(success<=0){
		double cond = GetConditionNumber(ResMat);
		PRINT("PulayMixer: success = " + to_string(success) + " ResidualMatrix is ill-conditioned (" + to_string(cond) + ")",data);
	}
	
	return success;
}

void AdaptPulayMixer(datastruct &data){
  double CCrate, RecentCCrate = 0., averageAdmixture = 0.; for(int s=0;s<data.S;s++) averageAdmixture += ABS(data.thetaVec[s])/((double)data.S);
  int lastIndex = data.CChistory.size()-1;
  //PRINT(" AdaptPulayMixer: CChistory = " + vec_to_str(data.CChistory),data);
  if(data.SCcount>(int)(1./averageAdmixture) && data.CChistory.size()>data.Pulay.scope){
    CCrate = ASYM(data.CChistory[lastIndex-1],data.CChistory[lastIndex]);
    for(int p=0;p<data.Pulay.scope-1;p++) RecentCCrate += ASYM(data.CChistory[lastIndex-2-p],data.CChistory[lastIndex-1-p])/((double)(data.Pulay.scope-1));
    if(data.PrintSC==1) PRINT(" AdaptPulayMixer: CCrate = " + to_string(CCrate) + " RecentCCrate = " + to_string(RecentCCrate),data);
    if(CCrate<0. && CCrate<RecentCCrate){
      data.Pulay.scope++;
      //data.Pulay.scope++; if(data.Pulay.weight<200.) data.Pulay.weight += 10.;
      //data.RelAcc *= 0.8;
      if(data.PrintSC==1) PRINT(" AdaptPulayMixer: CCrate sluggish {scope,weight,RelAcc} -> {" + to_string(data.Pulay.scope) + ","  + to_string(data.Pulay.weight) + ","  + to_string(data.RelAcc) + "}",data);
    }
    else{
      if(CCrate>0. && data.Pulay.scope>5) data.Pulay.scope--;
      //if(data.Pulay.weight>50.) data.Pulay.weight -= 10.;
      //if(data.RelAcc<data.RelAccOriginal) data.RelAcc /= 0.8;
      if(data.PrintSC==1) PRINT(" AdaptPulayMixer: CCrate ok       {scope,weight,RelAcc} -> {" + to_string(data.Pulay.scope) + ","  + to_string(data.Pulay.weight) + ","  + to_string(data.RelAcc) + "}",data);     
    }
  }
}

void AdaptLinearMixer(datastruct &data){
  double CCrate = 0., RecentCCrate = 0., averageAdmixture = 0.; for(int s=0;s<data.S;s++) averageAdmixture += ABS(data.thetaVec[s])/((double)data.S);
  int history = data.CChistory.size(), scope = (int)(1./averageAdmixture);
  if(history>2*scope+2){
    for(int p=0;p<scope;p++){
      CCrate += ASYM(data.CChistory[history-2-p],data.CChistory[history-1-p])/((double)scope-1.);
      RecentCCrate += ASYM(data.CChistory[history-3-2*p],data.CChistory[history-2-2*p])/((double)scope-1.);
    }
    if(data.PrintSC==1) PRINT(" AdaptLinearMixer: CCrate = " + to_string(CCrate) + " RecentCCrate = " + to_string(RecentCCrate),data);
    if(CCrate<RecentCCrate){
      for(int s=0;s<data.S;s++){ data.thetaVec[s] *= 0.99; data.thetaVec[s] = min(data.thetaVec[s],0.1); }
      if(data.PrintSC==1) PRINT(" AdaptLinearMixer: CCrate sluggish {thetaVec} -> {" + vec_to_str(data.thetaVec) + "}",data);
    }
    else{
      for(int s=0;s<data.S;s++){ data.thetaVec[s] *= 1.01; data.thetaVec[s] = min(data.thetaVec[s],0.1); }
      if(data.PrintSC==1) PRINT(" AdaptLinearMixer: CCrate ok       {thetaVec} -> {" + vec_to_str(data.thetaVec) + "}",data);     
    }
  }
}

double PulayScalarProduct(int s, int i, int j, datastruct &data){
  if(data.Pulay.MomentumSpaceQ){//use optimized metric
    double ATZERO = data.Pulay.ResidualVecFFT[i][s][data.CentreIndex][0]*data.Pulay.ResidualVecFFT[j][s][data.CentreIndex][0]+data.Pulay.ResidualVecFFT[i][s][data.CentreIndex][1]*data.Pulay.ResidualVecFFT[j][s][data.CentreIndex][1];//ATZERO should be real
    double res = (1.+data.Pulay.weight*AngularFactor(data.DIM)*SQRTPI/2.)*ATZERO;
    for(int l=0;l<data.CentreIndex;l++){
      double q = Norm(data.kVecAt[i]);
      res += (data.Pulay.ResidualVecFFT[i][s][l][0]*data.Pulay.ResidualVecFFT[j][s][l][0]+data.Pulay.ResidualVecFFT[i][s][l][1]*data.Pulay.ResidualVecFFT[j][s][l][1]) *(1+data.Pulay.weight/pow(q,data.DDIM-1.)) - data.Pulay.weight/pow(q,data.DDIM-1.) * EXP(-q*q) * ATZERO;
    }
    for(int l=data.CentreIndex+1;l<data.GridSize;l++){
      double q = Norm(data.kVecAt[i]);
      res += (data.Pulay.ResidualVecFFT[i][s][l][0]*data.Pulay.ResidualVecFFT[j][s][l][0]+data.Pulay.ResidualVecFFT[i][s][l][1]*data.Pulay.ResidualVecFFT[j][s][l][1]) *(1+data.Pulay.weight/pow(q,data.DDIM-1.)) - data.Pulay.weight/pow(q,data.DDIM-1.) * EXP(-q*q) * ATZERO;
    }
    return pow(data.Deltak,data.DDIM)*res;
  }
  else return VecMult(data.Pulay.ResidualVec[i][s],data.Pulay.ResidualVec[j][s],data);//use euclidean metric
}

void InjectSchedule(datastruct &data){
  StartTimer("InjectSchedule",data);
  for(int i=0;i<data.Schedule.size();i++){
    int schedule = data.Schedule[i];
    if(schedule==0){ 
      //no injections
    }
    else if(schedule>0 && schedule<3){
      //adjust muVec
      vector<double> a = VecDiff(data.muVec,data.PreviousmuVec);
      for(int s=0;s<data.S;s++) a[s] *= data.thetaVec[s];
      data.muVec = VecSum(data.PreviousmuVec,a);
      PRINT("                    Scheduled muVec   = " + vec_to_str(data.muVec),data);
    
      if(schedule==1){
	//one-time injection
	if(data.SCcount==100) MultiplyField(data,data.Den[1], data.DIM, data.EdgeLength, 0.1);
      }
      else if(schedule==2){
	if(data.SCcount>0 && data.SCcount%400==0){ data.Environments[1] = 1; GetEnv(1,data); data.Abundances[1] *= 1.5; }
	if(data.SCcount>400 && data.SCcount%400==200){ data.Environments[1] = 4; GetEnv(1,data); data.Abundances[1] *= 2./3.; }
      }
    }
    else if(schedule==3){
      double expon = floor(log10((double)data.SCcount));
      for(int i=1;i<10;i++) if(i*(int)pow(10.,expon)==data.SCcount){ PRINT("InjectSchedule: ImposeNoise on densities",data); ImposeNoise(1,data); }
    }
    else if(schedule==4){
      data.Noise = 0.1*exp(-40.*(double)data.SCcount/((double)data.maxSCcount));
      ImposeNoise(1,data);
      if(data.PrintSC) PRINT("InjectSchedule: ImposeNoise of magnitude " + to_string_with_precision(data.Noise,16),data);
    }
    else if(schedule==5){
      for(int s=0;s<data.S;s++){ if(data.CC<data.OldCC) data.thetaVec[s] *= 1.1; else data.thetaVec[s] *= 0.5; data.thetaVec[s] = max(10./((double)data.maxSCcount),min(0.8,data.thetaVec[s])); }
      if(data.Print>=0 && data.PrintSC) PRINT("InjectSchedule: Update thetaVec -> " + vec_to_str(data.thetaVec),data);
    }
    else if(schedule==6){
      //data.Noise = 0.1*exp(-40.*(double)data.SCcount/((double)data.maxSCcount));
      //ImposeNoise(1,data);
      //if(data.PrintSC) PRINT("InjectSchedule: ImposeNoise of magnitude " + to_string_with_precision(data.Noise,16),data);
      data.gamma = data.gammaOriginal*exp(-40.*(double)data.SCcount/((double)data.maxSCcount));
      if(data.PrintSC) PRINT("InjectSchedule: Update gamma -> " + to_string_with_precision(data.gamma,16),data);
    }  
    else if(schedule==7){//noise
      data.Noise = 0.1*exp(-10.*(double)data.SCcount/((double)data.maxSCcount));
      ImposeNoise(1,data);
      if(data.PrintSC) PRINT("InjectSchedule: ImposeNoise of magnitude " + to_string_with_precision(data.Noise,16),data);
    }
    else if(schedule==8){//Fourier filter
      for(int s=0;s<data.S;s++) FourierFilter(data.Den[s], data);
      if(data.PrintSC) PRINT("InjectSchedule: FourierFilter densities (upper " + to_string(data.FourierFilterPercentage) + "%)",data);
    }
    else if(schedule==9){//incremental increase of thetaVec
      double maxTheta = 0.2;
      for(int s=0;s<data.S;s++) if(data.thetaVec[s]<maxTheta) data.thetaVec[s] = data.thetaVecOriginal[s]+(maxTheta-data.thetaVecOriginal[s])*(double)data.SCcount/((double)data.maxSCcount);
      //if(data.PrintSC) PRINT("increase thetaVec -> " + vec_to_str(data.thetaVec),data);
    }   
    else if(schedule==10){
      //double factor1 = exp(-40.*(double)data.SCcount/((double)data.maxSCcount));
      double factor2 = exp(-10.*(double)data.SCcount/((double)data.maxSCcount));
      double factor3 = exp(-10.*(double)data.SCcount/((double)data.maxSCcount));
      //data.Noise = 0.1*factor1;
      MultiplyField(data,data.TVec, 1, data.TVec.size(), factor2);
      data.gammaH *= factor3;
      //ImposeNoise(1,data);
      if(data.PrintSC){
	//PRINT("InjectSchedule: ImposeNoise of magnitude " + to_string_with_precision(data.Noise,16),data);
	PRINT("InjectSchedule: TVec -> " + vec_to_str(data.TVec),data);
	PRINT("InjectSchedule: gammaH -> " + to_string_with_precision(data.gammaH,16),data);
      }
    }
    else if(schedule==11){//noise on first 10% of maxSCcount
      if( data.NoiseOriginal>MP && (double)data.SCcount<(0.1*(double)data.maxSCcount) ){
		data.Noise = data.NoiseOriginal*exp(-40.*(double)data.SCcount/(0.1*(double)data.maxSCcount));
		ImposeNoise(1,data);
      }
      if(data.PrintSC && (double)data.SCcount<(0.1*(double)data.maxSCcount)) PRINT("InjectSchedule: ImposeNoise of magnitude " + to_string_with_precision(data.Noise,16),data);
    }
    else if(schedule==61 && TASK.DynDFTe.mode==1){
		for(int s=0;s<data.S;s++){
			data.mpp[s] = min(1.,(double)data.SCcount/(0.5*(double)data.maxSCcount));
			GetEnv(s,data);
		}
	}
  }
  EndTimer("InjectSchedule",data);
}

void PrepareNextIteration(taskstruct &task, datastruct &data){//Print SCcount = 1,2,...,9,10,20,...,90,100,200,..
  data.SCcount++; //cout << "PrepareNextIteration" << endl;
  
  if(data.Print==3){
    if( (data.SCcount % 100 == 0) || data.SCcount == 1 || data.SCcount==data.maxSCcount || data.ExitSCcount>0 ) data.PrintSC = 1;
    else data.PrintSC = 0;
  }
  else{
    double expon = floor(log10((double)data.SCcount));
    for(int i=1;i<10;i++){
      if(data.Print>0 || (data.Print >= 0 && ((i*(int)pow(10.,expon)==data.SCcount || data.SCcount%1000==0 || data.SCcount==data.maxSCcount || data.ExitSCcount>0))) ){
	data.PrintSC = 1; break;
      }
      else if(data.Print==0) data.PrintSC = 0;
      else if(data.Print<0) data.PrintSC = -1;
    }
    //cout << data.PrintSC << endl;
  }
  if(!data.ExitSCloop){
    if(data.PrintSC==0){ data.controlfile << "."; cout << "."; cout.flush(); }
    else if(data.PrintSC>0){ data.controlfile << "(SCcount=" << data.SCcount << ")"; cout << "(SCcount=" << data.SCcount << ")" << endl; }
  }
  if(data.ExitSCloop && data.RelAcc>MP && Norm(VecDiff(data.TmpAbundances,data.Abundances))>2.*data.RelAcc*data.AccumulatedAbundances) cout << "PrepareNextIteration: Abundances inaccurate..." << endl;
  //if(data.Print==0 && data.PrintSC==1) data.FrameCount++; else if(data.Print>0) data.FrameCount = data.SCcount;
  //if(data.Print>=0 && (data.ExitSCcount>0 || data.SCcount>data.maxSCcount-2)) PrintIntermediateEnergies(data);
  if( data.MovieQ && data.PrintSC==1 && (data.ExitSCcount==0 || data.ExitSCloop) ) StoreMovieData(data);
  if(TASK.ABORT) task.ABORT = true;
  
	if(task.Type==61 && task.DynDFTe.mode==1){
		for(int s=0;s<data.S;s++) GetGuidingVectorField(s,data,TASK);
		if(data.PrintSC){
			PrintIntermediateEnergies(data);
// 			DFTeAlignment(1,0,data);
// 			PRINT("DFTeAlignment Energy = " + to_string_with_precision(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame),16),data);
// 			EkinDynDFTe(1,0,data);
// 			PRINT(" Kinetic DFTe Energy = " + to_string_with_precision(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame),16),data);
		}		
// 		if(data.SCcount==1){
// 		//if(((data.SCcount % (int)(max(2.,0.25*(double)data.maxSCcount))) == 0)){
// 			MatrixToFile(TASK.DynDFTe.v[0],"mpDPFT_DynDFTe_v.dat",16);
// 			MatrixToFile(TASK.DynDFTe.g[0],"mpDPFT_DynDFTe_g.dat",16);
// 		}
	}
	
	if(data.FLAGS.SCO){//determine learning rate lambda for SCO
		data.lambda = data.mpp[0];//0<lambda<=1: standard gradient descent
		double InitLearningRate = data.mpp[0], LearningRateDecay = data.mpp[1];
		if(data.mpp[0]<0.){//Barzilai-Borwein method
			vector<double> Diff = VecDiff(data.Den[0],data.OldDen[0]);
			double diff = Norm(Diff);
			if(diff>MP){
				GetSCOgradients(data.Den[0],data.TmpField2,data);//new gradients
				GetSCOgradients(data.OldDen[0],data.TmpField3,data);//old gradients
				vector<double> diffgrad = VecDiff(data.TmpField2,data.TmpField3);
				double diffgrad2 = max(Norm2(diffgrad),MP);
				data.lambda = ABS(ScalarProduct(Diff,diffgrad))/diffgrad2;
				//PRINT("PrepareNextIteration (Barzilai-Borwein): data.lambda -> " + to_string_with_precision(data.lambda,16),data);
				data.stall = 0;
			}
			else{
				//PRINT("PrepareNextIteration (Barzilai-Borwein): densities unchanged --- diff = " + to_string_with_precision(diff,16),data);
				data.stall++;
				if( data.stall==max(100,(int)(0.01*(double)data.maxSCcount)) ){
					PRINT("PrepareNextIteration (Barzilai-Borwein): Warning !!! SC-loop is stalling  ->  exit",data);
					data.ExitSCloop = 1;
				}
				data.lambda = 1.;
			}
		}
		else if(LearningRateDecay>0.) data.lambda = InitLearningRate*EXP(-LearningRateDecay*(double)data.SCcount/(double)data.maxSCcount);//decay
	}
  
}

void PrintIntermediateEnergies(datastruct &data){
  GetEnergy(data);
  for(int s=0;s<data.S;s++) PRINT("AuxilliaryEkin[s=" + to_string(s) + "] = " + vec_to_str(data.AuxEkin[s]),data);
  for(int s=0;s<data.S;s++) PRINT("AuxilliaryEint[s=" + to_string(s) + "] = " + vec_to_str(data.AuxEint[s]),data);
  PRINT("DispersalEnergies = " + vec_to_str(data.DispersalEnergies),data);
  PRINT("AlternativeDispersalEnergies = " + vec_to_str(data.AlternativeDispersalEnergies),data);
  PRINT("EnvironmentalEnergies = " + vec_to_str(data.EnvironmentalEnergies),data);
  PRINT("InteractionEnergies = " + vec_to_str(data.InteractionEnergies),data);
  if(data.S==1 && data.Environments[0]==10)  PRINT("NucleiEnergy = " + to_string(data.NucleiEnergy),data);
  PRINT("TotalEnergy = " + to_string_with_precision(data.Etot,16),data);
  
	data.IntermediateEnergies << data.SCcount;
	for(int s=0;s<data.S;s++){
		data.IntermediateEnergies << " " << data.DispersalEnergies[s];
		data.IntermediateEnergies << " " << data.EnvironmentalEnergies[s];
	}
	for(int i=0;i<data.Interactions.size();i++) data.IntermediateEnergies << " " << data.InteractionEnergies[i];
	data.IntermediateEnergies << " " << data.Etot << "\n";  
}




void GetEnergy(datastruct &data){
  	data.AuxEkin.clear(); data.AuxEint.clear(); data.DispersalEnergies.clear(); data.AlternativeDispersalEnergies.clear(); data.EnvironmentalEnergies.clear(); data.InteractionEnergies.clear(); data.Etot = 0.; data.EInt = 0.;
  	data.DispersalEnergies.resize(data.S); data.AlternativeDispersalEnergies.resize(data.S); data.EnvironmentalEnergies.resize(data.S); data.InteractionEnergies.resize(data.Interactions.size());
  	if(data.TaskType==100 && TASK.ex.System>=100) data.Etot = TASK.ex.Etot;
  	else if(data.FLAGS.SCO) data.Etot = GetSCOEnergy(data.Den[0],data);
  	else{
    	//get interaction energies first
    	for(int a=0;a<data.InteractionEnergies.size();a++){
        	data.InteractionEnergies[a] = GetInteractionEnergy(data.Interactions[a],data);
        	data.EInt += data.InteractionEnergies[a];
        }
    	data.Etot += data.EInt;
    	GetAuxEkin(data);
    	GetAuxEint(data);
    	for(int s=0;s<data.S;s++){
    		//get EnvironmentalEnergy first
    		data.EnvironmentalEnergies[s] = GetEnvironmentalEnergy(s,data);
    		data.DispersalEnergies[s] = GetDispersalEnergy(s,data);
    		data.Etot += data.DispersalEnergies[s] + data.EnvironmentalEnergies[s];
    	}
    	if(data.S==1 && data.Environments[0]==10) data.Etot += data.NucleiEnergy;
  	}
}

double GetSCOEnergy(vector<double> field/*pass by copy!*/, datastruct &data){
	double f = 0.;
	
	if(data.Interactions[0]==1000) for(int i=0;i<data.GridSize;i++) f += 0.5 * POW(field[i]-1./((double)(i+1)),2);//quadratic
	else if(data.Interactions[0]==1001) for(int i=0;i<data.GridSize;i++) f += 0.25 * POW(field[i]-1./((double)(i+1)),4);//quartic
	else if(data.Interactions[0]==1002){//inverted Gaussian
		for(int i=0;i<data.GridSize;i++) f += POW(field[i]-1./((double)(i+1)),2);
		f = 1.-EXP(-0.5*f);
	}
	else{
		vector<double> x(OPTSCO.D); for(int d=0;d<OPTSCO.D;d++) x[d] = field[d];//use only the first OPTSCO.D components of field
		if(data.Interactions[0]==1003){//Quadratic Programming Problem 20240502
			for(int i=0;i<OPTSCO.AuxVec.size();i++){
				for(int j=0;j<OPTSCO.AuxVec.size();j++) f += field[i]*OPTSCO.AuxMat[i][j]*field[j];
				f += OPTSCO.AuxVec[i]*field[i];
			}
		}
		else f = GetFuncVal(0,x,OPTSCO.function,OPTSCO);
	}
		
	return f;
}

void GetAuxEkin(datastruct &data){
  data.AuxEkin.resize(data.S);
  for(int s=0;s<data.S;s++){

    if(data.Units>0){
      double prefactor, hbar2overmeVA2 = 1./0.131163;
      if(data.DIM==1) prefactor = PI*PI/(6.*data.degeneracy*data.degeneracy);//corresponding to data.Units==1
      else if(data.DIM==2) prefactor = PI/data.degeneracy;//corresponding to data.Units==1
      else if(data.DIM==3) prefactor = pow(243.*PI*PI*PI*PI/(250.*data.degeneracy*data.degeneracy),1./3.);//corresponding to data.Units==1
      if(data.Units==2) prefactor *= hbar2overmeVA2;
      vector<double> integrand(data.GridSize);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++) integrand[i] = pow(POS(data.Den[s][i]),(data.DDIM+2.)/data.DDIM);   
      data.AuxEkin[s].push_back(prefactor * Integrate(data.ompThreads,data.method,data.DIM,integrand,data.frame));//TF kinetic energy
    }
    else data.AuxEkin[s].push_back(0.);
    
    if(data.DIM==3){
      	double Ekingradcorr2, hbar2overmeVA2 = 1./0.131163;
      	DelField(data.Den[s], s, data.DelFieldMethod, data);
      	vector<double> GradCorr2(data.GridSize);
      	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      	for(int i=0;i<data.GridSize;i++){
			double rho = ABS(data.Den[s][i]);
			if(rho>1.0e-6*data.DenMax[s]) GradCorr2[i] = data.GradSquared[s][i]/rho;
			else GradCorr2[i] = 0.;
      	}
      	Ekingradcorr2 = Integrate(data.ompThreads,data.method,data.DIM,GradCorr2,data.frame)/72.;//corresponding to data.Units==1
      	if(data.Units==2) Ekingradcorr2 *= hbar2overmeVA2;
      	//standard leading-order gradient corrections for 3D:
      	data.AuxEkin[s].push_back(Ekingradcorr2);

      	//total von-Weizsaecker (vW) kinetic energy or leading gradcorr (2nd order term only)
      	double Ekin2;
        if(data.FLAGS.vW){
        	double factor = 0.5;//corresponding to data.Units==1
        	if(data.Units==2) factor *= hbar2overmeVA2;
        	MultiplyField(data,data.SqrtDenGradSquared[s],data.DIM,data.EdgeLength,factor);
      		Ekin2 = Integrate(data.ompThreads,data.method,data.DIM,data.SqrtDenGradSquared[s],data.frame);
        }
        else Ekin2 = 9.*Ekingradcorr2;

      	data.AuxEkin[s].push_back(Ekin2);
    }
    else{ data.AuxEkin[s].push_back(0.); data.AuxEkin[s].push_back(0.); }
    
    if(data.S==1 && data.DIM==3){//kinetic energy based on the virial theorem, Eq. (27) in Levy & Perdew, PRA 32, 2010 (1985), inherits the units of Env[s]  
      DelField(data.Env[s], s, data.DelFieldMethod, data);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++) data.TmpField1[i] = data.Den[s][i]*(data.VecAt[i][0]*data.Dx[s][i]+data.VecAt[i][1]*data.Dy[s][i]+data.VecAt[i][2]*data.Dz[s][i]);
      data.AuxEkin[s].push_back(0.5*(Integrate(data.ompThreads,data.method,data.DIM,data.TmpField1,data.frame) - data.EInt));
    }
    else data.AuxEkin[s].push_back(0.);
  }
  //for(int s=0;s<data.S;s++) cout << "AuxilliaryEkin[s=" << s << "]   = " << vec_to_str(data.AuxEkin[s]) << endl;
}

void GetAuxEint(datastruct &data){
  data.AuxEint.resize(data.S);
  vector<double> integrand(data.GridSize);
  for(int s=0;s<data.S;s++){
    if(data.S==1 && data.DIM==3 && data.Units==2){
      
      //Dirac exchange (density-functional) in TF approximation
      DiracExchange(1,0,data);
      data.AuxEint[s].push_back(Integrate(data.ompThreads,data.method,data.DIM,data.TmpField2,data.frame));
      
      //Dirac exchange (potential-functional) in TF approximation
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
      for(int i=0;i<data.GridSize;i++){
	double test = data.muVec[s]-data.V[s][i];
	if(test>0.){
	  double qtilde2 = 2.*0.131163*test;
	  integrand[i] = qtilde2*qtilde2;
	}
	else integrand[i] = 0.;
      }
      data.AuxEint[s].push_back(-data.gammaH*pow(data.degeneracy,2.)/(16.*pow(PI,5.)) * Integrate(data.ompThreads,data.method,data.DIM,integrand,data.frame));
    
      GombasCorrelation(1,0,data);
      data.AuxEint[s].push_back(Integrate(data.ompThreads,data.method,data.DIM,data.TmpField2,data.frame));
      
      VWNCorrelation(1,0,data);
      data.AuxEint[s].push_back(Integrate(data.ompThreads,data.method,data.DIM,data.TmpField2,data.frame));
    }
    else{ data.AuxEint[s].resize(4); fill(data.AuxEint[s].begin(),data.AuxEint[s].end(),0.); }
  }
  //if(data.AuxEint.size()>0) for(int s=0;s<data.S;s++) cout << "AuxilliaryEint[s=" << s << "]   = " << vec_to_str(data.AuxEint[s]) << endl;
}

double GetDispersalEnergy(int s, datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
  	double DispersalEnergy = 0.;
  	for(int ekin=0;ekin<data.EkinTypes.size();ekin++){
    	if(data.EkinTypes[ekin]==0){//default for DensityExpression, see GetDensity for meaning of DensityExpression
      		double tauPrefactor=0.;
      		int n = data.EdgeLength;
  
      		if(data.DIM==2 && data.DensityExpression<3){
				FieldProductToTmpField1(data.Den[s], data.Den[s], data);
				//tauPrefactor = data.taupVec[s];
				if(data.System==1 || data.System==2 || data.FLAGS.ForestGeo){
	  				FieldProductToTmpField1(data.TmpField1,data.tauVecMatrix[s],data);
	  				DispersalEnergy += Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
				}
				else{
	  				tauPrefactor = data.tauVec[s];
	  				DispersalEnergy += 0.5*tauPrefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
				}
      		}
      		else if(data.DIM==2 && data.DensityExpression==3){
				SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.);
				CopyRealFieldToComplex(data,data.ComplexField, data.Den[s], 0, data.DIM, n);
				fftParallel(FFTW_FORWARD,data);
				double nTildePlusAtCentre = data.ComplexField[data.CentreIndex][0];//should be real
				//double t2 = data.tVec[s]*data.tVec[s], prefactor = 2.*PI*data.tauVec[s]*t2;
				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
				for(int i=0;i<n;i++){
	  				int index;
					double kx = data.k0+(double)i*data.Deltak, ky, k2, factor;
	  				for(int j=0;j<n;j++){
	    				index = i*n+j;
	    				if(index != data.CentreIndex){
	      					ky = data.k0+(double)j*data.Deltak;
	      					k2 = kx*kx + ky*ky;
	      					data.ComplexField[index][0] -= nTildePlusAtCentre * EXP(-k2);
	      					factor = 2.*PI/sqrt(k2);
	      					data.ComplexField[index][0] *= factor;
	      					data.ComplexField[index][1] *= factor;
	    				}
	    				else{//ToDo
	      					data.ComplexField[index][0] = 0.;
	      					data.ComplexField[index][1] = 0.;
	    				}
	    				double r2 = data.VecAt[index][0]*data.VecAt[index][0]+data.VecAt[index][1]*data.VecAt[index][1];
	    				double SingularityCompensation = nTildePlusAtCentre * 0.5 * SQRTPI * EXP(-0.125*r2) * gsl_sf_bessel_I0(0.125*r2);
	    				data.TmpField1[index] = SingularityCompensation;
	  				}
				}
				fftParallel(FFTW_BACKWARD,data);
				CopyComplexFieldToReal(data,data.TmpField2,data.ComplexField,0,data.DIM,n);
				AddField(data,data.TmpField2, data.TmpField1, data.DIM, n);
				FieldProductToTmpField1(data.TmpField2, data.Den[s], data);
				tauPrefactor = data.tauVec[s]*data.tVec[s];
				DispersalEnergy += 0.5*tauPrefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
      		}
      		else if(data.DensityExpression==5){//ToDo: 1D
				double E1_s;
				if(data.DIM==2){//hbar=m=1 explicitly
	  				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	  				for(int i=0;i<data.GridSize;i++){
	    				double U=data.V[s][i]-data.muVec[s], absNabU2 = ABS(data.GradSquared[s][i]), y;
	    				if(absNabU2>MP){
	      					y = 2.*U/pow(absNabU2,1./3.);
	      					double AI = 0., AIP = 0.; if(y<50.){ AI = gsl_sf_airy_Ai(y,GSL_PREC_DOUBLE); AIP = gsl_sf_airy_Ai_deriv(y,GSL_PREC_DOUBLE); }
	      					data.TmpField1[i] = 0.5*data.degeneracy*(data.Laplacian[s][i]*CurlyA(y,data)/(12.*PI) - pow(absNabU2,2./3.)*(y*y*CurlyA(y,data)+AI+y*AIP)/(8.*PI));
	    				}
	    				else data.TmpField1[i] = 0.;
	  				}
	  				E1_s = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
	  				FieldProductToTmpField1(data.V[s], data.Den[s], data); double IntVn = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame), muN = data.muVec[s]*data.TmpAbundances[s];
	  				DispersalEnergy += E1_s-IntVn+muN;
	  				if(data.PrintSC) PRINT("IntVn=" + to_string(IntVn) + " muN=" + to_string(muN) + " DispersalEnergy=" + to_string(DispersalEnergy),data);
				}
				else if(data.DIM==3){
// 	  int count = 0;
// 	  double result, prevresult;
// 	  bool SUCCESS = false;
// 	  vector<double> MppOriginal(data.MppVec[s]); data.MppVec[s][0] = 20.; data.MppVec[s][1] = 20.;
// 	  while(!SUCCESS && count<10){
// 	    if(data.PrintSC) PRINT(" Mpp[0]="  + to_string(data.MppVec[s][0]) + " Mpp[1]="  + to_string(data.MppVec[s][1]),data);
// 	    GetnAiT(1,s,data);
// 	    result = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
// 	    if(count>0 && ABS(ASYM(prevresult,result))<data.RelAcc) SUCCESS = true;
// 	    else{
// 	      data.MppVec[s][0] += 4.; data.MppVec[s][1] += 4.;	 
// 	      prevresult = result;
// 	      count++;
// 	    }
// 	    if(data.PrintSC) PRINT("e1=" + to_string(result) + " ASYM = " + to_string_with_precision(ABS(ASYM(prevresult,result)),16),data);
// 	  }
// 	  for(int i=0;i<3;i++) data.MppVec[s][i] = MppOriginal[i];
// 	  E1_s = result;
// 	  FieldProductToTmpField1(data.V[s], data.Den[s], data); double IntVn = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame), muN = data.muVec[s]*data.TmpAbundances[s];
// 	  data.AlternativeDispersalEnergies[s] = E1_s-IntVn+muN;
// 	  if(data.PrintSC) PRINT("IntVn=" + to_string(IntVn) + " muN=" + to_string(muN) + " AlternativeDispersalEnergy=" + to_string(data.AlternativeDispersalEnergies[s]),data);
	  
	  				//Get EkinAiT03D, our default choice for data.DensityExpression==5
	  				StartTimer("GetEkinAiT03D",data);
	  				DelField(data.V[s], s, data.DelFieldMethod, data);
	  				double alpha; if(data.Units==1) alpha = 1.; else if(data.Units==2) alpha = 1./0.1312342130759153;//alpha=hbar^2/(me*Angstrom^2*eV);
	  				double prefactor = data.degeneracy/(4.*PI*PI), c0 = pow(alpha,1./3.), c1 = pow(alpha,-3./2.), c2 = pow(alpha,-1./2.), EnergyComparison = data.EnvironmentalEnergies[s];
	  
	  				double EkinAiT03D = 0., prevEkinAiT03D = 0.;
	  				bool Success = false;
	  				int Iteration = 0, MAXIteration = data.MaxIteration+6;
	  				while(!Success){
	    				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	    				for(int i=0;i<data.GridSize;i++){
	      					if(data.SymmetryMask[i] || data.Symmetry==3){
								double U = data.V[s][i]-data.muVec[s], a = 0.5*c0*pow(ABS(data.GradSquared[s][i]),1./3.);
								if(a<data.MppVec[s][2]){
		  							if(-U>0.) data.TmpField1[i] = c1*pow(2.*(-U),2.5)/5.-c2*data.Laplacian[s][i]*sqrt(2.*(-U))/6.;
		  							else data.TmpField1[i] = 0.;
								}
								else{
		  							double x1 = max(1.,100.-U/a), previousResult = 1.;
		  							vector<double> auxFrame = {{0.},{x1}};
		  							int iteration = 0, gridsize = 64;
		  							bool success = false;
		  							while(!success){
		    							vector<double> integrand(gridsize);
		    							for(int j=0;j<gridsize;j++){
		      								double x = (double)j*max(1.,100.-U/a)/((double)(gridsize-1)), y = x+U/a, AI;
		      								if(y>100.) AI = 0.; else if(y>-300.) AI = gsl_sf_airy_Ai(y,GSL_PREC_DOUBLE); else AI = cos(2.*(-y)*sqrt(-y)/3.-0.25*PI)/(SQRTPI*pow(-y,0.25));
		      								integrand[j] = AI*(c1*pow(2.*a*x,2.5)/5.-c2*data.Laplacian[s][i]*sqrt(2.*a*x)/6.);
		    							}
		    							data.TmpField1[i] = Integrate(data.ompThreads,data.method, 1, integrand, auxFrame);
		    							if( iteration>0 && ( ASYMcritABS(data.TmpField1[i],previousResult,MP)<MP || ( ABS(ASYM(data.TmpField1[i],previousResult))<data.InternalAcc && NegligibleMagnitudeQ(EnergyComparison,data.area*prefactor*data.TmpField1[i],-log(data.InternalAcc)) ) ) ) success = true;
		    							else if(iteration==data.MaxIteration){
		      								//if(data.PrintSC && ABS(ASYM(data.TmpField1[i],previousResult)>data.InternalAcc)) PRINT("EkinAiT03Dn[" + vec_to_str(data.VecAt[i]) + "]: integrand = " + to_string_with_precision(data.TmpField1[i],16) + "[<- " + to_string_with_precision(previousResult,16) + "]",data);
		      								break;
		    							}
		    							else{ iteration++; gridsize *= 2; previousResult = data.TmpField1[i]; }
		  							}
								}
	      					}
	    				}
	    				ExpandSymmetry(0,data.TmpField1,data);
	    				EkinAiT03D = prefactor*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
	    				if(Iteration>0 && ABS(ASYM(EkinAiT03D,prevEkinAiT03D))<max(data.RelAcc,MP)) Success = true;
	    				else{
	      					if(data.PrintSC) PRINT("EkinAiT03D[MaxIteration=" + to_string(data.MaxIteration) + "/" + to_string(MAXIteration) + "] = " + to_string_with_precision(EkinAiT03D,16) + " with relative accuracy of " + to_string_with_precision(ABS(ASYM(EkinAiT03D,prevEkinAiT03D)),16.),data);
	      					if(data.MaxIteration<MAXIteration){
								Iteration++;
								data.MaxIteration++;
								prevEkinAiT03D = EkinAiT03D;
	      					}
                        	else{
								data.warningCount++;
								PRINT("GetDispersalEnergy: Warning!!! Target accuracy for EkinAiT03D not reached for species " + to_string(s),data);
								break;
	      					}
	    				}
	  				}
	  
					if(data.PrintSC) PRINT("EkinAiT03Dn = " + to_string(EkinAiT03D) + " with relative accuracy of " + to_string_with_precision(ABS(ASYM(EkinAiT03D,prevEkinAiT03D)),16.),data);
	  				EndTimer("GetEkinAiT03D",data);
	  				DispersalEnergy += EkinAiT03D;
				}
	
      		}
      		else if(data.DensityExpression>=6 && data.DensityExpression<=10){
              	if( data.FLAGS.AdiabaticEnergy && (data.DensityExpression==6 || data.DensityExpression==9 || data.DensityExpression==10) ){//adiabatic energy formula, Cangi2011
                  	int D = data.DIM;
                  	double Dfactorial = data.DDIM; if(D==3) Dfactorial = 6.;
                    int count = 0;
                  	#pragma omp parallel for schedule(dynamic) reduction(+: DispersalEnergy)
                  	for(int i=0;i<data.GridSize;i++){
                      	double add = 0.;
                      	for(int j=0;j<data.GridSize;j++){
                        	double a = Norm(VecDiff(data.VecAt[i],data.VecAt[j]));
                        	double b = 0.;
                        	double v = (data.V[s][j]-data.muVec[s]);
                        	if(v<-MP){
                          		b = sqrt(-2.*v);
                          		double tau = 0., arg = 2.*a*b;
                            	//if(a>MP) tau = (gsl_sf_bessel_Jn(D+1,arg)*POW(b,D-1)/POW(a,D+1) - POW(b,D)*gsl_sf_bessel_Jn(D,arg)/POW(a,D)) * v;
                                if(a>MP) tau = (boost::math::cyl_bessel_j(D+1,arg)*POW(b,D-1)/POW(a,D+1) - POW(b,D)*boost::math::cyl_bessel_j(D,arg)/POW(a,D)) * v;//a bit faster than gsl
                            	else tau = POW(b,2*D)*(1./(Dfactorial*(data.DDIM+1.))-1./Dfactorial) * v;
                            	add += tau;
                        	}
                        	if(i%(data.GridSize/100)==0 && i==j){
                              	cout << "DispersalEnergy add @ " << count << "/100: " << add << endl;
                                count++;
                            }
                        }
                        DispersalEnergy += add;
                  	}
                  	DispersalEnergy *= (data.degeneracy/POW(2.*PI,D))*POW(data.Deltax,2*D);
                }
                else{
                  	//TF-functional for nTF (and n3' from 1-RDM)
					#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
					for(int i=0;i<data.GridSize;i++){
	  					double test=data.muVec[s]-data.V[s][i];
	  					if(test>0.) data.TmpField1[i] = pow(test,(data.DDIM+2.)/2.);
	  					else data.TmpField1[i] = 0.;
					}
					DispersalEnergy += data.degeneracy*AngularFactor(data.DIM)*pow(2.,(data.DDIM+2.)/2.)/(pow(2.*PI*sqrt(data.U),data.DDIM)*(2.*data.DDIM+4.))*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
                }
      		}
      		else{//ToDo
				PRINT(".......... GetDispersalEnergy to be implemented for this system ..............",data);
      		}
    	}
    	else if(data.EkinTypes[ekin]>0) DispersalEnergy += data.AuxEkin[s][data.EkinTypes[ekin]-1];
  	}
  
  	cout << "DispersalEnergy = " << DispersalEnergy << endl;
  
  	return DispersalEnergy;
}

double GetE1Ai3Dat(datastruct &data){
  double result,prevresult;
  double count = 1.;
  int AiryZeroMax = 10;
  bool success = false;
  while(!success && count<4.5){
    double xlower = gsl_sf_airy_zero_Ai(AiryZeroMax), xupper = 100.;
    autogkstate s;
    autogkreport rep;
    autogksmooth(xlower, xupper, s);
    autogkintegrate(s, E1Ai3DxIntegrand, &data);
    autogkresults(s, result, rep);
    if(count>1.5){
      if(ABS(ASYM(prevresult,result))<sqrt(max(data.RelAcc,MP))) success = true; 
    }
    AiryZeroMax *= (int)(pow(2.,count)+0.5);
    prevresult = result;
    count += 1.;
    //PRINT("GetE1Ai3Dat[" + to_string(data.Index) + "]: E1 = " + to_string(result) + "(" + to_string(prevresult) + ")",data);
  }
  return result;
}

void E1Ai3DxIntegrand(double x, double xminusa, double bminusx, double &y, void *ptr){
  datastruct *data = (datastruct *) ptr;
  
  double alpha = 1.;//for data.Units==1
  if(data->Units==2) alpha = 1./0.1312342130759153;//alpha=hbar^2/(me*Angstrom^2*eV)
  double prefactor = data->degeneracy/pow((2.*PI*alpha),1.5), U = data->V[data->Species][data->Index]-data->muVec[data->Species], ar = 0.5*pow(alpha,1./3.)*pow(data->GradSquared[data->Species][data->Index],1./3.), tau = 1./(data->TVec[data->Species]), nu = tau*(U-x*ar);
  double AI;
  if(x>-300.) AI = gsl_sf_airy_Ai(x,GSL_PREC_DOUBLE);
  else AI = cos(2.*(-x)*sqrt(-x)/3.-0.25*PI)/(SQRTPI*pow(-x,0.25));
  
  y = prefactor * AI * (spline1dcalc(data->PolyLog5half,-nu)/pow(tau,2.5) - (alpha/12.)*data->Laplacian[data->Species][data->Index]*spline1dcalc(data->PolyLog1half,-nu)/sqrt(tau));
}

double GetEnvironmentalEnergy(int s, datastruct &data){
  FieldProductToTmpField1(data.Env[s], data.Den[s], data);
  return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
}

double GetInteractionEnergy(int InteractionType, datastruct &data){//see getVint for details on InteractionTypes
  if(InteractionType==0) return 0.;
  else if(InteractionType<0){
    SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
    for(int s=0;s<data.S;s++){
      getLIBXC(1, s, InteractionType, data.libxcPolarization, data);
      AddField(data,data.TmpField1, data.TmpField2, data.DIM, data.EdgeLength);
    }
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  }
  else if(InteractionType==1){// sum_{s,sp;sp>s}\int(dr)\beta*n_{s}*n_{sp}
    double Eint = 0.;
    for(int s=0;s<data.S;s++){
      for(int sp=s+1;sp<data.S;sp++){
		FieldProductToTmpField1(data.Den[s], data.Den[sp], data);
		Eint += Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
      }
    }
    return data.beta*Eint;
  }
  else if(data.DIM==2 && InteractionType==2){
    SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
    for(int s=0;s<data.S;s++){
      ResourceIntegrand(1, s, data);
      if(data.System==2 || data.FLAGS.ForestGeo) AddField(data,data.TmpField1, data.TMPField2[s], data.DIM, data.EdgeLength);
      else AddField(data,data.TmpField1, data.TmpField2, data.DIM, data.EdgeLength);
    }    
    //ResourceIntegrand(1, 0, data);
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  }
	else if((InteractionType>=3 && InteractionType<=8) || InteractionType==16 || InteractionType==19 || InteractionType==21){
		SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
			for(int ss=0;ss<data.S;ss++){
				if(InteractionType==8){ data.HartreeType = 0; HartreeRepulsion(1, ss, data); }
				else if(InteractionType==19 || InteractionType==21){
					if(InteractionType==19) data.HartreeType = 1;//part 1 of parasitic intraspecific Hartree-type repulsion, with prefactor data.gamma, can also be used to calculate the energy
					else if(InteractionType==21) data.HartreeType = -1;
					SetField(data,data.TmpField3, data.DIM, data.EdgeLength, 0.);
					for(int i=0;i<data.S;i++){
						if(i!=ss){
	    					data.FocalSpecies = ss;
							HartreeRepulsion(1, i, data);
							AddField(data,data.TmpField3, data.TmpField2, data.DIM, data.EdgeLength);
						}
					}
					#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
					for(int i=0;i<data.GridSize;i++) data.TmpField2[i] = data.Den[ss][i]*data.TmpField3[i];
				}
				else if(data.DIM==2) CompetitionIntegrand(1, InteractionType, ss, data);
				if(data.System==2 || data.FLAGS.ForestGeo) AddField(data,data.TmpField1, data.TMPField2[ss], data.DIM, data.EdgeLength);
				else AddField(data,data.TmpField1, data.TmpField2, data.DIM, data.EdgeLength);
			}
		return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
	}
  else if(data.DIM==2 && InteractionType==9){
    RenormalizedContact(1, 0, data);
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  }
  else if(InteractionType==10){
    FieldProductToTmpField1(data.Den[0], data.Den[0], data);
    return data.beta*Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  }
  else if(data.DIM==3 && data.S==1 && InteractionType==11){
    DiracExchange(1,0,data);
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  }
  else if(data.DIM==3 && data.S==1 && InteractionType==12){
    GombasCorrelation(1,0,data);
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  }
  else if(data.DIM==3 && data.S==1 && InteractionType==13){
    VWNCorrelation(1,0,data);
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  }  
  else if(data.DIM==2 && InteractionType==14){
    SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
    for(int ss=0;ss<data.S;ss++){
      DeltaEkin2D3Dcrossover(1, ss, data);
      AddField(data,data.TmpField1, data.TmpField2, data.DIM, data.EdgeLength);
    }
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame);
  }   
  else if(data.S==1 && data.DIM==3 && InteractionType==15){

  }
  else if(data.DIM==2 && data.S==2 && InteractionType==24){
//     Quasi2Dcontact(1, 0, data);
//     double Eint1 = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
//     Quasi2Dcontact(1, 1, data);
//     double Eint2 = Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);    
//     return Eint1+Eint2;
    Quasi2Dcontact(1, 0, data);//20230217: Count the energy only once, not twice!!!
    return Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
  }
  else if(data.DIM==2 && data.S==1 && InteractionType==61){
	  double Eint = 0.;
	  for(int s=0;s<data.S;s++){
		  DFTeAlignment(1, s, data);
		  Eint += Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
	  }
	  return Eint;
  }
  else if(data.DIM==2 && data.S==1 && InteractionType==62){
	  double Eint = 0.;
	  for(int s=0;s<data.S;s++){
		  EkinDynDFTe(1, s, data);
		  Eint += Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame);
	  }
	  return Eint;
  }  
  
  //default
  return 0.;
}

double GetNucleiEnergy(datastruct &data){
	data.NucleiEnergy = 0.;
    for(int nuc1=0;nuc1<data.NumberOfNuclei;nuc1++){
    	vector<double> RVec1 = {{data.NucleiPositions[0][nuc1]},{data.NucleiPositions[1][nuc1]},{data.NucleiPositions[2][nuc1]}};
    	for(int nuc2=nuc1+1;nuc2<data.NumberOfNuclei;nuc2++){
      		vector<double> RVec2 = {{data.NucleiPositions[0][nuc2]},{data.NucleiPositions[1][nuc2]},{data.NucleiPositions[2][nuc2]}};
      		data.NucleiEnergy += data.ValenceCharges[nuc1]*data.ValenceCharges[nuc2]/Norm(VecDiff(RVec1,RVec2));
    	}
	}
	data.NucleiEnergy *= data.gammaH;
	return data.NucleiEnergy;
}

void FourierFilter(vector<double> &Field, datastruct &data){//omp-parallelized - careful about performance when called from within omp loop!!!
	//PRINT("FourierFilter",data);
	int n = data.EdgeLength;
	data.BottomCutoff = data.steps/2-(int)(floor(0.5*(double)data.steps*(1.-0.01*(double)data.FourierFilterPercentage))+0.5);
	data.TopCutoff = data.steps/2+(int)(ceil(0.5*(double)data.steps*(1.-0.01*(double)data.FourierFilterPercentage))+0.5); 
  
	double normalization = Integrate(data.ompThreads,data.method, data.DIM, Field, data.frame);
	SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.);
	CopyRealFieldToComplex(data,data.ComplexField, Field, 0, data.DIM, n);
	fftParallel(FFTW_FORWARD,data);
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<n;i++){
		if(data.DIM==1){
			if(i<data.BottomCutoff || i>data.TopCutoff){
				data.ComplexField[i][0] = 0.;
				data.ComplexField[i][1] = 0.;
			}
		}
		else{
			for(int j=0;j<n;j++){
				if(data.DIM==2){
					if(i<data.BottomCutoff || i>data.TopCutoff || j<data.BottomCutoff || j>data.TopCutoff){
						data.ComplexField[i*n+j][0] = 0.;
						data.ComplexField[i*n+j][1] = 0.;
					}
				}
				else if(data.DIM==3){
					for(int k=0;k<n;k++){
						if(i<data.BottomCutoff || i>data.TopCutoff || j<data.BottomCutoff || j>data.TopCutoff || k<data.BottomCutoff || k>data.TopCutoff){
							data.ComplexField[i*n*n+j*n+k][0] = 0.;
							data.ComplexField[i*n*n+j*n+k][1] = 0.;
						}	    
					}
				}
			}
		}
	}
	fftParallel(FFTW_BACKWARD,data);
	CopyComplexFieldToReal(data,Field, data.ComplexField, 0, data.DIM, n);
	normalization /= Integrate(data.ompThreads,data.method, data.DIM, Field, data.frame);
	MultiplyField(data,Field, data.DIM, n, normalization);
}

void ExpandSymmetry(int ExpandDensityQ, vector<double> &Field, datastruct &data){
	if(data.Symmetry==1){
		spline1dinterpolant SPLINE;
		vector<double> X(data.HDI.size()), F(data.HDI.size());
		for(int j=0;j<X.size();j++){
			X[j] = Norm(data.VecAt[data.HDI[j]]);
			F[j] = Field[data.HDI[j]];
		}
		real_1d_array x,f; x.setcontent(X.size(), &(X[0])); f.setcontent(F.size(), &(F[0]));
		//spline1dbuildlinear(x, f, SPLINE);
		spline1dbuildcubic(x, f, SPLINE);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++) Field[i] = spline1dcalc(SPLINE,Norm(data.VecAt[i]));
	}
	else if(data.Symmetry==2){
		double FieldAtCentre = Field[data.CentreIndex];
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++) Field[i] = FieldAtCentre;
	}
	else if(data.Symmetry==3 && ExpandDensityQ){
		SplineOnStrideGrid(Field, data);
		//SplineOnMainGrid(Field, data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			if(!data.SymmetryMask[i]){
				if(data.DIM==1) Field[i] = spline1dcalc(data.TmpSpline1D[data.FocalSpecies],data.VecAt[i][0]);
				else if(data.DIM==2) Field[i] = spline2dcalc(data.TmpSpline2D[data.FocalSpecies],data.VecAt[i][0],data.VecAt[i][1]);
				else if(data.DIM==3) Field[i] = spline3dcalc(data.TmpSpline3D[data.FocalSpecies],data.VecAt[i][0],data.VecAt[i][1],data.VecAt[i][2]);
			}
		}
	}
	else if(data.Symmetry==4 && ExpandDensityQ){
		if(data.DIM==1){
			spline1dinterpolant SPLINE;
			vector<double> X(data.KD.CoarseGridSize), F(data.KD.CoarseGridSize);
			for(int j=0;j<X.size();j++){
				X[j] = data.VecAt[data.KD.CoarseIndices[j]][0];
				F[j] = Field[data.KD.CoarseIndices[j]];
			}

			// boost::math::interpolators::barycentric_rational<double> interpolant(X.data(), F.data(), X.size());
			// //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			// for(int i=0;i<data.GridSize;i++) if(!data.SymmetryMask[i]) Field[i] = interpolant(data.VecAt[i][0]);

			real_1d_array x,f; x.setcontent(X.size(), &(X[0])); f.setcontent(F.size(), &(F[0]));
			spline1dbuildlinear(x, f, SPLINE);
			//spline1dbuildcubic(x, f, SPLINE);
			//spline1dbuildmonotone(x, f, SPLINE);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++) if(!data.SymmetryMask[i]) Field[i] = spline1dcalc(SPLINE,data.VecAt[i][0]);
		}
	}
}


// G L O B A L   I N I T I A L I Z A T I O N S

void RetrieveTask(datastruct &data,taskstruct &task){
  if(task.Type==0){/*default, do nothing*/}
  else if(task.Type==1 && task.ParameterCount>1){//rectangular mu grid
    cout << endl << "-------> TASK = " << task.count[0]+1 << "/" << task.ParameterCount << endl;
    task.AxesPoints = (int)(pow((double)task.ParameterCount,1./(double)(data.S))+0.5);
    int p0 = (task.count[0] - task.count[0] % task.AxesPoints) / task.AxesPoints, p1 = task.count[0] % task.AxesPoints;
    double delta = pow(1.1,1./((double)task.AxesPoints))-1.;
    data.muVec[0] *= pow(1.+delta,(double)p0);
    data.muVec[1] *= pow(1.+delta,(double)p1);  
  }
  else if(task.Type==2){//GlobalMinimum{Energy[muVec]}
    data.muVec = task.VEC;
    //if(task.ParameterCount>0) data.maxSCcount = (int)task.Aux;
    data.InterpolVQ = (int)task.Aux;
    data.thetaVec = task.AuxVec;
  }
  else if(task.Type==3) data.Abundances = task.VEC; //GlobalMinimum{Energy[Abundances]}
  else if(task.Type==4 || task.Type==6){
    //fruit flies
    data.tauVec[0] = task.VEC[0];
    data.mpp[0] = task.VEC[1];//prefactor for environment
    data.gammaH = task.VEC[2];
  }
  else if(task.Type==5){
    //fruit flies
    data.mpp[0] = task.VEC[0];//prefactor for environment
    data.gammaH = task.VEC[1];
  }
  else if(task.Type==7){
    //fruit flies
    data.mpp[0] = task.VEC[0];//prefactor for environment
  }
  else if(task.Type==8 || task.Type==88){
    PRINT("successive InterpolV; task.count = " + to_string(task.count[0]),data);
    if(task.count[0]>0){
      data.steps *= (int)(pow(2.,(double)task.count[0])+0.5); PRINT("steps -> " + to_string(data.steps),data);
      if(task.Type==88){
		data.TVec = VecFact(data.TVec,pow(0.5,(double)task.count[0]));
		PRINT("BE AWARE: task.Type==88 with change of temperature: TVec -> " + vec_to_str(data.TVec),data);
      }
      data.InterpolVQ = 1; PRINT("InterpolVQ -> " + to_string(data.InterpolVQ),data);
      if(data.incrementalV<1.){ data.incrementalV = 1.; PRINT("incrementalV -> " + to_string(data.incrementalV),data); }
      PRINT("task.VEC -> " + vec_to_str(task.VEC),data);
      if(data.DensityExpression==5 && task.count[0]==data.TaskParameterCount-1){
		data.DeltamuModifier = task.VEC[data.S + data.MppVec[0].size()*data.S]; PRINT("DeltamuModifier -> " + to_string(data.DeltamuModifier),data);
      }
      data.MppVec.clear(); data.MppVec.resize(data.S); for(int s=0;s<data.S;s++) data.MppVec[s].resize(3);
      for(int s=0;s<data.S;s++) data.muVec[s] = task.VEC[s];
      if(data.DensityExpression==5) for(int s=0;s<data.S;s++) for(int a=0;a<data.MppVec[s].size();a++) data.MppVec[s][a] = task.VEC[data.S+s*data.MppVec[s].size()+a];
    }
  }
  else if(task.Type==9){//GradientDescent towards energy minimum in space of abundances
    data.Abundances = data.GradDesc.x; //cout << "data.Abundances--->" <<  vec_to_str(data.Abundances) << endl;
  }
  else if(task.Type==10 && task.lastQ){//TaskParameterCount repetitions -> then, recompute with InterpolVQ of lowest Etot
    data.InterpolVQ = 2; PRINT("InterpolVQ -> " + to_string(data.InterpolVQ),data);
    data.muVec = task.VEC; PRINT("muVec -> " + vec_to_str(data.muVec),data);
    //data.SCcriterion = 1.0e-14; PRINT("SCcriterion -> " + to_string_with_precision(data.SCcriterion,16),data);
    data.Schedule.resize(1); data.Schedule[0] = 0; PRINT("Schedule -> " + vec_to_str(data.Schedule),data);
    data.Noise = 0.; PRINT("Noise -> " + to_string(data.Noise),data);
    data.maxSCcount = 0; PRINT("maxSCcount -> " + to_string(data.maxSCcount),data);
  }  
  else if(task.Type==44 || task.Type==444){
    vector<double> fitparams(TASK.TotalNumFitParams);
    if(TASK.fitQ==5){//ForestGeo-Equilibrium-Abundances
      data.Abundances = data.VEC;
      if(task.Type==444) data.Abundances = task.abundances;
      fitparams = TASK.VEC;
    }
    else{//ForestGeoFit
      fitparams = data.VEC;
    }
      int NumGlobalParam = TASK.NumGlobalParam, NumEnvParam = TASK.NumEnvParam, NumResParam = data.Mpp.size(), NumIntParam = TASK.NumIntParam, NumberOfFitParametersPerSpecies = NumEnvParam+NumResParam+NumIntParam, NumMu = 0;
      if(data.RelAcc<MP) NumMu = data.S;
      if(NumGlobalParam>0) data.sigma = fitparams[0];
      for(int s=0;s<data.S;s++){
	for(int m=0;m<NumEnvParam;m++) data.mpp[NumEnvParam*s+m] = fitparams[NumGlobalParam+s*NumberOfFitParametersPerSpecies+m];//prefactors for environments
	for(int k=0;k<NumResParam;k++) data.Consumption[k][s] = fitparams[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+k];//nutrient requirements
	for(int m=0;m<NumIntParam;m++) data.MPP[s][m] = fitparams[NumGlobalParam+s*NumberOfFitParametersPerSpecies+NumEnvParam+NumResParam+m];//fitness proxies
      }
      if(NumMu>0) for(int s=0;s<NumMu;s++) data.muVec[s] = fitparams[NumGlobalParam+data.S*NumberOfFitParametersPerSpecies+s];
      else if(TASK.GuessMuVecQ){
	if(TASK.Report[data.SwarmParticle].activeQ){
	  if(task.Type==44) data.InterpolVQ = 3; else data.InterpolVQ = 0;
	  data.muVec = TASK.Report[data.SwarmParticle].muVec;
	}
      }
      for(int s=0;s<data.S;s++){
	if(NumIntParam>0) data.RepMut[s*data.S+s] = data.MPP[s][0];
	if(data.MPP[s].size()>1) for(int sp=s+1;sp<data.S;sp++){
	  if(NumIntParam>1) data.Amensalism[s*data.S+sp] = data.MPP[s][1] - data.MPP[sp][1];
	  if(NumIntParam>2) data.RepMut[s*data.S+sp] = data.MPP[s][2] - data.MPP[sp][2];
	  if(NumIntParam>3) data.Competition[s*data.S+sp] = data.MPP[s][3] - data.MPP[sp][3];
	} 
      }
    
  }
  else if(task.Type==61 && task.DynDFTe.mode==0){
	  for(int a=0;a<data.Interactions.size();a++){
		  if(data.Interactions[a]==61 || data.Interactions[a]==62) data.Interactions[a] = 0;
	  }
  }
}

void SetupSCO(datastruct &data){
	PRINT("SetupSCO...",data);
	SetDefaultOPTparams(OPTSCO);

	OPTSCO.function = data.Interactions[0]-1000;
	OPTSCO.D = 5;//10;//20;//2;//

	//OPTSCO.SearchSpaceMin = -1.; OPTSCO.SearchSpaceMax = 1.;
	
	//NYFunction
	OPTSCO.D = 5;
	OPTSCO.SearchSpaceMin = 0.; OPTSCO.SearchSpaceMax = 2.*PI;
	OPTSCO.AuxParams.clear(); OPTSCO.AuxParams = {{805., 1593., 836., 803., 1705., 851., 841., 1732., 834.}};//ny
	OPTSCO.AuxParams.push_back(accumulate(OPTSCO.AuxParams.begin(),OPTSCO.AuxParams.end(),0.));
	
	if(data.Interactions[0]==1003){//Quadratic Programming Problem 20240502
		// Call the function to find a matching file in the executable's directory
		OPTSCO.AuxMat.clear();
		OPTSCO.AuxVec.clear();
		OPTSCO.AuxMat = vector<vector<double>>(ReadMat(findMatchingFile("TabFunc_QuadraticProgram_A",".dat")));
		OPTSCO.D = OPTSCO.AuxMat.size();
		ReadVec(findMatchingFile("TabFunc_QuadraticProgram_c",".dat"), OPTSCO.AuxVec);
		for(int i=0;i<OPTSCO.D;i++) cout << vec_to_str(OPTSCO.AuxMat[i]) << endl;
		cout << endl << vec_to_str(OPTSCO.AuxVec) << endl << endl;
	}
	
	OPTSCO.SearchSpaceLowerVec.clear(); OPTSCO.SearchSpaceLowerVec.resize(OPTSCO.D);
	OPTSCO.SearchSpaceUpperVec.clear(); OPTSCO.SearchSpaceUpperVec.resize(OPTSCO.D);
}

void SanityChecks(datastruct &data){
  
  //for DensityExpression==2,3
  if(data.DensityExpression==2 || data.DensityExpression==3){
    GetFieldStats(0,0,data);
    for(int s=0;s<data.S;s++){
      double test = max(max(ABS(data.muVec[s]-data.EnvMin[s]),ABS(data.muVec[s]-data.EnvMax[s])),max(ABS(data.muVec[s]-data.VMin[s]),ABS(data.muVec[s]-data.VMax[s])));
      if(data.TVec[s]>MP){  
	double TemperatureCheck = test/data.TVec[s];
	if( TemperatureCheck>500. && (data.DensityExpression==2 || data.DensityExpression==3) ){ data.warningCount++; PRINT("SanityChecks: Warning!!! Choose higher temperature (T>" + to_string(test/500.) + ") for species " + to_string(s),data); }
      }
      else{
	data.TVec[s] = test/50.;
	PRINT("SanityChecks: Default temperature chosen (T =" + to_string(data.TVec[s]) + ") for species " + to_string(s),data);
      }
    }
  }
  //for DensityExpression==3
//   for(int s;s<data.S;s++){
//     double TerritoryCheck = data.tVec[s]*data.tVec[s]*data.k0*data.k0;
//     if(data.DensityExpression==3 && TerritoryCheck>5.){ data.warningCount++; PRINT("SanityChecks: Warning!!! Choose larger area (edge) for territorial species " + to_string(s),data); }
//   }
  
  abortQ(data);
}

void Misc(datastruct &data){
  if(data.System==1 || data.System==2 || data.FLAGS.ForestGeo){
    double area = data.area; if(data.System==35 || data.System==36) area *= 0.5;
    data.AverageResourceSquared.resize(data.K); for(int k=0;k<data.K;k++){ data.AverageResourceSquared[k] = data.AvailableResources[k]/area * data.AvailableResources[k]/area;
    if(data.AverageResourceSquared[k]<MP) PRINT("Misc: Warning!!! AverageResourceSquared[k=" + to_string(k) + "] = " + to_string_with_precision(data.AverageResourceSquared[k],16) + " too small...",data); }
    data.tauVecMatrix.resize(data.S); data.LimitingResource.resize(data.S);
    for(int s=0;s<data.S;s++){
      if(data.tauVec[s]<MP){ data.warningCount++; PRINT("Misc: Warning!!! tau too small => redefine tau->1",data); data.tauVec[s] = 1.; }
      data.tauVecMatrix[s].resize(data.GridSize); data.LimitingResource[s].resize(data.GridSize);
      SetField(data,data.LimitingResource[s], data.DIM, data.EdgeLength, 0.);
      for(int i=0;i<data.EdgeLength;i++){
	for(int j=0;j<data.EdgeLength;j++){
	  int index = i*data.EdgeLength+j;
	  if((data.System==35 || data.System==36) && j>=(data.EdgeLength+1)/2) data.tauVecMatrix[s][index] = 0.;
	  else data.tauVecMatrix[s][index] = 0.5*data.tauVec[s];
	}
      }
    }

    if(data.FLAGS.ForestGeo){
      for(int s=0;s<data.S;s++){
	double heuristic = 0.1;
	ResourceIntegrand(0, s, data);//load tauVecMatrix
	double AverageDensity = data.Abundances[s]/data.area; if(data.System==35 || data.System==36) AverageDensity *= (double)data.EdgeLength/((double)(data.EdgeLength+1)/2.);
	double AvtauVecMatrix = 0.;
	for(int i=0;i<data.GridSize;i++) AvtauVecMatrix += data.tauVecMatrix[s][i];
	AvtauVecMatrix /= (double)data.GridSize; if(data.System==35 || data.System==36) AvtauVecMatrix *= (double)data.EdgeLength/((double)(data.EdgeLength+1)/2.);
 	data.muVec[s] = heuristic*(AverageDensity*2.*AvtauVecMatrix+data.EnvAv[s]); //cout << "Misc: data.muVec[s] = " << data.muVec[s] << endl; usleep(10*sec);
      }
    }
    
  }
  
  //uncomment if needed
//   PRINT("Calculate energies based on imported mpDPFT_Den.dat and mpDPFT_V.dat ",data);
//   ifstream infile;
//   infile.open("mpDPFT_V.dat");
//   string line; int InputSize = 0; double tmp; vector<vector<double>> ImportField;
//   while(getline(infile,line)){
//     istringstream LINE(line);
//     if(!line.empty()){
//       vector<double> Vline;
//       while(LINE>>tmp) Vline.push_back(tmp);
//       ImportField.push_back(Vline);
//       InputSize++;
//     }
//   }
//   infile.close();
//   cout << "Check GridSize = " << data.GridSize << " == " << InputSize << " ?" << endl;
//   if(ImportField[0].size()!=data.DIM+data.S){ data.errorCount++; PRINT("LoadV: Error!!! dimensional mismatch: #lineEntries=" + to_string(ImportField[0].size()),data); }
//   for(int s=0;s<data.S;s++){
//     for(int i=0;i<InputSize;i++){ 
//       if(ImportField[i][data.DIM+s]!=ImportField[i][data.DIM+s]) PRINT("LoadV: Error!!! ImportField for s=" + to_string(s) + " is NAN at i=" + to_string(i),data);
//       data.V[s][i] = ImportField[i][data.DIM+s];
//     }
//   }
//   infile.open("mpDPFT_Den.dat");
//   InputSize = 0; vector<vector<double>> Denimport;
//   while(getline(infile,line)){
//     istringstream LINE(line);
//     if(!line.empty()){
//       vector<double> Vline;
//       while(LINE>>tmp) Vline.push_back(tmp);
//       Denimport.push_back(Vline);
//       InputSize++;
//     }
//   }
//   infile.close();
//   cout << "Check GridSize = " << data.GridSize << " == " << InputSize << " ?" << endl;
//   if(Denimport[0].size()!=data.S){ data.errorCount++; PRINT("LoadDen: Error!!! dimensional mismatch: #lineEntries=" + to_string(Denimport[0].size()),data); }
//   for(int s=0;s<data.S;s++){
//     for(int i=0;i<InputSize;i++){
//       if(Denimport[i][s]!=Denimport[i][s]) PRINT("LoadDen: Error!!! Denimport for s=" + to_string(s) + " is NAN at i=" + to_string(i),data);
//       data.Den[s][i] = Denimport[i][s];
//     }
//     //Regularize(data.regularize,data.Den[s],data.RegularizationThreshold,data);
//     cout << "Check mu = " << to_string_with_precision(data.muVec[s],16) << ", Abundance[" << s << "] = " << Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame) << endl;
//   }
//   GetFieldStats(1,0,data);
//   PrintIntermediateEnergies(data);
  
}

void InitializeGlobalFFTWplans(datastruct &data){//CANNOT BE CALLED FROM DIFFERENT THREADS IN PARALLEL  !!! - only fftw_execute() is thread-safe !!!
  if(data.FLAGS.FFTW){
	  StartTimer("InitializeGlobalFFTWplans",data);
    
    if(!fftw_init_threads()) PRINT("FFTW: Parallelization error !!!!",data);
    fftw_plan_with_nthreads(omp_get_max_threads());
    data.inFFTW = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * data.GridSize);
    data.outFFTW = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * data.GridSize);
    if(data.DIM==1){
      data.FFTWPLANFORWARDPARALLEL = fftw_plan_dft_1d(data.steps, data.inFFTW, data.outFFTW, FFTW_FORWARD, FFTW_MEASURE);
      data.FFTWPLANBACKWARDPARALLEL = fftw_plan_dft_1d(data.steps, data.inFFTW, data.outFFTW, FFTW_BACKWARD, FFTW_MEASURE);
    }
    else if(data.DIM==2){
      data.FFTWPLANFORWARDPARALLEL = fftw_plan_dft_2d(data.steps, data.steps, data.inFFTW, data.outFFTW, FFTW_FORWARD, FFTW_MEASURE);
      data.FFTWPLANBACKWARDPARALLEL = fftw_plan_dft_2d(data.steps, data.steps, data.inFFTW, data.outFFTW, FFTW_BACKWARD, FFTW_MEASURE);
    }
    else if(data.DIM==3){
      data.FFTWPLANFORWARDPARALLEL = fftw_plan_dft_3d(data.steps, data.steps, data.steps, data.inFFTW, data.outFFTW, FFTW_FORWARD, FFTW_MEASURE);
      data.FFTWPLANBACKWARDPARALLEL = fftw_plan_dft_3d(data.steps, data.steps, data.steps, data.inFFTW, data.outFFTW, FFTW_BACKWARD, FFTW_MEASURE);
    }
  }
  EndTimer("InitializeGlobalFFTWplans",data);
}

void AllocateTMPmemory(datastruct &data){
	StartTimer("AllocateTMPmemory",data);
    
  int n = data.EdgeLength;
  vector<vector<double>> complexfield(0); complexfield.resize(data.GridSize);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<n;i++){
    if(data.DIM==1) complexfield[i].resize(2);
    else if(data.DIM>1){
      for(int j=0;j<n;j++){
	if(data.DIM==2)	complexfield[i*n+j].resize(2);
	else if(data.DIM==3){
	  for(int k=0;k<n;k++) complexfield[i*n*n+j*n+k].resize(2);
	}
      }
    }
  }
  data.ComplexField = complexfield;
  data.TmpComplexField = complexfield;
  vector<double> TmpField(data.GridSize); data.TmpField1 = TmpField; data.TmpField2 = TmpField; data.TmpField3 = TmpField;
  bool TMP = false; for(int i=0;i<data.Interactions.size();i++) if((data.Interactions[i]>=3 && data.Interactions[i]<=7) || data.Interactions[i]==16 || data.Interactions[i]==18) TMP = true;
  if(data.K>0 || data.System==2 || data.FLAGS.ForestGeo || TMP){ data.TMPField2.resize(data.S); for(int s=0;s<data.S;s++) data.TMPField2[s].resize(data.GridSize); }
  EndTimer("AllocateTMPmemory",data);
}

void InitializeEnvironments(datastruct &data){
  data.Env.resize(data.S); data.V.resize(data.GridSize); data.EnvMin.resize(data.S); data.EnvMax.resize(data.S); data.VMin.resize(data.S); data.VMax.resize(data.S); data.EnvAv.resize(data.S);
  if(data.FLAGS.Del){
    data.Dx.resize(data.S); data.Dy.resize(data.S); data.Dz.resize(data.S); data.D2x.resize(data.S); data.D2y.resize(data.S); data.D2z.resize(data.S); data.GradSquared.resize(data.S); data.Laplacian.resize(data.S);
    if(data.FLAGS.vW){
      data.SqrtDenDx.resize(data.S);
      data.SqrtDenDy.resize(data.S);
      data.SqrtDenDz.resize(data.S);
      data.SqrtDenGradSquared.resize(data.S);
    }
  }
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int s=0;s<data.S;s++){
    data.Env[s].resize(data.GridSize); data.V[s].resize(data.GridSize);
    if(data.FLAGS.Del){
      data.Dx[s].resize(data.GridSize); data.Dy[s].resize(data.GridSize); data.Dz[s].resize(data.GridSize); data.D2x[s].resize(data.GridSize); data.D2y[s].resize(data.GridSize); data.D2z[s].resize(data.GridSize); data.GradSquared[s].resize(data.GridSize); data.Laplacian[s].resize(data.GridSize);
      if(data.FLAGS.vW){
        data.SqrtDenDx[s].resize(data.GridSize);
        data.SqrtDenDy[s].resize(data.GridSize);
        data.SqrtDenDz[s].resize(data.GridSize);
        data.SqrtDenGradSquared[s].resize(data.GridSize);
      }
    }
  }
  for(int s=0;s<data.S;s++) GetEnv(s,data);
  data.V = data.Env;
  //for(int s=0;s<data.S;s++) for(int i=0;i<data.GridSize;i++) if(data.V[s][i]!=data.V[s][i]) cout << s << " " << i << " " << data.V[s][i] << endl;
}

void AllocateDensityMemory(datastruct &data){
	    StartTimer("AllocateDensityMemory",data);
	
  data.TmpAbundances.resize(data.Abundances.size());
  data.Den.resize(data.S); data.DenMin.resize(data.S); data.DenMax.resize(data.S);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int s=0;s<data.S;s++) data.Den[s].resize(data.GridSize);
  if(data.DensityExpression==5){
    data.nAiTevals.resize(data.S);
    data.OldnAiTEVALS.resize(data.S);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int s=0;s<data.S;s++){
      data.OldnAiTEVALS[s] = 1.;
      data.nAiTevals[s].resize(data.GridSize);
      for(int i=0;i<data.GridSize;i++){
	data.nAiTevals[s][i].resize(6);
	for(int a=0;a<data.nAiTevals[s][i].size();a++) data.nAiTevals[s][i][a] = 0;
      }
    }
  }
  if(data.FLAGS.LIBXC){
    data.LIBXC_rho = new double[data.GridSize];//density, required in Hartree atomic units (Hartree & Bohr)
    data.LIBXC_sigma = new double[data.GridSize];//contracted gradients of the density (that is, (\nabla\rho)^2 in case of unpolarized), required in Hartree atomic units
    data.LIBXC_exc = new double[data.GridSize];//energy per unit particle
    data.LIBXC_vxc = new double[data.GridSize];//first (partial) derivative of the energy per unit volume
    data.LIBXC_vsigma = new double[data.GridSize];//first partial derivative of the energy per unit volume in terms of sigma; irrelevant for DPFT
  }
  EndTimer("AllocateDensityMemory",data);
}

void SetupMovieStorage(datastruct &data){
	StartTimer("SetupMovieStorage",data);
  data.MovieStorage.resize(data.EdgeLength);
  GetCutPositions(data);
  for(int i=0;i<data.EdgeLength;i++) data.MovieStorage[i].push_back(-data.CutLine/2.+(double)i*data.CutLine/((double)data.steps));//column 1
  EndTimer("SetupMovieStorage",data);
}

void ImportV(vector<vector<double>> &ImportField, datastruct &data){//
  ifstream infile;
  string name = "mpDPFT_V.dat"; PRINT("LoadV: Interpolate V from mpDPFT_V.dat",data);
  infile.open(name);
  string line; double tmp;
  while(getline(infile,line)){
    istringstream LINE(line);
    if(!line.empty()){
      vector<double> Vline;
      while(LINE>>tmp) Vline.push_back(tmp);
      ImportField.push_back(Vline);
    }
  }
}

void LoadV(datastruct &data, taskstruct &task){
  if(data.InterpolVQ==1){
  //import V
  vector<vector<double>> ImportField;
  ImportV(ImportField,data);
  int InputSize = ImportField.size(); 

  //interpolate V
  int n = (int)(pow((double)InputSize,1./data.DDIM)+0.5), EL = data.EdgeLength;
  for(int s=0;s<data.S;s++){
    if(data.DIM==1){
      real_1d_array x,f; vector<double> X(n),F(n); spline1dinterpolant spline;
      for(int i=0;i<n;i++){ X[i] = ImportField[i][0]; F[i] = ImportField[i][data.DIM+s]; }
      x.setcontent(n, &(X[0])); f.setcontent(n, &(F[0])); spline1dbuildlinear(x, f, spline);
      for(int l=0;l<EL;l++) data.V[s][l] = spline1dcalc(spline,data.lattice[l]);
    }
    else if(data.DIM==2){
      real_1d_array x,y,f; vector<double> X(n), Y(n), F(n*n); for(int i=0;i<n;i++){ X[i] = ImportField[i*n][0]; Y[i] = ImportField[i][1]; }
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<n;i++) for(int j=0;j<n;j++) F[j*n+i] = ImportField[i*n+j][data.DIM+s];//rearrange ImportField for ALGLIB interpolation
      spline2dinterpolant spline; x.setcontent(n, &(X[0])); y.setcontent(n, &(Y[0])); f.setcontent(n*n, &(F[0]));
      spline2dbuildbilinearv(x, n, y, n, f, 1, spline);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<EL;i++){
	for(int j=0;j<EL;j++) data.V[s][i*EL+j] = spline2dcalc(spline, data.Lattice[i], data.Lattice[EL+j]);
      }
    }
    else if(data.DIM==3){
      real_1d_array x,y,z,f; vector<double> X(n), Y(n), Z(n), F(n*n*n);; for(int i=0;i<n;i++){ X[i] = ImportField[i*n*n][0]; Y[i] = ImportField[i*n][1]; Z[i] = ImportField[i][2]; }
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) F[k*n*n+j*n+i] = ImportField[i*n*n+j*n+k][data.DIM+s];//rearrange ImportField for ALGLIB interpolation
      spline3dinterpolant spline; x.setcontent(n, &(X[0])); y.setcontent(n, &(Y[0])); z.setcontent(n, &(Z[0])); f.setcontent(n*n*n, &(F[0]));
      spline3dbuildtrilinearv(x, n, y, n, z, n, f, 1, spline); /*cout << "test spline: " << spline3dcalc(spline, 1., 1., 1.) << endl;*/
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<EL;i++){
	for(int j=0;j<EL;j++){
	  for(int k=0;k<EL;k++){ data.V[s][i*EL*EL+j*EL+k] = spline3dcalc(spline, data.Lattice[i], data.Lattice[EL+j], data.Lattice[2*EL+k]); /*cout << data.V[s][i*EL*EL+j*EL+k] << endl;*/ }
	}
      }
    }    
  }
  
  //Cross-checks
  if(ImportField[0].size()!=data.DIM+data.S){ data.errorCount++; PRINT("LoadV: Error!!! dimensional mismatch: #lineEntries=" + to_string(ImportField[0].size()),data); }
  double IntegratedImportField = 0.;
  for(int s=0;s<data.S;s++){
    for(int i=0;i<InputSize;i++){ 
      if(ImportField[i][data.DIM+s]!=ImportField[i][data.DIM+s]) PRINT("LoadV: Error!!! ImportField for s=" + to_string(s) + " is NAN at i=" + to_string(i),data);
      IntegratedImportField += ImportField[i][data.DIM+s];
    }
  }
  IntegratedImportField *= pow(2.*ABS(ImportField[0][0]),data.DDIM)/(double)(InputSize);  
  double IntegratedV = 0.; for(int s=0;s<data.S;s++) for(int i=0;i<data.GridSize;i++) IntegratedV += data.V[s][i]; IntegratedV *= pow(data.Deltax,data.DDIM);
  if(ABS(ASYM(IntegratedImportField,IntegratedV))>0.05){ data.warningCount++; PRINT("LoadV: Warning!!!  IntegratedImportField = " + to_string(IntegratedImportField) + " vs IntegratedV = " + to_string(IntegratedV) + "  <--> possibly different data.edge, etc.",data); }
  }
  else if(data.InterpolVQ==2) data.V = task.V;
  else if(data.InterpolVQ==3) data.V = task.Report[data.SwarmParticle].V;
}

void UpdateApVec(int init, datastruct &data){//not used for now
  data.ApVec.resize(data.S); data.taupVec.resize(data.S);
  double AreaElement = data.Deltax*data.Deltax;
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int s=0;s<data.S;s++){
    data.ApVec[s] = 0.;
    if(init){
      if(ABS(data.Wall)>MP) data.ApVec[s] = data.WallArea;
      else data.ApVec[s] = data.area;
    }
    else{
      for(int i=0;i<data.GridSize;i++) data.ApVec[s] += Heaviside(data.Den[s][i]);
      data.ApVec[s] *= AreaElement;
    }
    data.taupVec[s] = data.tauVec[s]*data.ApVec[s];
  }
}

void InitializeRandomNumberGenerator(datastruct &data){
	StartTimer("InitializeRandomNumberGenerator",data);
	
    double MaxInt = (double)(std::numeric_limits<int>::max()), Now = ABS((double)chrono::high_resolution_clock::now().time_since_epoch().count() / 1.0e+10), now = (double)((int)Now), seed = 1.0e+10*(Now-now);
    if(seed<1.0e-6) seed = (double)rand();
    while(seed>=MaxInt) seed /= 2.; while(seed<1.0e+06) seed *= 2.;
    mt19937_64 MTGEN((int)seed);
    uniform_real_distribution<double> RN(-1.0,1.0);
    uniform_real_distribution<double> RNpos(0.,1.0);
    data.MTGEN = MTGEN;
    data.RN = RN;
    data.RNpos = RNpos;
    if(data.TaskType==100){
      TASK.ex.MTGEN = MTGEN;
      TASK.ex.RNpos = RNpos;
    }
    //for(int i=0;i<100;i++) cout << data.RNpos(data.MTGEN) << " ";
    EndTimer("InitializeRandomNumberGenerator",data);
}

void ProcessAuxilliaryData(datastruct &data){
	//ReadCubeFile(data);
}

void ReadCubeFile(datastruct &data){//extract densities along cut from (x-major-ordered) density cube files
	
	//important:
	//rename .cube-files to .dat-files !!!
	// declare proper steps in mpDPFT.input
	
  int Edgelength;
  double Bohr = 0.52917721067e-10, Angstrom = 1.0e-10;
  double c = Angstrom/Bohr, c3 = c*c*c, Degeneracy, CubeEdge, VoxelLength; //"voxel-length"=CubeEdge/EdgeLength
  vector<double> FileDataValues, diag;

  data.Den.resize(1); data.Env.resize(1); data.V.resize(1); data.Den[0].resize(data.GridSize); data.Env[0].resize(data.GridSize); data.V[0].resize(data.GridSize);
  SetField(data,data.Den[0], data.DIM, data.EdgeLength, 0.);
  SetField(data,data.Env[0], data.DIM, data.EdgeLength, 0.);
  SetField(data,data.V[0], data.DIM, data.EdgeLength, 0.);
  
  ifstream infile;
  int OrbitalQ = 0, NumHeaderLines, NumberOfAtoms;
  
  //BEGIN USER INPUT
  
  //densities from wave function file (OrbitalQ = 1):
  
  //OrbitalQ = 1; Degeneracy = 2.; CubeEdge=2.*4.251884/c; Edgelength = 300; VoxelLength=CubeEdge/(double)Edgelength; //in Angstrom;  
  //string FileName = "Mg_b3lyp1s.cube"; NumHeaderLines=8;
  //result: N[Riemann(150)] = 1.986518313348605; N[CubicSpline(1000000)] = 2.040372654270921; N[AkimaSpline(1000000)] = 2.040494072507891;
  //OrbitalQ = 1; Degeneracy = 2.; CubeEdge=2.*2.125942/c; Edgelength = 300; VoxelLength=CubeEdge/(double)Edgelength; //in Angstrom;  
  //string FileName = "Mg_b3lyp1s_2.cube"; NumHeaderLines=8;
  //result: N[Riemann(150)] = 1.999103634460121; N[CubicSpline(1000000)] = 2.005857366062849; N[AkimaSpline(1000000)] = 2.006903638481428;
//   OrbitalQ = 1; Degeneracy = 2.; CubeEdge=2.*4.251884/c; Edgelength = 300; VoxelLength=CubeEdge/(double)Edgelength; //in Angstrom;  
//   string FileName = "Mg_lsda1s.cube"; NumHeaderLines=8;
  //result: N[Riemann(150)] = 1.986535739307527; N[CubicSpline(1000000)] = 2.04032375855057; N[AkimaSpline(1000000)] = 2.040443680342799;
  
  //OrbitalQ = 1; Degeneracy = 2.; CubeEdge=2.*12./c; Edgelength = 289; VoxelLength=CubeEdge/(double)Edgelength; //in Angstrom;   
  //string FileName = "Mg_b3lyp_augccpvqz_mo1_1sOrbital_B3LYP.cube"; NumHeaderLines=7;
  //result: N[Riemann(145)] = 2.059285014484474; N[CubicSpline(1000000)] = 2.271759053920894; N[AkimaSpline(1000000)] = 2.250401993403695;
  //string FileName = "Mg_lsda_augccpvqz_mo1_1sOrbital_LSDA.cube"; NumHeaderLines=7;
  //result: N[Riemann(145)] = 2.060974838513778; N[CubicSpline(1000000)] = 2.273178098160824; N[AkimaSpline(1000000)] = 2.251882502880374;
  

  //densities from density files:
  
	//string FileName = "AlNP256pts.dat"; NumberOfAtoms = 201; CubeEdge=36./*Angstrom*/;
  
	//string FileName = "TabFunc_Al2_valence_den_cube.dat"/*data.steps=127*/; NumberOfAtoms = 2; CubeEdge=18./*Angstrom*/;
	string FileName = "TabFunc_Al2_valence_den_cube_256points.dat"/*data.steps=255*/; NumberOfAtoms = 2; CubeEdge=18./*Angstrom*/;
 
  
  //END USER INPUT

  NumHeaderLines=6+NumberOfAtoms;
  Edgelength = data.steps+1;
  VoxelLength=CubeEdge/(double)Edgelength; //in Angstrom;
  string tmpstring;
  
  cout << "process " << FileName << endl;
  infile.open(FileName);
  string line;
  double tmp, maxtmp=0., crosscheck=0., dV=pow(VoxelLength,3.);
  int row=1, col=0, size = Edgelength*Edgelength*Edgelength, NumberOfEmptyLines=0;
  int wordcount=0, Count=0;
  cout << "cube-file header:" << endl << endl;
  while(row <= NumHeaderLines){ getline(infile,line); cout << line << endl; row++; }
  cout << endl;
  while(getline(infile,line)){
    istringstream LINE(line);
    if(!line.empty()){
      while(LINE>>tmp){
		if(OrbitalQ){ tmp *= tmp; tmp *= Degeneracy; }//1s wave function -> 1s^2 density
		tmp *= c3;//->density in Angstrom^-3
		data.Den[0][wordcount]=tmp;
		wordcount++; if((wordcount % (int)((double)size/100.))==0) cout << "ReadCubeFile: " << wordcount << "/" << size << "..." << endl;
		if(tmp > maxtmp) maxtmp = tmp;
		crosscheck += dV*tmp;
      }
    }
    else NumberOfEmptyLines++;
  }
  cout << "particle number crosscheck: N=" << crosscheck << endl;
  cout << "!!!Ensure: FinalNumberOfValues=" << wordcount << "==" << size << "==" << data.GridSize << "?" << endl;
  cout << "NumberOfEmptyLines=" << NumberOfEmptyLines << "; maximal value=" << maxtmp << endl;
  infile.close();
  
  ofstream ProcessedCubeFile;
  FileName = "mpDPFT_ProcessedCubeFile_" + FileName;
  ProcessedCubeFile.open (FileName.c_str());
  cout << "data.S==" << data.S << "?" << endl;
  vector<vector<vector<double>>> FI = GetFieldInterpolationsAlongCut(data);
  for(int i=0;i<data.EdgeLength;i++) ProcessedCubeFile << std::setprecision(16) << -data.CutLine/2.+(double)i*data.CutLine/((double)data.steps) << " " << FI[0][0][i] << "\n"; 
  ProcessedCubeFile.close();
  cout << FileName << " ...created" << endl;
}

void GetSmoothRandomField(int FromFileQ, double scale, datastruct &data){//for 2D, load into data.SmoothRandomField
  if(FromFileQ){
    //PRINT("Load TabFunc_SmoothRandomField.dat",data);
    
    data.SmoothRandomField.resize(data.GridSize);
    vector<double> tmpfield; double val;
    ifstream infile;
    infile.open("TabFunc_SmoothRandomField.dat");
    string line;
    int lineCount = 0;
    while(getline(infile, line)){
      istringstream iss(line);
      iss >> val; tmpfield.push_back(val);
      lineCount++;
    }
    
    int n = (int)(pow((double)lineCount,1./data.DDIM)+0.5);
    vector<double> lattice(n);
    for(int i=0;i<n;i++) lattice[i] = -0.5*data.edge+data.edge*(double)i/((double)(n-1));
    real_1d_array x,f; x.setcontent(n, &(lattice[0])); f.setcontent(lineCount, &(tmpfield[0]));
    spline2dinterpolant spline; spline2dbuildbilinearv(x, n, x, n, f, 1, spline);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.EdgeLength;i++){//the x-index of f has to run fastest for alglib::spline2dbuildbilinearv !!!
      for(int j=0;j<data.EdgeLength;j++){
	data.SmoothRandomField[j*data.EdgeLength+i] = spline2dcalc(spline, data.Lattice[i], data.Lattice[data.EdgeLength+j]);
      }
    }

  }
  else{
    vector<vector<double>> RP(100); for(int j=0;j<4;j++) RP[j].resize(4); for(int seed=0;seed<100;seed++) for(int j=0;j<4;j++) RP[seed][j] = 0.5 * data.edge * data.RN(data.MTGEN);
    data.SmoothRandomField.resize(data.GridSize);
    SetField(data,data.SmoothRandomField, data.DIM, data.EdgeLength, 0.);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      for(int seed=0;seed<100;seed++) data.SmoothRandomField[i] += RP[seed][0] * EXP(-2.*((data.VecAt[i][0]-RP[seed][1])*(data.VecAt[i][0]-RP[seed][1])+(data.VecAt[i][1]-RP[seed][2])*(data.VecAt[i][1]-RP[seed][2]))/(RP[seed][3]*RP[seed][3]));
    }
    auto range = minmax_element(begin(data.SmoothRandomField),end(data.SmoothRandomField));
    double Scale = *range.second - *range.first;
    SetField(data,data.TmpField1, data.DIM, data.EdgeLength, - *range.first);
    AddField(data,data.SmoothRandomField, data.TmpField1, data.DIM, data.EdgeLength);
    MultiplyField(data,data.SmoothRandomField, data.DIM, data.EdgeLength, scale/Scale);
  }
}

void GetSpecialFunctions(datastruct &data){
	StartTimer("GetSpecialFunctions",data);
  if(data.FLAGS.PolyLogs){
    data.PolyLogm3half = GetSpecialFunction("TabFunc_PolyLogm3half.dat");
    data.PolyLogm1half = GetSpecialFunction("TabFunc_PolyLogm1half.dat");
    data.PolyLog1half = GetSpecialFunction("TabFunc_PolyLog1half.dat");
    data.PolyLog3half = GetSpecialFunction("TabFunc_PolyLog3half.dat");
    data.PolyLog5half = GetSpecialFunction("TabFunc_PolyLog5half.dat");
    data.AuxDawsonErfc = GetSpecialFunction("TabFunc_AuxDawsonErfc.dat");
    //data.PolyLogm3half = GetPolyLogm3half();
    //data.PolyLogm1half = GetPolyLogm1half();
    //data.PolyLog1half = GetPolyLog1half(); 
    //data.PolyLog3half = GetPolyLog3half(); 
    //data.PolyLog5half = GetPolyLog5half();
    //data.AuxDawsonErfc = GetAuxDawsonErfc();
  }
  if(data.FLAGS.CurlyA) data.CurlyA = GetSpecialFunction("TabFunc_CurlyA.dat"); //data.CurlyA = GetCurlyA();
  if(data.FLAGS.Gyk){
    data.Gykspline1D = GetSpline1D("TabFunc_Gyk1D.dat",true);
    data.Gykspline2D = GetSpline1D("TabFunc_Gyk2D.dat",true);
    data.Gykspline3D = GetSpline1D("TabFunc_Gyk3D.dat",true);
  }
  // if(data.FLAGS.KD){
  //   data.HG1F2_a = GetSpline1D("TabFunc_Hypergeometric1F2_a.dat",true);
  //   data.HG1F2_b = GetSpline1D("TabFunc_Hypergeometric1F2_b.dat",true);
  // }
  if(data.TaskType==100 && data.FLAGS.GetOrbitals){
    GetOrb(data);
  }
  if(IntegerElementQ(24,data.Interactions)) Load2D3Deta(data);
  EndTimer("GetSpecialFunctions",data);
}

void InitializeKD(datastruct &data){  
	if(data.KD.UseTriangulation){
		StartTimer("InitializeKD",data);
		data.KD.GoodTrianglesFilename = findMatchingFile("TabFunc_K" + to_string(data.DIM),".dat");
        if(!ExtractIntegerFromPattern(data.KD.GoodTrianglesFilename, data.KD.quality)){
          PRINT("Unable to determine KD Good Triangle Quality -> set data.KD.quality = " + to_string(data.KD.NumChecks),data);
          data.KD.quality = data.KD.NumChecks;//fall-back option if extraction failed
        }
        if(data.KD.ReevaluateTriangulation>0) data.KD.quality = data.KD.ReevaluateTriangulation;//if up- or downgrade of quality requested
        PRINT("KD Good Triangle Quality = " + to_string(data.KD.quality),data);
		vector<vector<double>> RawTriangles = ReadMat(data.KD.GoodTrianglesFilename);
		data.KD.GoodTriangles.clear(); data.KD.GoodTriangles.resize(RawTriangles.size());
		data.KD.Vertices.clear(); data.KD.Vertices.resize(3*RawTriangles.size());
		data.KD.tags.setlength(data.KD.Vertices.size());
        vector<double> TriangleAreas(RawTriangles.size());
		//Reshape RawTriangles into GoodTriangles and into Vertices
		#pragma omp parallel for schedule(static)
		for(int t=0;t<RawTriangles.size();t++){
			vector<vector<double>> T(3);//one triangle
			data.KD.GoodTriangles[t].resize(3);//three points per triangle
			for(int p=0;p<3;p++){//points of triangle
				T[p].resize(3);
				data.KD.GoodTriangles[t][p].resize(3);//three components per point
				for(int c=0;c<3;c++){//components of point
					T[p][c] = RawTriangles[t][p*3+c];
				}
				data.KD.Vertices[t*3+p].resize(3);
				data.KD.Vertices[t*3+p] = T[p];
				data.KD.tags[t*3+p] = t*3+p;
			}
			data.KD.GoodTriangles[t] = T;
            TriangleAreas[t] = TriangleArea(T);
		}
		if(TriangleAreas.size()>0) data.KD.MinArea = SortMedian(TriangleAreas);
        PRINT("KD.MinArea = " + to_string(data.KD.MinArea),data);
		data.KD.OrigNumGoodTriangles = data.KD.GoodTriangles.size();
		PRINT("KD.GoodTriangles (" + to_string(data.KD.OrigNumGoodTriangles) + ") loaded",data);
		PRINT("KD.Vertices (" + to_string(data.KD.Vertices.size()) + ") loaded",data);
		data.KD.NewPoints.clear();
		UpdateTriangulation(data.DIM, data.KD.NewPoints, data.KD.Vertices, data.KD.tags, data.KD.GoodTriangles, data.KD.MinArea, data.InternalAcc, 1., true, data.KD.NumChecks, data.KD.pointsArray, data.KD.TriangulationReport, data.KD.Vertextree);
		PRINT(data.KD.TriangulationReport,data);
		EndTimer("InitializeKD",data);
	}
}

void GetOrb(datastruct &data){
  	TASK.ex.Orb.resize(TASK.ex.settings[0]);
    for(int l=0;l<TASK.ex.settings[0];l++) TASK.ex.Orb[l].resize(data.GridSize);
  	//#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)//not thread-safe, reason unknown...
  	for(int i=0;i<data.GridSize;i++){
    	for(int l=0;l<TASK.ex.settings[0];l++) TASK.ex.Orb[l][i] = Orbital(l,data.VecAt[i],TASK.ex);
  	}
  	for(int l=0;l<TASK.ex.settings[0];l++){//check if all states have parity either +1 or -1
    	for(int i=1;i<data.steps/2;i++){
      		if(ABS(ABS(TASK.ex.Orb[l][data.CentreIndex+i])-ABS(TASK.ex.Orb[l][data.CentreIndex-i]))>MP){
				cout << "GetOrb, ill-defined parity... " << vec_to_str(TASK.ex.QuantumNumbers[l]) << " " << i << " " << TASK.ex.Orb[l][data.CentreIndex+i] << " " << TASK.ex.Orb[l][data.CentreIndex-i] << endl; usleep(1000);
      		}
    	}
  	}
}

spline1dinterpolant GetSpline1D(string FileName, bool sortedQ){//Ensure that x values of input file are free of duplicates!
  	//cout << "GetSpline1D: interpolate " << FileName << endl;
  	ifstream infile;
  	string line;
  	infile.open(FileName.c_str());
  	double tmp;
  	int rQ = 1;
  	vector<double> x,f;
    bool success = true;
  	while(success && getline(infile,line)){
    	istringstream LINE(line);
    	if(line != ""){
      		while(LINE>>tmp){
              	if(!std::isfinite(tmp)){
					success = false; break;
              	}
              	else{
					if(rQ){ x.push_back(tmp); rQ=0; }
					else{ f.push_back(tmp); rQ=1; }
              	}
      		}
    	}
  	}
  	int dsize = x.size();
  	infile.close();
    //cout << x.size() << " " << f.size() << endl;

    //Ensure that x values of input file are sorted in ascending order
    vector<double> fSorted(0);
    if(!sortedQ){
    	int count = 0;
    	fSorted.resize(f.size());
    	for(auto i: sort_indices(x)){
			f[count] = f[i];
      		count++;
    	}
    	sort(x.begin(),x.end());
    	for(int i=1;i<x.size();i++) if(x[i]==x[i-1]) cout << i << endl;
    	f.swap(fSorted);
    }
  
  	//produce interpolation
  	real_1d_array X,F;
  	spline1dinterpolant Spline;
    if(success){
  		X.setcontent(dsize, &(x[0]));
  		F.setcontent(dsize, &(f[0]));
    	//cout << " ...content set" << endl;
    	try{
      		alglib::spline1dbuildmonotone(X, F, Spline);
    	} catch (const alglib::ap_error &e) {
      		std::cerr << "ALGLIB Exception: " << e.msg << std::endl;
      		throw;
    	}
  		//spline1dbuildcubic(X, F, Spline); cout << " ...cubic spline built" << endl;
  		spline1dbuildmonotone(X, F, Spline); //cout << " ...monotone spline built" << endl;
    }
    else cout << "GetSpline1D: Error !!! input data file probably corrupted." << endl;
  
  	return Spline;
}

spline1dinterpolant GetSpecialFunction(string TabFunc_Name){
  spline1dinterpolant c;
  ifstream infile;
  infile.open(TabFunc_Name);
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetPolyLog1half(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_PolyLog1half.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetPolyLogm3half(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_PolyLogm3half.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetPolyLogm1half(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_PolyLogm1half.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetPolyLog3half(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_PolyLog3half.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetPolyLog5half(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_PolyLog5half.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;
}

spline1dinterpolant GetCurlyA(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_CurlyA.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;  
}

spline1dinterpolant GetAuxDawsonErfc(void){
  spline1dinterpolant c;
  ifstream infile;
  infile.open("TabFunc_AuxDawsonErfc.dat");
  string line;
  double val;
  vector<double> X(1),Y(1);
  getline(infile, line); istringstream issx(line); issx >> val; X[0] = val;
  getline(infile, line); istringstream issy(line); issy >> val; Y[0] = val;
  while(!infile.eof()){
    getline(infile, line); istringstream issx(line); issx >> val; X.push_back(val);
    getline(infile, line); istringstream issy(line); issy >> val; Y.push_back(val);
  }
  real_1d_array x, y;
  x.setcontent(X.size(), &(X[0]));
  y.setcontent(Y.size(), &(Y[0]));
  spline1dbuildcubic(x, y, c);
  return c;  
}

double polylog(string order, double arg, datastruct &data){
  if(arg<-699.) return 0.;
  else{
    if(order=="m3half"){
	if(arg>699.) return -pow(arg,-1.5)/Gammam1half;
	else return spline1dcalc(data.PolyLogm3half,arg);
    }
    else if(order=="m1half"){
	if(arg>699.) return -pow(arg,-0.5)/Gamma1half;
	else return spline1dcalc(data.PolyLogm1half,arg);
    }
    else if(order=="1half"){
	if(arg>699.) return -pow(arg,0.5)/Gamma3half;
	else return spline1dcalc(data.PolyLog1half,arg);
    }
    else if(order=="3half"){
	if(arg>699.) return -pow(arg,1.5)/Gamma5half;
	else return spline1dcalc(data.PolyLog3half,arg);
    }
    else if(order=="5half"){
	if(arg>699.) return -pow(arg,2.5)/Gamma7half;
	else return spline1dcalc(data.PolyLog5half,arg);
    }      
  }
  
  //default
  return 0.;
}

void InitializePulayMixer(datastruct &data){
	data.Pulay.mixer = (int)(data.Mixer[0]+0.5);
	data.Pulay.scope = (int)data.Mixer[1];


    double freeRAM = getFreeRAM(data);

    unsigned long long requiredRAM = (unsigned long long)data.S * (unsigned long long)data.Pulay.scope * (unsigned long long)data.GridSize * (unsigned long long)(4*8);
    double freeRatio = freeRAM/(double)requiredRAM;

    PRINT("InitializePulayMixer: Required (Free) RAM = " + to_string_with_precision((double)requiredRAM,4) + " (" + to_string_with_precision(freeRAM,4) + ") bytes",data);
    if(freeRatio<0.9){
      	data.Pulay.scope = (int)(0.9*(double)data.Pulay.scope*freeRatio);
        PRINT("InitializePulayMixer: Warning !!! data.Pulay.scope adjusted to " + to_string(data.Pulay.scope),data);
        if(data.Pulay.scope<=1) data.Pulay.mixer = 0;
        PRINT("InitializePulayMixer: Warning !!! Pulay mixing deactivated.",data);

    }
	data.Pulay.scopeOriginal = data.Pulay.scope;
	if(abs(data.Mixer[2])>MP){
      	data.Pulay.weight = data.Mixer[2];
      	if(data.Pulay.weight<0.) data.Pulay.weight = data.k0*data.k0+data.Deltak*data.Deltak;
		else{
			PRINT("InitializePulayMixer: Warning !!! recommended weight = " + to_string(data.k0*data.k0+data.Deltak*data.Deltak) + " instead of " + to_string(data.Pulay.weight),data);
        }
	}
	if(data.Pulay.mixer>1){
		data.Pulay.ResidualMatrixSep.clear();
		data.Pulay.ResidualMatrixSep = vector<vector<vector<double>>>(data.S,vector<vector<double>>(0,vector<double>(0,0.)));
	}
	if(data.DIM>1 && data.Pulay.weight>0.) data.Pulay.MomentumSpaceQ = true;
	//if(data.Pulay.mixer==1) data.Pulay.MomentumSpaceQ = false;
	//if(data.Mixer[3]>0.) data.Pulay.EagerExecutionQ = true;
}

void ReadVec(string FileName, vector<double> &vec){
	ifstream infile;
    infile.open(FileName);
    string line;
    double val;
    while(getline(infile,line)){
      istringstream iss(line);
      if(line != ""){
		iss.str(line);
		iss >> val;
		vec.push_back(val);
      }
    }
    infile.close();	
}

vector<vector<double>> ReadMat(string FileName){
	vector<vector<double>> mat(0);
    ifstream in(FileName);
    string record;
    while(getline(in,record)){
        istringstream iss(record);
        vector<double> row((istream_iterator<double>(iss)),istream_iterator<double>());
        mat.push_back(row);
    }
    return mat;
}

// B A S I C    R O U T I N E S

bool NegligibleMagnitudeQ(double bigQ, double smallQ, double MaxExponentDifference){
  if(smallQ == 0.) return true;
  else if(bigQ == 0. || bigQ<smallQ) return false;
  else{
    if(log10(ABS(bigQ))-log10(ABS(smallQ))>MaxExponentDifference) return true;
    else return false;
  }
}

double VecMult(vector<double> &v,vector<double> &w, datastruct &data){
  double res = 0.;
  for(int i=0;i<v.size();i++) res += v[i]*w[i];
  return res;
}


// A U X I L L I A R Y    R O U T I N E S 

bool BoxBoundaryQ(int index, datastruct &data){
  if(data.DIM==1){ if(index==0 || index==data.steps) return true; else return false; }
  else if(data.DIM==2){
    int i = (int)((double)index/(double)data.EdgeLength+MP);
    int j = index-i*data.EdgeLength;
    if(i==0 || i==data.steps || j==0 || j==data.steps) return true; else return false;
  }
  else if(data.DIM==3){
    int i = (int)((double)index/((double)(data.EdgeLength*data.EdgeLength))+MP);
    int j = (int)((double)(index-i*data.EdgeLength*data.EdgeLength)/(double)data.EdgeLength+MP);
    int k = index-i*data.EdgeLength*data.EdgeLength-j*data.EdgeLength;
    if(i==0 || i==data.steps || j==0 || j==data.steps || k==0 || k==data.steps) return true; else return false;
  }
  
  //default
  return false;
}

vector<double> rGrid(int dim, int steps, vector<double> &frame ){
  vector<double> rgrid(dim*(steps+1));
  switch(dim){
    case 1: {
      double x1 = frame[0];
      double x2 = frame[1];
      double Deltax=(x2-x1)/(double)steps;
      for(int i=0;i<=steps;i++) rgrid[i]=x1+Deltax*(double)i;
      return rgrid; break;}
    case 2: {
      double x1 = frame[0];
      double x2 = frame[1];
      double y1 = frame[2];
      double y2 = frame[3];
      double Deltax=(x2-x1)/(double)steps;
      double Deltay=(y2-y1)/(double)steps;
      for(int i=0;i<=steps;i++) rgrid[i]=x1+Deltax*i;
      for(int i=0;i<=steps;i++) rgrid[steps+1+i]=y1+Deltay*i;
      return rgrid; break;}
    case 3: {
      double x1 = frame[0];
      double x2 = frame[1];
      double y1 = frame[2];
      double y2 = frame[3];
      double z1 = frame[4];
      double z2 = frame[5];
      double Deltax=(x2-x1)/(double)steps;
      double Deltay=(y2-y1)/(double)steps;
      double Deltaz=(z2-z1)/(double)steps;
      for(int i=0;i<=steps;i++) rgrid[i]=x1+Deltax*i;
      for(int i=0;i<=steps;i++) rgrid[steps+1+i]=y1+Deltay*i;
      for(int i=0;i<=steps;i++) rgrid[2*(steps+1)+i]=z1+Deltaz*i;
      return rgrid; break;}
  }
  
  //default
  return rgrid;
}

void VecAtIndex(int Index, datastruct &data, vector<double> &out){
  switch(data.DIM){
    case 1: {
      out[0] = data.Lattice[Index];
      break;
    }
    case 2: {
      int i = (int)((double)Index/(double)data.EdgeLength);
      out[0] = data.Lattice[i];
      out[1] = data.Lattice[(data.EdgeLength-1)+(Index-i*data.EdgeLength+1)];  
      break;
    }
    case 3: {
      int i = (int)((double)Index/((double)data.EdgeLength*(double)data.EdgeLength));
      int j = (int)(((double)Index - (double)i*(double)data.EdgeLength*(double)data.EdgeLength)/(double)data.EdgeLength);
      out[0] = data.Lattice[i];
      out[1] = data.Lattice[data.EdgeLength+j];
      out[2] = data.Lattice[2*data.EdgeLength+(Index-i*data.EdgeLength*data.EdgeLength-j*data.EdgeLength)];
      break;
    }
  }  
}

void kVecAtIndex(int Index, datastruct &data, vector<double> &out){
  switch(data.DIM){
    case 1: {
      out[0] = data.kLattice[Index];
      break;
    }
    case 2: {
      int i = (int)((double)Index/(double)data.EdgeLength);
      out[0] = data.kLattice[i];
      out[1] = data.kLattice[(data.EdgeLength-1)+(Index-i*data.EdgeLength+1)];  
      break;
    }
    case 3: {
      int i = (int)((double)Index/((double)data.EdgeLength*(double)data.EdgeLength));
      int j = (int)(((double)Index - (double)i*(double)data.EdgeLength*(double)data.EdgeLength)/(double)data.EdgeLength);
      out[0] = data.kLattice[i];
      out[1] = data.kLattice[data.EdgeLength+j];
      out[2] = data.kLattice[2*data.EdgeLength+(Index-i*data.EdgeLength*data.EdgeLength-j*data.EdgeLength)];
      break;
    }
  }  
}

vector<vector<double>> GetVecsAt(datastruct &data){
  vector<vector<double>> vecsat(data.GridSize);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    vecsat[i].resize(data.DIM);
    VecAtIndex(i, data, vecsat[i]);
  }
  return vecsat;
}

vector<vector<double>> GetkVecsAt(datastruct &data){
  vector<vector<double>> vecsat(data.GridSize);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    vecsat[i].resize(data.DIM);
    kVecAtIndex(i, data, vecsat[i]);
  }
  return vecsat;
}

void GetNorm2kVecsAt(datastruct &data){
  data.Norm2kVecAt.resize(data.GridSize);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.GridSize;i++){
    data.Norm2kVecAt[i] = Norm2(data.kVecAt[i]);
  }
}

void DelField(vector<double> &InputField, int s, int method, datastruct &data){
	int n = data.EdgeLength;
	double a = data.Lattice[1]-data.Lattice[0], twoa = 2.*a, twelvea = 12.*a;

	//boundary regions are suppressed with data.RegularizationThreshold (if applicable)
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
	for(int i=0;i<data.GridSize;i++) data.TmpField3[i] = InputField[i];

    if(IntegerElementQ(2,data.regularize) || (data.TaperOff && (method==4 || method==5)) ){
        //cout << "DelField: Suppress boundary region for differentiation" << endl;
		double maxInputFieldAtThreshold;
		int firstQ = 1;
		for(int i=0;i<data.GridSize;i++){
			if(ABS(Norm(data.VecAt[i])-data.edge*data.RegularizationThreshold)<2.*data.Deltax){//check field values on the sphere beyond which the field should be tapered off
				if(firstQ){ maxInputFieldAtThreshold = InputField[i]; firstQ = 0; }
				if(InputField[i]>maxInputFieldAtThreshold) maxInputFieldAtThreshold = InputField[i];
			}
		}
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++) data.TmpField3[i] -= maxInputFieldAtThreshold;//shift the whole field downward
		Regularize({{2}}, data.TmpField3, data.RegularizationThreshold, data);//and taper it off towards the box boundary
	}
  
//   if(method==0){//five-point stencil derivative of InputField
//     int I = data.EdgeLength; double a = data.Lattice[1]-data.Lattice[0], twoa = 2.*a, twelvea = 12.*a;
//     #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int i=0;i<I;i++){
//       int index = i;
//       int J = 1; if(data.DIM>1) J = I;
//       for(int j=0;j<J;j++){
// 	index = i*I+j;
// 	int K = 1; if(data.DIM>2) K = I;
// 	for(int k=0;k<K;k++){
// 	  index = i*I*I+j*J+k;
// 	  
// 	  if(i==0 || i==I-1) data.TmpField1[index] = 0.;
// 	  else if(i==1 || i==I-2) data.TmpField1[index] = (InputField[index+I]-InputField[index-I])/twoa;
// 	  else data.TmpField1[index] = (-InputField[index+2*I] + 8.*InputField[index+I] - 8.*InputField[index-I] + InputField[index-2*I])/twelvea;
//       
// 	  if(j==0 || j==I-1) data.TmpField2[index] = 0.;
// 	  else if(j==1 || j==I-2) data.TmpField2[index] = (InputField[index+1]-InputField[index-1])/twoa;
// 	  else data.TmpField2[index] = (-InputField[index+2] + 8.*InputField[index+1] - 8.*InputField[index-1] + InputField[index-2])/twelvea;
// 	}
//       }
//     }  
//   }
	if(method>=0 && method<=3){//0: one-sided finite difference; 1-3: differentiating cubic (monotone) (akima) splines of InputField
		if(data.DIM==1){
			vector<double> X(n), F(n); double dummy;
			for(int i=0;i<n;i++){
				X[i] = data.VecAt[i][0];
				F[i] = data.TmpField3[i];	
			}
			spline1dinterpolant spline; real_1d_array x,f; x.setcontent(n, &(X[0])); f.setcontent(n, &(F[0]));
			if(method==1) spline1dbuildcubic(x, f, spline);
			else if(method==2) spline1dbuildmonotone(x, f, spline);
			else if(method==3) spline1dbuildakima(x, f, spline);//CAUTION: akima spline not working here ?
			if(method>0) for(int i=0;i<n;i++) spline1ddiff(spline,X[i],dummy,data.Dx[s][i],data.D2x[s][i]);
			else{
				if(method==0){
					for(int i=0;i<n-1;i++) data.Dx[s][i] = (F[i+1]-F[i])/a;
					data.Dx[s][n-1] = data.Dx[s][n-2];
					for(int i=0;i<n-1;i++) data.D2x[s][i] = (data.Dx[s][i+1]-data.Dx[s][i])/a;
					data.D2x[s][n-1] = 0.;
/*	  if(i==0 || i==n-1) data.Dx[s][i] = 0.;
	  else if(i==1 || i==n-2) data.Dx[s][i] = (F[i+1]-F[i-1])/twoa;
	  else data.Dx[s][i] = (-F[i+2] + 8.*InputField[i+1] - 8.*InputField[i-1] + InputField[i-2])/twelvea;*/
				}
			}
		}
		else if(data.DIM==2){
			for(int direction = 0; direction<data.DIM;direction++){
				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
				for(int i=0;i<n;i++){
					int index; vector<double> X(n), F(n); double dummy;
					for(int j=0;j<n;j++){
						if(direction==1) index = i*n+j; else if(direction==0) index = j*n+i;
						X[j] = data.VecAt[index][direction];
						F[j] = data.TmpField3[index];
					}
					spline1dinterpolant spline; real_1d_array x,f; x.setcontent(n, &(X[0])); f.setcontent(n, &(F[0]));
					if(method==1) spline1dbuildcubic(x, f, spline);
					else if(method==2) spline1dbuildmonotone(x, f, spline);
					else if(method==3) spline1dbuildakima(x, f, spline);//CAUTION: akima spline not working here ?
					if(method>0) for(int j=0;j<n;j++){
						if(direction==1) index = i*n+j; else if(direction==0) index = j*n+i;
						if(direction==1) spline1ddiff(spline,X[j],dummy,data.Dy[s][index],data.D2y[s][index]);
						else if(direction==0) spline1ddiff(spline,X[j],dummy,data.Dx[s][index],data.D2x[s][index]);
					}
					else{
						if(method==0){
							for(int j=0;j<n;j++){
								if(direction==1){
									index = i*n+j;
									if(j<n-1) data.Dy[s][index] = (F[j+1]-F[j])/a;		    
									else data.Dy[s][index] = (F[j]-F[j-1])/a;
								}
								else if(direction==0){
									index = j*n+i;
									if(j<n-1) data.Dx[s][index] = (F[j+1]-F[j])/a;		    
									else data.Dx[s][index] = (F[j]-F[j-1])/a;
								}
							}
							for(int j=0;j<n;j++){
								if(direction==1){
									index = i*n+j;
									if(j<n-1) data.D2y[s][index] = (data.Dy[s][i*n+j+1]-data.Dy[s][i*n+j])/a;		    
									else data.D2y[s][index] = 0.;
								}
								else if(direction==0){
									index = j*n+i;
									if(j<n-1) data.D2x[s][index] = (data.Dx[s][(j+1)*n+i]-data.Dx[s][j*n+i])/a;		    
									else data.D2x[s][index] = 0.;
								}
							}	      
						}
					}
				}
			}
		}
		else if(data.DIM==3){
			for(int direction = 0; direction<data.DIM;direction++){
				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
				for(int i=0;i<n;i++){
					int index; vector<double> X(n), F(n); double dummy;
					for(int j=0;j<n;j++){
						for(int k=0;k<n;k++){
							if(direction==2) index = i*n*n+j*n+k; else if(direction==1) index = i*n*n+k*n+j; else if(direction==0) index = k*n*n+j*n+i;
							X[k] = data.VecAt[index][direction];
							F[k] = data.TmpField3[index];
						}
						spline1dinterpolant spline; real_1d_array x,f; x.setcontent(n, &(X[0])); f.setcontent(n, &(F[0]));
						if(method==1) spline1dbuildcubic(x, f, spline);
						else if(method==2) spline1dbuildmonotone(x, f, spline);
						else if(method==3) spline1dbuildakima(x, f, spline);//CAUTION: akima spline not working here ?
						if(method>0) for(int k=0;k<n;k++){
							if(direction==2) index = i*n*n+j*n+k; else if(direction==1) index = i*n*n+k*n+j; else if(direction==0) index = k*n*n+j*n+i;
							if(direction==2) spline1ddiff(spline,X[k],dummy,data.Dz[s][index],data.D2z[s][index]);
							else if(direction==1) spline1ddiff(spline,X[k],dummy,data.Dy[s][index],data.D2y[s][index]);
							else if(direction==0) spline1ddiff(spline,X[k],dummy,data.Dx[s][index],data.D2x[s][index]);
						}
						else{
							if(method==0){
								for(int k=0;k<n;k++){
									if(direction==2){
										index = i*n*n+j*n+k;
										if(k<n-1) data.Dz[s][index] = (F[k+1]-F[k])/a;		    
										else data.Dz[s][index] = (F[k]-F[k-1])/a;
									}
									else if(direction==1){
										index = i*n*n+k*n+j;
										if(k<n-1) data.Dy[s][index] = (F[k+1]-F[k])/a;		    
										else data.Dy[s][index] = (F[k]-F[k-1])/a;
									}
									else if(direction==0){
										index = k*n*n+j*n+i;
										if(k<n-1) data.Dx[s][index] = (F[k+1]-F[k])/a;		    
										else data.Dx[s][index] = (F[k]-F[k-1])/a;
									}
								}
								for(int k=0;k<n;k++){
									if(direction==2){
										index = i*n*n+j*n+k;
										if(k<n-1) data.D2z[s][index] = (data.Dz[s][i*n*n+j*n+k+1]-data.Dz[s][i*n*n+j*n+k])/a;		    
										else data.D2z[s][index] = 0.;
									}
									else if(direction==1){
										index = i*n*n+k*n+j;
										if(k<n-1) data.D2y[s][index] = (data.Dy[s][i*n*n+(k+1)*n+j]-data.Dy[s][i*n*n+k*n+j])/a;		    
										else data.D2y[s][index] = 0.;
									}
									else if(direction==0){
										index = k*n*n+j*n+i;
										if(k<n-1) data.D2x[s][index] = (data.Dx[s][(k+1)*n*n+j*n+i]-data.Dx[s][k*n*n+j*n+i])/a;		    
										else data.D2x[s][index] = 0.;
									}
								}	      
							}
						}	    
					}
				}
			}
		}
	}
	else if(method==4 || method==5){//differentiating InputField via FFT
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ data.ComplexField[i][0] = data.TmpField3[i]; data.ComplexField[i][1] = 0.; }
		fftParallel(FFTW_FORWARD, data);
		data.TmpComplexField = data.ComplexField;//Vtilde(kVec)
		for(int deriv=1;deriv<=2;deriv++){
			for(int j=0;j<data.DIM;j++){
				if(deriv>1 || j>0) data.ComplexField = data.TmpComplexField;
				#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
				for(int i=0;i<data.GridSize;i++){
					double fac; if(deriv==1) fac = -data.kVecAt[i][j]; else if(deriv==2) fac = -data.kVecAt[i][j]*data.kVecAt[i][j];
					data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac;
				}
				fftParallel(FFTW_BACKWARD, data);
				if(deriv==1){
					if(j==0) CopyComplexFieldToReal(data,data.Dx[s], data.ComplexField, 1, data.DIM, n);
					else if(j==1) CopyComplexFieldToReal(data,data.Dy[s], data.ComplexField, 1, data.DIM, n);
					else if(j==2) CopyComplexFieldToReal(data,data.Dz[s], data.ComplexField, 1, data.DIM, n);
				}
				else if(deriv==2){
					if(j==0) CopyComplexFieldToReal(data,data.D2x[s], data.ComplexField, 0, data.DIM, n);
					else if(j==1) CopyComplexFieldToReal(data,data.D2y[s], data.ComplexField, 0, data.DIM, n);
					else if(j==2) CopyComplexFieldToReal(data,data.D2z[s], data.ComplexField, 0, data.DIM, n);
				}
			}
		}
		if(data.FLAGS.vW){//for sqrt density in vW-KEDF
          	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
          	for(int i=0;i<data.GridSize;i++){ data.ComplexField[i][0] = sqrt(POS(data.TmpField3[i])); data.ComplexField[i][1] = 0.; }
          	fftParallel(FFTW_FORWARD, data);
          	data.TmpComplexField = data.ComplexField;//Vtilde(kVec)
            for(int j=0;j<data.DIM;j++){
                if(j>0) data.ComplexField = data.TmpComplexField;
              	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
              	for(int i=0;i<data.GridSize;i++){
                	double fac = -data.kVecAt[i][j];
                	data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac;
              	}
              	fftParallel(FFTW_BACKWARD, data);
                if(j==0) CopyComplexFieldToReal(data,data.SqrtDenDx[s], data.ComplexField, 1, data.DIM, n);
                else if(j==1) CopyComplexFieldToReal(data,data.SqrtDenDy[s], data.ComplexField, 1, data.DIM, n);
                else if(j==2) CopyComplexFieldToReal(data,data.SqrtDenDz[s], data.ComplexField, 1, data.DIM, n);
            }
        }
	}
  
	if(method==5){//Stabilize derivatives via convolution
		SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.Dx[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
		fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.Dx[s], data.ComplexField, 0, data.DIM, n);

        if(data.FLAGS.vW){
          	SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.SqrtDenDx[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
          	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
          	for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
          	fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.SqrtDenDx[s], data.ComplexField, 0, data.DIM, n);
        }

		SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.D2x[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data); 
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
		fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.D2x[s], data.ComplexField, 0, data.DIM, n);

		if(data.DIM>1){
			SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.Dy[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
			fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.Dy[s], data.ComplexField, 0, data.DIM, n);

            if(data.FLAGS.vW){
              	SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.SqrtDenDy[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
              	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
              	for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
              	fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.SqrtDenDy[s], data.ComplexField, 0, data.DIM, n);
            }

			SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.D2y[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
			fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.D2y[s], data.ComplexField, 0, data.DIM, n);       
		}
		if(data.DIM>2){
			SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.Dz[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
			fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.Dz[s], data.ComplexField, 0, data.DIM, n);

            if(data.FLAGS.vW){
              	SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.SqrtDenDz[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
              	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
              	for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
              	fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.SqrtDenDz[s], data.ComplexField, 0, data.DIM, n);
            }

			SetComplexField(data,data.ComplexField, data.DIM, n, 0., 0.); CopyRealFieldToComplex(data,data.ComplexField, data.D2z[s], 0, data.DIM, n); fftParallel(FFTW_FORWARD, data);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
			fftParallel(FFTW_BACKWARD, data); CopyComplexFieldToReal(data,data.D2z[s], data.ComplexField, 0, data.DIM, n);      
		}    
	} 
  
	//cout << "first and second derivatives calculated" << endl;
  
	SetField(data,data.GradSquared[s], data.DIM, n, 0.);
	FieldProductToTmpField1(data.Dx[s], data.Dx[s], data); AddField(data,data.GradSquared[s], data.TmpField1, data.DIM, n);
	if(data.DIM>1){ FieldProductToTmpField1(data.Dy[s], data.Dy[s], data); AddField(data,data.GradSquared[s], data.TmpField1, data.DIM, n); }
	if(data.DIM>2){ FieldProductToTmpField1(data.Dz[s], data.Dz[s], data); AddField(data,data.GradSquared[s], data.TmpField1, data.DIM, n); }

	if(data.FLAGS.vW){
      	SetField(data,data.SqrtDenGradSquared[s], data.DIM, n, 0.);
      	FieldProductToTmpField1(data.SqrtDenDx[s],data.SqrtDenDx[s],data); AddField(data,data.SqrtDenGradSquared[s], data.TmpField1, data.DIM, n);
      	FieldProductToTmpField1(data.SqrtDenDy[s],data.SqrtDenDy[s],data); AddField(data,data.SqrtDenGradSquared[s], data.TmpField1, data.DIM, n);
      	FieldProductToTmpField1(data.SqrtDenDz[s],data.SqrtDenDz[s],data); AddField(data,data.SqrtDenGradSquared[s], data.TmpField1, data.DIM, n);
    }
  
	SetField(data,data.Laplacian[s], data.DIM, n, 0.);
	AddField(data,data.Laplacian[s], data.D2x[s], data.DIM, n);
	if(data.DIM>1) AddField(data,data.Laplacian[s], data.D2y[s], data.DIM, n); 
	if(data.DIM>2) AddField(data,data.Laplacian[s], data.D2z[s], data.DIM, n);
  
	//cout << "gradients and laplacian stored" << endl;
  
	if(method==5){//Stabilize GradSquared, SqrtDenGradSquared, and Laplacian via convolution
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ data.ComplexField[i][0] = data.GradSquared[s][i]; data.ComplexField[i][1] = 0.; }
		fftParallel(FFTW_FORWARD, data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
		fftParallel(FFTW_BACKWARD, data);
		CopyComplexFieldToReal(data,data.GradSquared[s], data.ComplexField, 0, data.DIM, n);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++) data.GradSquared[s][i] = POS(data.GradSquared[s][i]);

        if(data.FLAGS.vW){
          	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
          	for(int i=0;i<data.GridSize;i++){ data.ComplexField[i][0] = data.SqrtDenGradSquared[s][i]; data.ComplexField[i][1] = 0.; }
          	fftParallel(FFTW_FORWARD, data);
          	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
          	for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
          	fftParallel(FFTW_BACKWARD, data);
          	CopyComplexFieldToReal(data,data.SqrtDenGradSquared[s], data.ComplexField, 0, data.DIM, n);
          	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
          	for(int i=0;i<data.GridSize;i++) data.SqrtDenGradSquared[s][i] = POS(data.SqrtDenGradSquared[s][i]);
        }

		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ data.ComplexField[i][0] = data.Laplacian[s][i]; data.ComplexField[i][1] = 0.; }
		fftParallel(FFTW_FORWARD, data);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){ double fac = EXP(data.ConvolutionSigma*Norm2(data.kVecAt[i])); data.ComplexField[i][0] *= fac; data.ComplexField[i][1] *= fac; }
		fftParallel(FFTW_BACKWARD, data);
		CopyComplexFieldToReal(data,data.Laplacian[s], data.ComplexField, 0, data.DIM, n);
	}
  
}

void FieldProductToTmpField1(vector<double> &Field1, vector<double> &Field2, datastruct &data){
  int EdgeLength = data.EdgeLength;
  if(data.DIM==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++) for(int j=0;j<EdgeLength;j++){ 
      //#pragma omp simd
      for(int k=0;k<EdgeLength;k++) data.TmpField1[i*n2+j*EdgeLength+k] = Field1[i*n2+j*EdgeLength+k] * Field2[i*n2+j*EdgeLength+k];
    }
  }
  else if(data.DIM==2){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){ 
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++) data.TmpField1[i*EdgeLength+j] = Field1[i*EdgeLength+j] * Field2[i*EdgeLength+j];
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++) data.TmpField1[i] = Field1[i] * Field2[i];   
  }
}

void MultiplyField(datastruct &data,vector<double> &Field, int dim, int EdgeLength, double factor){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++) for(int j=0;j<EdgeLength;j++){ 
      //#pragma omp simd
      for(int k=0;k<EdgeLength;k++) Field[i*n2+j*EdgeLength+k] *= factor;
    }
  }
  else if(dim==2){
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){ 
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++) Field[i*EdgeLength+j] *= factor;
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++) Field[i] *= factor;   
  }
}

void MultiplyComplexField(datastruct &data,vector<vector<double>> &Field, int dim, int EdgeLength, double factor){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++) for(int j=0;j<EdgeLength;j++){ 
      //#pragma omp simd
      for(int k=0;k<EdgeLength;k++){
	Field[i*n2+j*EdgeLength+k][0] *= factor;
	Field[i*n2+j*EdgeLength+k][1] *= factor;
      }
    }
  }
  else if(dim==2){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){ 
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	Field[i*EdgeLength+j][0] *= factor;
	Field[i*EdgeLength+j][1] *= factor;
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
      Field[i][0] *= factor;
      Field[i][1] *= factor; 
    }
  }
}

void SetField(datastruct &data,vector<double> &Field, int dim, int EdgeLength, double value){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++) for(int j=0;j<EdgeLength;j++){ 
      //#pragma omp simd
      for(int k=0;k<EdgeLength;k++) Field[i*n2+j*EdgeLength+k] = value;
    }
  }
  else if(dim==2){
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){ 
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++) Field[i*EdgeLength+j] = value;
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++) Field[i] = value;   
  }
}

void AddField(datastruct &data,vector<double> &Field, vector<double> &FieldToAdd, int dim, int EdgeLength){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      for(int j=0;j<EdgeLength;j++){
	//#pragma omp simd
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  Field[index] += FieldToAdd[index];
	}
      }
    }
  }
  else if(dim==2){
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  Field[index] += FieldToAdd[index];
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
      Field[i] += FieldToAdd[i];
    }
  }
}

void SetComplexField(datastruct &data,vector<vector<double>> &Field, int dim, int EdgeLength, double Re, double Im){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++) for(int j=0;j<EdgeLength;j++){
      //#pragma omp simd
      for(int k=0;k<EdgeLength;k++){
	Field.at(i*n2+j*EdgeLength+k).at(0) = Re; Field.at(i*n2+j*EdgeLength+k).at(1) = Im;
      }
    }
  }
  else if(dim==2){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){ Field.at(i*EdgeLength+j).at(0) = Re; Field.at(i*EdgeLength+j).at(1) = Im; }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){ Field.at(i).at(0) = Re; Field.at(i).at(1) = Im; }
  }
}

void AddFactorComplexField(datastruct &data,vector<vector<double>> &Field, vector<vector<double>> &FieldToAdd, int dim, int EdgeLength, double factorRe, double factorIm){
  int size = EdgeLength; if(dim>1) size *= EdgeLength; if(dim>2) size *= EdgeLength;
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<size;i++){
    Field[i][0] += factorRe*FieldToAdd[i][0];
    Field[i][1] += factorIm*FieldToAdd[i][1];
  }
}

void CopyComplexField(datastruct &data,vector<vector<double>> &Field, vector<vector<double>> &FieldToCopy, int dim, int EdgeLength){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      for(int j=0;j<EdgeLength;j++){
	//#pragma omp simd
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  Field.at(index).at(0) = FieldToCopy.at(index).at(0);
	  Field.at(index).at(1) = FieldToCopy.at(index).at(1);
	}
      }
    }
  }
  else if(dim==2){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  Field.at(index).at(0) = FieldToCopy.at(index).at(0);
	  Field.at(index).at(1) = FieldToCopy.at(index).at(1);
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
      Field.at(i).at(0) = FieldToCopy.at(i).at(0);
      Field.at(i).at(1) = FieldToCopy.at(i).at(1);
    }
  }
}

void CopyRealFieldToComplex(datastruct &data,vector<vector<double>> &Field, vector<double> &FieldToCopy, int Targetcomponent, int dim, int EdgeLength){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  Field[index][Targetcomponent] = FieldToCopy.at(index);
	}
      }
    }
  }
  else if(dim==2){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  Field[index][Targetcomponent] = FieldToCopy.at(index);
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
      Field[i][Targetcomponent] = FieldToCopy.at(i);
    }
  }
}


void CopyComplexFieldToReal(datastruct &data,vector<double> &Field, vector<vector<double>> &FieldToCopy, int component, int dim, int EdgeLength){
  if(dim==3){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  Field[index] = FieldToCopy.at(index).at(component);
	}
      }
    }
  }
  else if(dim==2){
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  Field[index] = FieldToCopy.at(index).at(component);
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
      Field[i] = FieldToCopy.at(i).at(component);
    }
  }
}

void CheckNAN(string routine, vector<vector<double>> &FieldToCheck, int dim, datastruct &data){
  int EdgeLength = data.EdgeLength, n2 = EdgeLength*EdgeLength, C = FieldToCheck[0].size();
  if(dim==3){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index, check;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  for(int c=0;c<C;c++){
	    switch(fpclassify(FieldToCheck.at(index).at(c))){
	      case FP_INFINITE:  data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! infinity @ [" + to_string(index) + "][" + to_string(c) + "]",data); break;
	      case FP_NAN:       data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! imaginary number or 0/0 @ [" + to_string(index) + "][" + to_string(c) + "]",data); break;
	      //case FP_ZERO:      data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_SUBNORMAL: data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_NORMAL:    break;
	      default: break;
	    }
	  }
	}
      }
    }
  }
  else if(dim==2){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index;
      //#pragma omp simd
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  for(int c=0;c<C;c++){
	    switch(fpclassify(FieldToCheck.at(index).at(c))){
	      case FP_INFINITE:  data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! infinity @ [" + to_string(index) + "][" + to_string(c) + "]",data); break;
	      case FP_NAN:       data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! imaginary number or 0/0 @ [" + to_string(index) + "][" + to_string(c) + "]",data); break;
	      //case FP_ZERO:      data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_SUBNORMAL: data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_NORMAL:    break;
	      default: break;
	    }
	  }
      }
    }
  }
  else{
    //#pragma omp simd
    for(int i=0;i<EdgeLength;i++){
	  for(int c=0;c<C;c++){
	    switch(fpclassify(FieldToCheck.at(i).at(c))){
	      case FP_INFINITE:  data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! infinity @ [" + to_string(i) + "][" + to_string(c) + "]",data); break;
	      case FP_NAN:       data.warningCount++; PRINT("CheckNAN(" + routine + "): Warning!!! imaginary number or 0/0 @ [" + to_string(i) + "][" + to_string(c) + "]",data); break;
	      //case FP_ZERO:      data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_SUBNORMAL: data.warningCount++; PRINT("CheckNAN: Warning!!! ",data); break;
	      //case FP_NORMAL:    break;
	      default: break;
	    }
	  }
    }
  }
}

double MaxAbsOfComplexField(datastruct &data,vector<vector<double>> &Field, int dim, int EdgeLength){
  if(dim==3){
    vector<double> TmpStorage(EdgeLength); fill(TmpStorage.begin(), TmpStorage.end(), 0.);
    int n2 = EdgeLength*EdgeLength;
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index; double test = 0., Re, Im;
      for(int j=0;j<EdgeLength;j++){
	for(int k=0;k<EdgeLength;k++){
	  index = i*n2+j*EdgeLength+k;
	  Re = Field.at(index).at(0); Im = Field.at(index).at(1);
	  if(ABS(Re)<1.0e+150 && ABS(Im)<1.0e+150){
	    test = Re*Re+Im*Im;
	    if(test>TmpStorage[i]){ TmpStorage[i] = test; /*cout << TmpStorage[i] << endl;*/ }
	  }
	}
      }
      TmpStorage[i] = sqrt(POS(TmpStorage[i]));
    }
    double max = TmpStorage[0];
    for(int i=0;i<EdgeLength;i++) if(TmpStorage[i]>max) max = TmpStorage[i];
    return POS(max);
  }
  else if(dim==2){
    vector<double> TmpStorage(EdgeLength); fill(TmpStorage.begin(), TmpStorage.end(), 0.);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<EdgeLength;i++){
      int index; double test = 0.;
      for(int j=0;j<EdgeLength;j++){
	  index = i*EdgeLength+j;
	  test = Field.at(index).at(0)*Field.at(index).at(0)+Field.at(index).at(1)*Field.at(index).at(1);
	  if(test>TmpStorage[i]) TmpStorage[i] = test;
      }
      TmpStorage[i] = sqrt(POS(TmpStorage[i]));
    }
    double max = TmpStorage[0];
    for(int i=0;i<EdgeLength;i++) if(TmpStorage[i]>max) max = TmpStorage[i];
    return POS(max);    
  }
  else{
    double test = 0., TmpStorage = 0.;
    for(int i=0;i<EdgeLength;i++){
      test = Field.at(i).at(0)*Field.at(i).at(0)+Field.at(i).at(1)*Field.at(i).at(1);
      if(test>TmpStorage) TmpStorage = test;
    }
    return sqrt(POS(TmpStorage));
  }  
}

// I N T E G R A T I O N    A N D    F F T    R O U T I N E S

double Integrate(int ompThreads, int method, int dim, vector<double> &f, vector<double> &frame){
  switch(method){
    case 1: {return Riemann(ompThreads, dim, f, frame); break; }
    case 2: {return BooleRule(ompThreads, dim, f, frame); break; }
    case 3: {return RiemannBareSum(ompThreads, dim, f, frame); break; }
  }
  
  //default
  return 0.;
}

double Riemann(int ompThreads, int dim, vector<double> &f, vector<double> &frame){
  
  int I = (int)(0.5+pow((double)f.size(),1./(double)dim));
  double res = 0.;
  if(dim==1){   
    for(int i=0;i<f.size();i++){
      double iprorated = 1;
      if(i==0 || i==I-1) iprorated = 0.5;       
      res += iprorated * f[i];
    }
  }
  else if(dim==2){
    vector<double> RES(I);
    #pragma omp parallel for schedule(dynamic) if(ompThreads>1)
    for(int i=0;i<I;i++){
      double iprorated = 1;
      if(i==0 || i==I-1) iprorated = 0.5;      
      for(int j=0;j<I;j++){
	double jprorated = 1;
	if(j==0 || j==I-1) jprorated = 0.5;	
	RES[i] += iprorated * jprorated * f[i*I+j];
      }
    }
    for(int i=0;i<I;i++) res += RES[i];
  }
  else if(dim==3){
    vector<double> RES(I);
    #pragma omp parallel for schedule(dynamic) if(ompThreads>1)
    for(int i=0;i<I;i++){
      double iprorated = 1;
      if(i==0 || i==I-1) iprorated = 0.5;
      for(int j=0;j<I;j++){
	double jprorated = 1;
	if(j==0 || j==I-1) jprorated = 0.5;
	for(int k=0;k<I;k++){
	  double kprorated = 1;
	  if(k==0 || k==I-1) kprorated = 0.5;
	  RES[i] += iprorated * jprorated * kprorated * f[i*I*I+j*I+k];
	}
      }
    }
    for(int i=0;i<I;i++) res += RES[i];    
  }
  for(int i=0;i<dim;i++) res *= (frame[2*i+1]-frame[2*i])/((double)I-1.);
  return res;
}

double RiemannBareSum(int ompThreads, int dim, vector<double> &f, vector<double> &frame){
  //CAUTION: RiemannBareSum assigns the integral of a slightly larger box than frame, because points of f at the boundary contribute in full. Adjust RiemannBareSumCompensationFactor accordingly.
  double res = 0.;
  int fsize = f.size(), I = (int)(0.5+pow((double)fsize,1./(double)dim));
  //#pragma omp parallel for schedule(static) reduction(+: res) if(ompThreads>1 && fsize>10000)
  for(int i=0;i<fsize;i++) res += f[i];
  for(int i=0;i<dim;i++) res *= (frame[2*i+1]-frame[2*i])/((double)I-1.);
  return res*RiemannBareSumCompensationFactor;
}

double BooleRule(int ompThreads, int dim, vector<double> &f, vector<double> &frame){
  int EdgeLength = (int)(0.5+pow((double)f.size(),1./((double)dim)));
  switch(dim){
    case 1: { return BooleRule1D(ompThreads, f, EdgeLength, frame[0], frame[1], (EdgeLength-1)/4); break;}
    case 2: { return BooleRule2D(ompThreads, f, f.size(), EdgeLength, frame[0], frame[1], frame[2], frame[3], (EdgeLength-1)/4); break;}
    case 3: { return BooleRule3D(ompThreads, f, f.size(), EdgeLength, frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], (EdgeLength-1)/4); break;}
  }
  
  //default
  return 0.;
}

double BooleRule1D(int ompThreads, vector<double> &f, int Lf, double a1, double b1, int bp){
  double h=(b1-a1)/(((double)bp)*4.);
  double res=7.*f[0];
  for(int i=1;i<=Lf-4;i+=4) res+=(32.*f[i]+12.*f[i+1]+32.*f[i+2]+14.*f[i+3]);
  res-=14.*f[Lf-1];
  res+=7.*f[Lf-1];

  return res*(2.*h/45.);
}

double BooleRule2D(int ompThreads, vector<double> &entiref, int Lf, int Lf2, double a1, double b1, double a2, double b2, int bp){
  double h=(b1-a1)/(((double)bp)*4.);
  
  vector<double> f(Lf2);
  #pragma omp parallel for schedule(dynamic) if(ompThreads>1)
  for(int i=0;i<Lf2;i++){
    vector<double> f2(Lf2);
    for(int j=0;j<Lf2;j++){
      f2[j]=entiref[i*Lf2+j];
    }
    f[i]=BooleRule1D(ompThreads,  f2, Lf2, a2, b2, bp);
  }

  return BooleRule1D(ompThreads,  f, Lf2, a2, b2, bp);
}

double BooleRule3D(int ompThreads, vector<double> &entiref, int Lf, int Lf2, double a1, double b1, double a2, double b2, double a3, double b3, int bp){
  double h=(b1-a1)/(((double)bp)*4.);
  
  vector<double> f(Lf2);
  
  //cout << " TO BE CHECKED !" << endl;
  
  //#pragma omp parallel for schedule(dynamic) if(ompThreads>1)
  for(int i=0;i<Lf2;i++){
    vector<double> f2(Lf2);
    for(int j=0;j<Lf2;j++){
      vector<double> f3(Lf2);
      for(int k=0;k<Lf2;k++){
	f3[k]=entiref[i*Lf2*Lf2+j*Lf2+k];
      }
      f2[j]=BooleRule1D(ompThreads,  f3, Lf2, a3, b3, bp);
    }
    f[i]=BooleRule1D(ompThreads,  f2, Lf2, a2, b2, bp);
  }
  //#pragma omp barrier

  return BooleRule1D(ompThreads,  f, Lf2, a2, b2, bp);
}

void SplineOnMainGrid(vector<double> &InputField, datastruct &data){//create spline of InputField
  int n = data.EdgeLength;
  if(data.DIM==1){
    vector<double> F(n);
    for(int i=0;i<n;i++){ F[i] = InputField[i]; }
    real_1d_array x,f; x.setcontent(n, &(data.lattice[0])); f.setcontent(n, &(F[0]));
    if(data.SplineType==1) spline1dbuildlinear(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==2) spline1dbuildakima(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==3) spline1dbuildcubic(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==4) spline1dbuildmonotone(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnMainGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }
  else if(data.DIM==2){
    vector<double> F(n*n);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) F[j*n+i] = InputField[i*n+j];//rearrange InputField for ALGLIB interpolation
    real_1d_array x,y,f; x.setcontent(n, &(data.lattice[0])); y.setcontent(n, &(data.lattice[0])); f.setcontent(n*n, &(F[0]));
    if(data.SplineType==1) spline2dbuildbilinearv(x, n, y, n, f, 1, data.TmpSpline2D[data.FocalSpecies]);
    else if(data.SplineType==3) spline2dbuildbicubicv(x, n, y, n, f, 1, data.TmpSpline2D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnMainGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }
  else if(data.DIM==3){
    vector<double> F(n*n*n);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) F[k*n*n+j*n+i] = InputField[i*n*n+j*n+k];//rearrange InputField for ALGLIB interpolation
    real_1d_array x,y,z,f; x.setcontent(n, &(data.lattice[0])); y.setcontent(n, &(data.lattice[0])); z.setcontent(n, &(data.lattice[0])); f.setcontent(n*n*n, &(F[0]));
    if(data.SplineType==1) spline3dbuildtrilinearv(x, n, y, n, z, n, f, 1, data.TmpSpline3D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnMainGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }   
}

void SplineOnStrideGrid(vector<double> &InputField, datastruct &data){//create spline of InputField based on current StrideLevel
  GetStridelattice(data);
  int n = data.steps/data.stride+1, EL = data.EdgeLength;
  if(data.DIM==1){
    vector<double> F(n);
    for(int i=0;i<n;i++){ F[i] = InputField[data.stride*i]; }
    real_1d_array x,f; x.setcontent(n, &(data.Stridelattice[0])); f.setcontent(n, &(F[0]));
    if(data.SplineType==1) spline1dbuildlinear(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==2) spline1dbuildakima(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==3) spline1dbuildcubic(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else if(data.SplineType==4) spline1dbuildmonotone(x, f, data.TmpSpline1D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnStrideGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }
  else if(data.DIM==2){
    vector<double> F(n*n);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) F[j*n+i] = InputField[(data.stride*i)*EL+(data.stride*j)];//rearrange InputField for ALGLIB interpolation
    real_1d_array x,y,f; x.setcontent(n, &(data.Stridelattice[0])); y.setcontent(n, &(data.Stridelattice[0])); f.setcontent(n*n, &(F[0]));
    if(data.SplineType==1) spline2dbuildbilinearv(x, n, y, n, f, 1, data.TmpSpline2D[data.FocalSpecies]);
    else if(data.SplineType==3) spline2dbuildbicubicv(x, n, y, n, f, 1, data.TmpSpline2D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnStrideGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }
  else if(data.DIM==3){
    vector<double> F(n*n*n);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) F[k*n*n+j*n+i] = InputField[(data.stride*i)*EL*EL+(data.stride*j)*EL+(data.stride*k)];//rearrange InputField for ALGLIB interpolation
    real_1d_array x,y,z,f; x.setcontent(n, &(data.Stridelattice[0])); y.setcontent(n, &(data.Stridelattice[0])); z.setcontent(n, &(data.Stridelattice[0])); f.setcontent(n*n*n, &(F[0]));
    if(data.SplineType==1) spline3dbuildtrilinearv(x, n, y, n, z, n, f, 1, data.TmpSpline3D[data.FocalSpecies]);
    else{ data.errorCount++; PRINT("SplineOnStrideGrid: Error!!! SplineType = " + to_string(data.SplineType) + " not implemented.",data); }
  }   
}

void GetStridelattice(datastruct &data){
  //cout << "GetStridelattice" << endl;
  data.stride = (int)(pow(2.,(double)data.StrideLevel)+0.5);
  int n = data.steps/data.stride+1;
  data.Stridelattice.clear(); data.Stridelattice.resize(n);
  //cout << data.StrideLevel << " " << data.stride << " " << data.stride*(n-1) << " " << data.lattice.size() << endl;
  for(int i=0;i<n;i++) data.Stridelattice[i] = data.lattice[data.stride*i];
  //cout << vec_to_str(data.Stridelattice) << endl;
}

void fftParallel(int sign, datastruct &data){//perform fft on data.ComplexField //omp-parallelized - careful about performance when called from within omp loop!!!

  //ToDo: 1D
  
  //monitor box boundaries of computational space:
  if(data.MONITOR) if(monitorBox(data.DIM, data)){
    data.warningCount++; 
    if(sign==FFTW_FORWARD) PRINT("fftParallel: Warning!!! r-space ComplexField before FFT is leaking! ---> enlarge spatial box",data);
    else PRINT("fftParallel: Warning!!! k-space ComplexField before invFFT is leaking! ---> increase spatial resolution",data);
  }
  
  int s = data.steps, s2=s*s, EL = data.EdgeLength, EL2 = EL*EL;
  double L = data.edge, x0 = -L/2., Deltax=L/(double)s, k0 = -PI/Deltax, Deltak=2.*PI/L, argPrefactor;
  if(sign==FFTW_FORWARD) argPrefactor = PI; else argPrefactor = x0*Deltak;

  //feed FFTW algorithm with appropriate values from data.ComplexField
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<s;i++){
    double arg, sinarg, cosarg, Re, Im; int index;
    if(data.DIM==1){
      arg = argPrefactor*(double)i; sinarg = sin(arg); cosarg = cos(arg);
      Re = data.ComplexField.at(i).at(0); Im = data.ComplexField.at(i).at(1);
      data.inFFTW[i][0] = cosarg*Re-sinarg*Im;
      data.inFFTW[i][1] = sinarg*Re+cosarg*Im;      
    }
    else{
      for(int j=0;j<s;j++){
	if(data.DIM==2){
	  index = i*s+j;
	  arg = argPrefactor*(double)(i+j); sinarg = sin(arg); cosarg = cos(arg);
	  Re = data.ComplexField.at(i*EL+j).at(0); Im = data.ComplexField.at(i*EL+j).at(1);
	  data.inFFTW[index][0] = cosarg*Re-sinarg*Im;
	  data.inFFTW[index][1] = sinarg*Re+cosarg*Im;
	}
	else if(data.DIM==3){
	  for(int k=0;k<s;k++){
	    index = i*s2+j*s+k;
	    arg = argPrefactor*(double)(i+j+k); sinarg = sin(arg); cosarg = cos(arg);
	    Re = data.ComplexField.at(i*EL2+j*EL+k).at(0); Im = data.ComplexField.at(i*EL2+j*EL+k).at(1);
	    data.inFFTW[index][0] = cosarg*Re-sinarg*Im;
	    data.inFFTW[index][1] = sinarg*Re+cosarg*Im;	   
	  }
	}
      }
    }
  }
  
  //double test0 = 0., test1 = 0.; for(int i=0;i<data.ComplexField.size();i++){ test0 += ABS(data.inFFTW[i][0]); test1 += ABS(data.inFFTW[i][1]); } cout << "fftParallel before fft: " << test0 << " " << test1 << endl;

  //execute FFT plan
  if(sign==FFTW_FORWARD) fftw_execute(data.FFTWPLANFORWARDPARALLEL);
  else fftw_execute(data.FFTWPLANBACKWARDPARALLEL);
  
  //test0 = 0., test1 = 0.; for(int i=0;i<data.ComplexField.size();i++){ test0 += ABS(data.outFFTW[i][0]); test1 += ABS(data.outFFTW[i][1]); } cout << "fftParallel after fft: " << test0 << " " << test1 << endl;
  
  double factor, constant1, argPrefactor2;
  if(sign==FFTW_FORWARD){ factor = pow(Deltax,data.DDIM); constant1 = -x0*data.DDIM*k0; argPrefactor2 = -x0*Deltak; }
  else{ factor = pow(1./L,data.DDIM); constant1 = k0*data.DDIM*x0; argPrefactor2 = k0*Deltax; }
  
  //export normalized ComplexField
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<s;i++){
    double arg, sinarg, cosarg, Re, Im; int index;
    if(data.DIM==1){
      arg = constant1+argPrefactor2*(double)i; sinarg = sin(arg); cosarg = cos(arg);
      Re = data.outFFTW[i][0]; Im = data.outFFTW[i][1];
      data.ComplexField.at(i).at(0) = factor * (cosarg*Re-sinarg*Im);
      data.ComplexField.at(i).at(1) = factor * (sinarg*Re+cosarg*Im);      
    }
    else{
      for(int j=0;j<s;j++){
	if(data.DIM==2){
	  index = i*s+j;
	  arg = constant1+argPrefactor2*(double)(i+j); sinarg = sin(arg); cosarg = cos(arg);
	  Re = data.outFFTW[index][0]; Im = data.outFFTW[index][1];
	  data.ComplexField.at(i*EL+j).at(0) = factor * (cosarg*Re-sinarg*Im);
	  data.ComplexField.at(i*EL+j).at(1) = factor * (sinarg*Re+cosarg*Im);
	}
	else if(data.DIM==3){
	  for(int k=0;k<s;k++){
	    index = i*s2+j*s+k;
	    arg = constant1+argPrefactor2*(double)(i+j+k); sinarg = sin(arg); cosarg = cos(arg);
	    Re = data.outFFTW[index][0]; Im = data.outFFTW[index][1];
	    data.ComplexField.at(i*EL2+j*EL+k).at(0) = factor * (cosarg*Re-sinarg*Im);
	    data.ComplexField.at(i*EL2+j*EL+k).at(1) = factor * (sinarg*Re+cosarg*Im);	  
	  }
	}
      }
    }
  }
  
  //test0 = 0., test1 = 0.; for(int i=0;i<data.ComplexField.size();i++){ test0 += ABS(data.ComplexField[i][0]); test1 += ABS(data.ComplexField[i][1]); } cout << "fftParallel after normalization: " << test0 << " " << test1 << endl; 

  if(data.DIM==1){
    //fill endpoint
    data.ComplexField.at(s).at(0) = data.ComplexField.at(0).at(0);
    data.ComplexField.at(s).at(1) = data.ComplexField.at(0).at(1);
  }
  else if(data.DIM==2){
    //fill column with index EdgeLength-1==s
    for(int j=0;j<s;j++){
      data.ComplexField.at(EL*s+j).at(0) = data.ComplexField.at(j).at(0);
      data.ComplexField.at(EL*s+j).at(1) = data.ComplexField.at(j).at(1);
    }
    data.ComplexField.at(EL*s+s).at(0) = data.ComplexField.at(0).at(0);
    data.ComplexField.at(EL*s+s).at(1) = data.ComplexField.at(0).at(1);
    //fill row with index EL-1==s
    for(int i=0;i<s;i++){
      data.ComplexField.at(i*EL+s).at(0) = data.ComplexField.at(i*EL).at(0);
      data.ComplexField.at(i*EL+s).at(1) = data.ComplexField.at(i*EL).at(1);
    }
  }
  else if(data.DIM==3){
    //fill entire maximal-i-sheet, i.e. i==EdgeLength-1==s
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int j=0;j<s;j++){
      for(int k=0;k<s;k++){
	data.ComplexField.at(EL2*s+EL*k+j).at(0) = data.ComplexField.at(EL*k+j).at(0);
	data.ComplexField.at(EL2*s+EL*k+j).at(1) = data.ComplexField.at(EL*k+j).at(1);
      }
    }
    for(int j=0;j<s;j++){
      data.ComplexField.at(EL2*s+EL*j+s).at(0) = data.ComplexField.at(EL2*s+EL*j).at(0);
      data.ComplexField.at(EL2*s+EL*j+s).at(1) = data.ComplexField.at(EL2*s+EL*j).at(1);
    }
    for(int k=0;k<s;k++){
      data.ComplexField.at(EL2*s+EL*s+k).at(0) = data.ComplexField.at(EL2*s+k).at(0);
      data.ComplexField.at(EL2*s+EL*s+k).at(1) = data.ComplexField.at(EL2*s+k).at(1);
    }
    data.ComplexField.at(EL2*s+EL*s+s).at(0) = data.ComplexField.at(0).at(0);
    data.ComplexField.at(EL2*s+EL*s+s).at(1) = data.ComplexField.at(0).at(1);
    //fill remainder of maximal-k-sheet
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<s;i++){
      for(int j=0;j<s;j++){
	data.ComplexField.at(EL2*i+EL*j+s).at(0) = data.ComplexField.at(EL2*i+EL*j).at(0);
	data.ComplexField.at(EL2*i+EL*j+s).at(1) = data.ComplexField.at(EL2*i+EL*j).at(1);
      }
    }
    for(int i=0;i<s;i++){
      data.ComplexField.at(EL2*i+EL*s+s).at(0) = data.ComplexField.at(EL2*i+s).at(0);
      data.ComplexField.at(EL2*i+EL*s+s).at(1) = data.ComplexField.at(EL2*i+s).at(1);      
    }
    //fill remainder of maximal-j-sheet
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<s;i++){
      for(int k=0;k<s;k++){
	data.ComplexField.at(EL2*i+EL*s+k).at(0) = data.ComplexField.at(EL2*i+k).at(0);
	data.ComplexField.at(EL2*i+EL*s+k).at(1) = data.ComplexField.at(EL2*i+k).at(1);
      }
    }    
  }
  
  //test0 = 0., test1 = 0.; for(int i=0;i<data.ComplexField.size();i++){ test0 += data.ComplexField[i][0]; test1 += data.ComplexField[i][1]; } cout << "fftParallel after completion: " << test0 << " " << test1 << endl;

  //monitor box boundaries of computational space:
  if(data.MONITOR) if(monitorBox(data.DIM, data)){
    data.warningCount++; 
    if(sign==FFTW_FORWARD) PRINT("fftParallel: Warning!!! k-space ComplexField after FFT is leaking! ---> increase spatial resolution",data);
    else PRINT("fftParallel: Warning!!! r-space ComplexField after invFFT is leaking! ---> enlarge spatial box",data);
  }
  
}

int monitorBox(int dim, datastruct &data){
  int leaking = 0, EL = data.EdgeLength, I = (int)(pow(2.,(double)dim)+0.5);
  vector<int> boxIndices;
//   if(dim==1){ boxIndices.push_back(0); boxIndices.push_back(EL-1); }
//   else if(dim==2){ boxIndices.push_back(0); boxIndices.push_back(EL-1); boxIndices.push_back((EL-1)*EL); boxIndices.push_back(EL*EL-1); }
//   else if(dim==3){ boxIndices.push_back(0); boxIndices.push_back(EL-1); boxIndices.push_back((EL-1)*EL); boxIndices.push_back(EL*EL-1); boxIndices.push_back((EL-1)*EL*EL); boxIndices.push_back((EL-1)*EL*EL+EL-1); boxIndices.push_back((EL-1)*EL*EL+(EL-1)*EL); boxIndices.push_back(EL*EL*EL-1); }
  boxIndices.push_back((EL-1)/2); boxIndices.push_back(data.GridSize-1-(EL-1)/2); boxIndices.push_back((EL-1)/2*EL); boxIndices.push_back((EL-1)/2*EL+EL-1);
  //for(int i=0;i<4;i++) cout << "boxIndices " << boxIndices[i] << endl; usleep(10*sec);
  double test, averageAPCF = 0., maxAPCF = 0., a, b;
  for(int i=0;i<data.ComplexField.size();i++){
    test = sqrt(data.ComplexField.at(i).at(0)*data.ComplexField.at(i).at(0)+data.ComplexField.at(i).at(1)*data.ComplexField.at(i).at(1));
    averageAPCF += test;
    if(test>maxAPCF) maxAPCF = test;
  }
  averageAPCF /= (double)data.ComplexField.size();
  double amax = 0., bmax = 0.; 
  for(int i=0;i<I;i++){
    test = sqrt(data.ComplexField.at(boxIndices[i]).at(0)*data.ComplexField.at(boxIndices[i]).at(0)+data.ComplexField.at(boxIndices[i]).at(1)*data.ComplexField.at(boxIndices[i]).at(1));
    a = test/averageAPCF; b = test/maxAPCF;
    if(a>0.01 && b>0.001){ leaking = 1; amax = max(a,amax); bmax = max(b,bmax); }
  }
  //if(leaking) cout << "monitorBox(data.ComplexField):   Warning!!!  (corner/average --- corner/max): " << amax << " --- " << bmax << endl;
  if(leaking) PRINT("monitorBox: Warning!!! EdgeCentre/max = " + to_string(bmax),data);
  else PRINT("monitorBox: OK",data);
  return leaking;
}


void ConjugateGradientDescent(int tf, int GlobalMinCandidateCounterMax, int countMax, vector<double> &Start, taskstruct &task, datastruct &data){
  //USERINPUT
  double muThreshold = 1.0e-10;  
  
  if(tf>=0) cout << "test ConjugateGradientDescent: " << endl;
  int count = 0, GlobalMinCandidateCounter = 0;
  double res, minres, tfscale;

  TASK.Type = task.Type;
  TASK.Aux = task.Aux; //TASKPRINT("TASK.Aux = " + to_string(TASK.Aux),task,1);
  TASK.count.resize(2);
  
  switch(tf){
    case -4: { cout << "GetAuxilliaryFit" << endl; GLOBALINTEGER = tf; break; }
    case -3: { cout << "GetGlobalMinimum Abundances" << endl; GLOBALINTEGER = tf; break; }
    case -2: { cout << "GetGlobalMinimum muVec" << endl; GLOBALINTEGER = tf; break; }
    case 0: { cout << "Shifted 4D-Gauss = -1 @ [0,1,2,3]" << endl; GLOBALINTEGER = tf; tfscale = 10.; break; }
    case 1: { cout << "Ackley function = 0 @ [0,0]" << endl; GLOBALINTEGER = tf;  tfscale = 10.; break; }
    case 2: { cout << "Beale function = 0 @ [3,0.5]" << endl; GLOBALINTEGER = tf; tfscale = 10.; break; }
    case 3: { cout << "Himmelblau's function = 0 @ {[3,2],[-2.805118,3.131312],[-3.779310,-3.283185],[3.584428,-1.848126]}" << endl; GLOBALINTEGER = tf;  tfscale = 10.; break; }
    case 4: { cout << "Levi function = 0 @ [1,1]" << endl; GLOBALINTEGER = tf; tfscale = 20.; break; }
    case 5: { cout << "Three-hump camel function = 0 @ [0,0]" << endl; GLOBALINTEGER = tf; tfscale = 10.; break; }
    case 6: { cout << "Easom function = -1 @ [" << PI << "," << PI << "]" << endl; GLOBALINTEGER = tf; tfscale = 200.; break; }
    case 7: { cout << "Cross-in-tray function = -2.06261 @ [+-1.34941,+-1.34941]" << endl; GLOBALINTEGER = tf; tfscale = 20.; break; }
    case 8: { cout << "Constrained Eggholder function = -959.6407 @ [" << 512. << "," << 404.2319 << "]" << endl; GLOBALINTEGER = tf; tfscale = 1000.; break; }
    case 9: { cout << "Constrained Hoelder table function = -19.2085 @ [+-8.05502,+-9.66549]" << endl; GLOBALINTEGER = tf; tfscale = 20.; break; }
    case 10: { cout << "Schaffer function = 0.292579 @ {[0,+-" << 1.25313 << "],[+-" << 1.25313 << ",0]}" << endl; GLOBALINTEGER = tf; tfscale = 200.; break; }
    case 11: { cout << "Sqrt[Ackley^2+Camel^2] = 0 @ [0,0,0,0]" << endl; GLOBALINTEGER = tf; tfscale = 10.; break; }
    default: { break; }
  }  
  
  if(tf<=-100){
    cout << "GetGlobalMinimum 1p-exact DFT" << endl;
    GLOBALINTEGER = tf;
  }
   
  while(count<countMax && GlobalMinCandidateCounter<GlobalMinCandidateCounterMax){
    count++;
    TASK.count[0] = count;
    TASK.CGinProgress = true;
    TASK.lastQ = false;

    vector<double> InitParam, Scale; double scale = 1.;
    if(tf<=-100){
      InitParam = OccNumToAngles(Start);
      Scale.resize(InitParam.size());
      fill(Scale.begin(),Scale.end(),1.);
    }   
    else{
      int NumParam = task.HyperBox.size(); 
      if(tf>=0){ NumParam = 2; if(tf==0 || tf==11) NumParam = 4; }
      InitParam.resize(NumParam);
      Scale.resize(NumParam);
      for(int h=0;h<NumParam;h++){
	if(tf>=0){ Scale[h] = tfscale; InitParam[h] = Scale[h]*data.RN(data.MTGEN); }
	else{
	  Scale[h] = /*0.5**/(task.HyperBox[h][1]-task.HyperBox[h][0]);
	  if(Norm(Start)<MP) InitParam[h] = task.HyperBox[h][0] /*+ 0.5*Scale[h]*/ + Scale[h]*data.RNpos(data.MTGEN);
	  else InitParam[h] = Start[h];
	  if(tf==-2 && ABS(InitParam[h])<muThreshold){ if(InitParam[h]>-muThreshold) InitParam[h] += muThreshold; else InitParam[h] -= muThreshold; }
	}
      }
      scale = accumulate(Scale.begin(),Scale.end(),0.0)/Scale.size();
    }
    TASKPRINT("ConjugateGradientDescent " + to_string(count) + ": Start from " + vec_to_str(InitParam) + "\n",task,1);
    
    real_1d_array x,s;
    x.setcontent(InitParam.size(), &(InitParam[0]));
    s.setcontent(Scale.size(), &(Scale[0]));
    
    //USERINPUT
    ae_int_t maxits = 0;//0 for unlimited 
    double epsf = 0, epsg = 0, epsx = 0, diffstep = 1.0e-6*scale;
    if(data.System==0){ epsf = 1.0e-4; diffstep = 1.0e-6*scale; }
    else if(data.System==1){ epsf = 0; diffstep = 1.0e-1*scale; }
    else if(data.System==2){
      epsg = MP; epsx = MP; epsf = /*MP;*/ 1.0e-4; /*1.0e-8;*/ /*0;*/ diffstep = /*1.0e-6;*/ /*1.0e-4*scale;*/ /*1.0e-2*scale;*/ scale;
      //epsg = 0; epsx = 0; epsf = 0; /*1.0e-4;*/ /*1.0e-8;*/ /*0;*/ diffstep = 1.0e-6; /*1.0e-4*scale;*/ /*1.0e-2*scale;*/ /*scale;*/
    }
    if(tf<=-100){ epsg = 0; epsx = 0; epsf = 0;/*1.0e-4;*/ diffstep = 1.0e-1*scale; }
    //OPTstruct opt; opt.pso.TargetMinEncounters = 17;
    
    mincgstate state;
    mincgcreatef(x, diffstep, state);
    mincgsetcond(state, epsg, epsf, epsx, maxits);
    mincgsetscale(state, s);
    mincgsetprecscale(state);

    mincgreport rep;
    mincgoptimize(state, FunctionToBeMinimized);
    //mincgoptimize(state, FunctionToBeMinimized, NULL, (void*)&opt);
    mincgresults(state, x, rep);
    task.controltaskfile << TASK.controltaskfile.str();
    
    string TerminationType;
    switch(rep.terminationtype){
      case -8: { TerminationType = "internal integrity control detected infinite or NAN values in function/gradient. Abnormal termination signalled."; break; }
      case 1:  { TerminationType = "relative function improvement is no more than EpsF = " + to_string_with_precision(epsf,16); break; }
      case 2:  { TerminationType = "relative step is no more than EpsX = " + to_string_with_precision(epsx,16); break; }
      case 4:  { TerminationType = "gradient norm is no more than EpsG = " + to_string_with_precision(epsg,16); break; }
      case 5:  { string MaxIts; if(maxits==0) MaxIts = "infinite #"; else MaxIts = to_string(maxits); TerminationType = MaxIts + " steps were taken"; break; }
      case 7:  { TerminationType = "stopping conditions are too stringent, further improvement is impossible, X contains best point found so far."; break; }
      case 8:  { TerminationType = "terminated by user who called mincgrequesttermination(). X contains point which was current accepted when termination request was submitted."; break; }
      default: { break; }
    }
    TASKPRINT("ConjugateGradientDescent: recompute result from conjugate gradient descent",task,1);
    if(tf==-3) for(int i=0;i<x.length();i++) if(x[i]<0.){ TASKPRINT("ConjugateGradientDescent: x[" + to_string(i) + "] = " + to_string(x[i]) + "->" + to_string(-x[i]) + "\n",task,1); x[i]=0.; }
    if(count==countMax){
      TASK.CGinProgress = false;
      TASK.lastQ = true;
      TASK.ConjGradParam = {{epsf},{epsg},{epsx},{diffstep}};
    }
    res = functionToBeMinimized(x,GLOBALINTEGER);
//     task.controltaskfile << TASK.controltaskfile.str();
    
    string ParamVec = "";
    if(tf<=-100){
      for(int l=0;l<TASK.ex.OccNum.size();l++){ TASK.ex.OccNum[l] = 1.+cos(x[l]); ParamVec += to_string_with_precision(TASK.ex.OccNum[l],8) + " "; }
      TASK.ex.CGDgr = (int)rep.nfev;
      InitParam = AnglesToOccNum(InitParam,TASK.ex);
    }
    else for(int i=0;i<x.length();i++) ParamVec += to_string_with_precision(x[i],8) + " ";
    
    if(count==1 || res<minres){
      TASKPRINT("ConjugateGradientDescent: " + to_string(count) + ". Candidate for global minimum " + to_string_with_precision(res,10) + " @ {" + ParamVec + "}",task,1);
      TASKPRINT("                         [ #Iterations = " + to_string(rep.iterationscount) + " #GradEvals = " + to_string(rep.nfev) + " #TerminationType = " + TerminationType + " from starting point" + "{" + vec_to_str_with_precision(InitParam,8) + "} ]",task,1);
      GlobalMinCandidateCounter++;
      if(GlobalMinCandidateCounter==GlobalMinCandidateCounterMax) TASKPRINT("GlobalMinCandidateCounter = " + to_string(GlobalMinCandidateCounter) + " @ count = " + to_string(count),task,1);
      minres = res;
    }
    if(count==countMax){
      TASKPRINT("ConjugateGradientDescent: countMax = " + to_string(countMax) + " reached...",task,1);
      TASKPRINT("*** ConjugateGradientDescent completed ***\n",task,1);  
    }
    
    //for(int i=0;i<x.length();i++) Start[i] = x[i];//be aware of length of Start vs x
    
    if(tf==-2) TASKPRINT("ConjugateGradientDescent: Emin = " + to_string(res) + " @ muVec = " + ParamVec + "\n",task,1);
    else if(tf==-3) TASKPRINT("ConjugateGradientDescent: Emin = " + to_string(res) + " @ Abundances = " + ParamVec + "\n",task,1);
    else if(tf==-4) TASKPRINT("ConjugateGradientDescent: AuxFit = " + to_string(res) + " @ " + ParamVec + "\n",task,1);
    else if(tf<=-100) TASKPRINT("ConjugateGradientDescent: Emin = " + to_string(res) + " @ OccNum = " + vec_to_str(TASK.ex.OccNum) + "\n",TASK,1);
    //usleep(10*sec);
    
//     TASKPRINT("Print task.controltaskfile:\n",task,1);
//     cout << task.controltaskfile.str() << endl;
//     TASKPRINT("Print TASK.controltaskfile:\n",task,1);
//     cout << TASK.controltaskfile.str() << endl;  
    
  }

}

void FunctionToBeMinimized(const real_1d_array &x, double &func, void *ptr){
  vector<double> XX(x.length()); for(int i=0;i<XX.size();i++) XX[i] = x[i];
  real_1d_array xx; xx.setcontent(XX.size(), &(XX[0]));
  //OPTstruct opt = *((OPTstruct *)ptr); cout << opt.pso.TargetMinEncounters << endl;
  func = functionToBeMinimized(xx,GLOBALINTEGER);
}

double functionToBeMinimized(real_1d_array &x, int globalinteger){
  
  if(globalinteger<0){
    if(TASK.ABORT){
      cout << "functionToBeMinimized: Abort !!!" << endl;
      return 0.;
    }
    else if(globalinteger<=-100){//1p-exact DFT
      for(int l=0;l<TASK.ex.settings[0];l++) TASK.ex.OccNum[l] = 1.+cos(x[l]);
      //does not work?
      //NearestProperOccNum(TASK.ex.OccNum,TASK.ex);//replace OccNum...
      //vector<double> Angles = OccNumToAngles(TASK.ex.OccNum); for(int l=0;l<TASK.ex.settings[0];l++) x[l] = Angles[l];//...and x
      if(printQ(TASK.ex.FuncEvals)) TASKPRINT("ConjGradDesc(" + to_string(TASK.ex.FuncEvals) + "): OccNum = " + vec_to_str_with_precision(TASK.ex.OccNum,8) + "\n",TASK,1);
      return Etot_1pExDFT(TASK.ex.OccNum,TASK.ex);
    }
    else{
      datastruct localdata;
      TASK.VEC.resize(x.length());
      for(int s=0;s<x.length();s++){
	if(globalinteger==-2) TASK.VEC[s] = x[s];
	else if(globalinteger==-3) TASK.VEC[s] = POS(x[s]);
	//else if(globalinteger==-3) TASK.VEC[s] = min(max(0.01,x[s]),5.);
	else if(globalinteger==-4) TASK.VEC[s] = max(0.001,x[s]);
      }
      
      //TASK.Type = -globalinteger;
      GetData(TASK,localdata);
      if(globalinteger==-2){
	
	//adaptive maxSCcount
//	datastruct tmplocaldata;
// 	if(localdata.SCcount<localdata.maxSCcount) TASK.Aux = 1.05*(double)localdata.SCcount;
// 	else{ GetInputParameters(TASK,tmplocaldata); TASK.Aux = (double)tmplocaldata.maxSCcount; }
// 	TASKPRINT("...update maxSCcount -> " + to_string((int)TASK.Aux),TASK,1);
	
// 	TASKPRINT("...SC = " + to_string(localdata.SCcount),TASK,1);
// 	if(GLOBALCOUNT%(1+4*localdata.S)==0){ TASKPRINT("...restart",TASK,1); TASK.Aux = MP; }
// 	else{ TASK.Aux = 2.+MP; TASK.V = localdata.V; }
// 	TASKPRINT("...InterpolVQ = " + to_string((int)TASK.Aux),TASK,1);
	if(TASK.lastQ) TASK.controltaskfile << localdata.controlfile.str();
      }
      return localdata.Etot;
    }
  }
  else switch(globalinteger){
    case 0: { return -EXP(-x[0]*x[0]-(x[1]-1.)*(x[1]-1.)-(x[2]-2.)*(x[2]-2.)-(x[3]-3.)*(x[3]-3.)); break; }
    case 1: { return -20.*EXP(-0.2*sqrt(0.5*x[0]*x[0]+x[1]*x[1]))-EXP(0.5*(cos(2*PI*x[0])+cos(2*PI*x[1])))+EXP(1.)+20.; break; }
    case 2: { return pow(1.5-x[0]+x[0]*x[1],2.) + pow(2.25-x[0]+x[0]*x[1]*x[1],2.) + pow(2.625-x[0]+x[0]*x[1]*x[1]*x[1],2.); break; }
    case 3: { return pow(x[0]*x[0]+x[1]-11.,2.) + pow(x[0]+x[1]*x[1]-7.,2.); break; }
    case 4: { return pow(sin(3.*PI*x[0]),2.) + pow(x[0]-1.,2.)*(1.+pow(sin(3.*PI*x[1]),2.)) + pow(x[1]-1.,2.)*(1.+pow(sin(2.*PI*x[1]),2.)); break; }
    case 5: { return 2.*x[0]*x[0]-1.05*pow(x[0],4.)+pow(x[0],6.)/6.+x[0]*x[1]+x[1]*x[1]; break; }
    case 6: { return -cos(x[0])*cos(x[1])*EXP(-pow(x[0]-PI,2.)-pow(x[1]-PI,2.)); break; }
    case 7: { return -0.0001*pow(1.+ABS(sin(x[0])*sin(x[1])*EXP(ABS(100.-sqrt(x[0]*x[0]+x[1]*x[1])/PI))),0.1); break; }
    case 8: { if(ABS(x[0])>512. || ABS(x[1])>512.) return 0.; else return -(x[1]+47.)*sin(sqrt(ABS(0.5*x[0]+(x[1]+47.))))-x[0]*sin(sqrt(ABS(x[0]-(x[1]+47.)))); break; }
    case 9: { if(ABS(x[0])>10. || ABS(x[1])>10.) return 0.; else return -ABS(sin(x[0])*cos(x[1])*EXP(ABS(1.-sqrt(x[0]*x[0]+x[1]*x[1])/PI))); break; }
    case 10: { return 0.5 + (pow(cos(sin(ABS(x[0]*x[0]-x[1]*x[1]))),2.)-0.5)/pow(1.+0.001*(x[0]*x[0]+x[1]*x[1]),2.); break; }    
    case 11: { return sqrt(pow(-20.*EXP(-0.2*sqrt(0.5*x[0]*x[0]+x[1]*x[1]))-EXP(0.5*(cos(2*PI*x[0])+cos(2*PI*x[1])))+EXP(1.)+20.,2.)+pow(2.*x[2]*x[2]-1.05*pow(x[2],4.)+pow(x[2],6.)/6.+x[2]*x[3]+x[3]*x[3],2.)); break; }
    default: { return 0.; break; }
  }

}

void GradientDescent(taskstruct &task){
  int ParamDim = DATA.TaskHyperBox.size()/2;
  double scale = 0.; for(int i=0;i<ParamDim;i++) scale += (DATA.TaskHyperBox[2*i+1] - DATA.TaskHyperBox[2*i])/((double)ParamDim);
  double averageAbundance = 0.; for(int s=0;s<DATA.S;s++) averageAbundance += DATA.Abundances[s]/((double)DATA.S);
  scale = min(scale,averageAbundance);
  double AbsoluteTargetAcc = 1.0e-4; if(DATA.RelAcc>MP) AbsoluteTargetAcc = DATA.RelAcc*scale;
  double gammaStart = 1.0e-2*scale;
  double AnnealMagnitude = 3.;
  double AnnealThreshold = 0.7;
  double fmin;
  int maxIter = 200;
  vector<vector<double>> Report, XStart;
  vector<double> x(ParamDim), xStart(x), xmin(x);
  bool success = false;
  while(!success && !TASK.ABORT){
    task.count[1] = 1;
    for(int i=0;i<ParamDim;i++) x[i] = DATA.TaskHyperBox[2*i]+DATA.RNpos(DATA.MTGEN)*(DATA.TaskHyperBox[2*i+1] - DATA.TaskHyperBox[2*i]);
    xStart = x; XStart.push_back(xStart);
    vector<double> report;//columns: x df f iter Error gamma
    gradient_descent(report, x, AbsoluteTargetAcc, gammaStart, maxIter, AnnealThreshold, AnnealMagnitude, task);
    TASKPRINT("GradientDescent: history",task,1);
    TASKPRINT("                 scale             = " + to_string_with_precision(scale,16),task,1);
    TASKPRINT("                 AbsoluteTargetAcc = " + to_string_with_precision(AbsoluteTargetAcc,16),task,1);
    TASKPRINT("                 gammaStart        = " + to_string_with_precision(gammaStart,16),task,1);
    TASKPRINT("                 AnnealMagnitude   = " + to_string_with_precision(AnnealMagnitude,16),task,1);
    Report.push_back(report);
    if(report[2*ParamDim+1]==maxIter) AnnealMagnitude *= 0.7; else AnnealMagnitude /= 0.8;
    if(Report.size()==1 || report[2*ParamDim]<fmin){ fmin = report[2*ParamDim]; xmin = x; }
    double fminEncounter = 0;
    for(int r=0;r<Report.size();r++){
      vector<double> xtest; for(int i=0;i<ParamDim;i++) xtest.push_back(Report[r][i]);
      if(ABS(Report[r][2*ParamDim]-fmin)<10.*AbsoluteTargetAcc && ABS(2.*VecMult(xtest,xmin,DATA)-(Norm2(xtest)+Norm2(xmin)))<10.*AbsoluteTargetAcc){ fminEncounter++; }
      TASKPRINT("f = " + to_string_with_precision(Report[r][2*ParamDim],16) + " @ x = " + vec_to_str(xtest) + " [xStart = " + vec_to_str(XStart[r]) + "] <--report #" + to_string(r+1),task,1);
    }
    if(fminEncounter==DATA.TaskParameterCount) success = true;
    vector<double> df; for(int i=0;i<ParamDim;i++) df.push_back(report[ParamDim+i]);
    TASKPRINT("GradientDescent: report " + to_string(Report.size()) + " (x df f iter Error gamma) ",task,1);
    TASKPRINT("                               x = " + vec_to_str(x),task,1);
    TASKPRINT("                              df = " + vec_to_str(df),task,1);
    TASKPRINT("              f iter Error gamma = " + to_string_with_precision(report[2*ParamDim],16) + " " + to_string(report[2*ParamDim+1]) + " " + to_string_with_precision(report[2*ParamDim+2],16) + " " + to_string_with_precision(report[2*ParamDim+3],16),task,1);
    TASKPRINT("                 fminEncounter   = " + to_string(fminEncounter) + " fmin = " + to_string(fmin) + " xmin = " + vec_to_str(xmin) + " xStart = " + vec_to_str(xStart),task,1);
    TASKPRINT("                 AnnealMagnitude = " + to_string(AnnealMagnitude),task,1);
    task.count[0]++;
  }

  TASKPRINT("                 Local minimum f = " + to_string(Report[Report.size()-1][2*ParamDim]) + " @ x = " + vec_to_str(x),task,1);  
}

void gradient_descent(vector<double> &report, vector<double> &x, double AbsoluteTargetAcc, double gammaStart, int maxIter, double AnnealThreshold, double AnnealMagnitude, taskstruct &task){
  double f, gamma = gammaStart, Error = AbsoluteTargetAcc + 1, anneal = 0.;
  int iter = 0, successCount = 0, fokCount = 0, MaxsuccessCount = 3, MaxfokCount = 10;
  vector<vector<double>> xHistory;
  vector<double> xNew(x), xOld(x), df(x), dfOld(x), xDiff(x), Dx(x), Df(x), fdiff, dfHistory, ErrorHistory, fHistory;
  bool SmalldfQ = false, success = false, Abort = false;
  while(iter < maxIter && (Error > AbsoluteTargetAcc || !success) && fokCount<MaxfokCount){
    xHistory.push_back(x); if(iter>0) xOld = xHistory[xHistory.size()-2];
    dfOld = df;
    fdiff.resize(x.size());
        
    task.VEC.resize(x.size()); task.VEC = x; task.count[2] = 1;
    for(int i=0;i<=x.size();i++){
      datastruct localdata;
      localdata.GradDesc.x.resize(x.size());
      localdata.GradDesc.x = x; //cout << vec_to_str(localdata.GradDesc.x) << endl;
      if(i<x.size()) localdata.GradDesc.x[i] += gammaStart;
      //cout << vec_to_str(localdata.GradDesc.x) << endl;
      GetData(task,localdata);
      if(localdata.SCcount>=localdata.maxSCcount){ Abort = true; break; }
      if(i==x.size()){ f = localdata.GradDesc.f; }
      else fdiff[i] = localdata.GradDesc.f;
      task.count[2]++;
      task.count[3]++;
    }
    task.count[1]++;
    if(Abort) break;
    
    //test case
//     //for(int i=0;i<df.size();i++) df[i] = 4.*(x[i]*x[i]*x[i])-9.*(x[i]*x[i]);
//     f = pow(x[0],4.)-3.*pow(x[0],3.) + pow(x[1],4.)-3.*pow(x[1],3.) + pow(x[2],4.)-3.*pow(x[2],3.);
//     fdiff[0] = pow(x[0]+gammaStart,4.)-3.*pow(x[0]+gammaStart,3.) + pow(x[1],4.)-3.*pow(x[1],3.) + pow(x[2],4.)-3.*pow(x[2],3.);
//     fdiff[1] = pow(x[0],4.)-3.*pow(x[0],3.) + pow(x[1]+gammaStart,4.)-3.*pow(x[1]+gammaStart,3.) + pow(x[2],4.)-3.*pow(x[2],3.);
//     fdiff[2] = pow(x[0],4.)-3.*pow(x[0],3.) + pow(x[1],4.)-3.*pow(x[1],3.) + pow(x[2]+gammaStart,4.)-3.*pow(x[2]+gammaStart,3.);
    
    for(int i=0;i<x.size();i++) df[i] = (fdiff[i]-f)/gammaStart;
    dfHistory.push_back(Norm(df)); fHistory.push_back(f); double dfAverage = 0., fAverage = 0., I = min(10.,(double)dfHistory.size());
    for(int i=1;i<=(int)(I+0.5);i++){ dfAverage += dfHistory[dfHistory.size()-i]/I; fAverage += fHistory[fHistory.size()-i]/I; }
    if(iter>0){
      if(ABS(f-fAverage)<DATA.RelAcc*ABS(f+fAverage)) fokCount++; else fokCount = 0;
      cout << "gradient_descent: " << Norm(df) << " " << AnnealThreshold*dfAverage << endl;
      if(Norm(df)<AnnealThreshold*dfAverage){
	anneal = /*Heaviside(DATA.RN(DATA.MTGEN))*/ Heaviside((0.5+0.5*tanh(10.*ASYM(f,fHistory[fHistory.size()-2])))/sqrt((double)iter)-DATA.RNpos(DATA.MTGEN)) * DATA.RN(DATA.MTGEN) * (AnnealMagnitude + pow(AnnealMagnitude*AnnealMagnitude,DATA.RNpos(DATA.MTGEN)));
	cout << "anneal ---> " << anneal << endl;
	if(ErrorHistory[ErrorHistory.size()-1]>ErrorHistory[ErrorHistory.size()-2]) gammaStart *= 0.5;
	else gammaStart *= 1.2;
      }
      else anneal = 0.;
      Dx = VecDiff(x,xOld); Df = VecDiff(df,dfOld);
      if(Norm2(Df)>0.) gamma = ABS(VecMult(Dx,Df,DATA))/Norm2(Df); else gamma = gammaStart;
      if(ABS(anneal)!=0.) gamma *= anneal;
      
    }
    xDiff = VecFact(df,gamma); if(Norm(xDiff)==0.) break;
    xNew = VecDiff(x,xDiff); for(int i=0;i<x.size();i++) xNew[i] = min(max(xNew[i],DATA.TaskHyperBox[2*i]),DATA.TaskHyperBox[2*i+1]);//don't leave the Hyperbox
    Error = Norm(xDiff); ErrorHistory.push_back(Error);
    if(Error <= AbsoluteTargetAcc) successCount++; else successCount = 0; if(successCount>=MaxsuccessCount) success = true; //cout << successCount << " " << success << endl;
    TASKPRINT("gradient_descent: iter = " + to_string(iter+1) + "/" + to_string(maxIter) + ": f = " + to_string_with_precision(f,16) + "\n gamma = " + to_string_with_precision(gamma,16) + "(anneal=" + to_string_with_precision(anneal,16) + ")" + " -> Error = " + to_string_with_precision(Error,16) + "(" + to_string_with_precision(AbsoluteTargetAcc,16) + ") ---> xNew = " + vec_to_str(xNew) + " successCount = " + to_string(successCount) + "/" + to_string(MaxsuccessCount),task,1);
    x = xNew;
    iter++;
  }
  //if(iter==maxIter && Error > AbsoluteTargetAcc){ ConjugateGradientDescent(-3, 1, 1, x, task, DATA); }
  for(int i=0;i<x.size();i++) report.push_back(x[i]);
  for(int i=0;i<x.size();i++) report.push_back(df[i]);
  //test case
  //f = pow(x[0],4.)-3.*pow(x[0],3.) + pow(x[1],4.)-3.*pow(x[1],3.) + pow(x[2],4.)-3.*pow(x[2],3.);
  report.push_back(f);//here, f is actually the value at xOld, not xNew, but it should not differ much if the minimum is reached, and it avoids another GetData
  report.push_back((double)iter);
  report.push_back(Error);
  report.push_back(gamma);
}


// T E S T G R O U N D    R O U T I N E S

void RunTests(taskstruct &task, datastruct &data){
  //testSWAP(1,data);  
  //testFFT(data);
  //testOMP(4096,data);
  //testCPU(100,data);
  //for(int tf=0;tf<=11;tf++) ConjugateGradientDescent(tf, 10, 1000, task, data);
  //for(int tf=0;tf<=11;tf++) ConjugateGradientDescent(tf, 1, 1, task, data); cout << "sleep..." << endl; sleep(1000*sec);
  //testBoxBoundaryQ(data);
  //GradientDescent(data);
/*  getLIBXC(1, 0, -XC_LDA_X, data.libxcPolarization, data); */
  //testKD(data);
  //test1pEx(data);
  //testALGfit(data);
  //testisfinite();
  //testPCA(data);
  //testRBF(data);
  //testNN(data);
  //OptimizeNN(data);
  //testcec14(data,task);
  //testOptimizers(data,task);
  //test1pExDFTN2(data,task);
  //testALGintegrator(data,task);
  //testHydrogenicHint(data,task);
  //testK1(data,task);
  //vector<double> x(12,3.0); testIAMIT(x,data,task);
}

void RunAuxTasks(taskstruct &task, datastruct &data){
	ParetoFront(data,task);
}


void FillComplexField(int option, datastruct &data){
  if(option==1){
    #pragma omp parallel for schedule(static) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double n2 = Norm2(data.VecAt[i]);
      data.ComplexField[i][0] = EXP(-n2/(2.*pow(data.edge/2.,2.)))*sin(n2);
      data.ComplexField[i][1] = 0.;
    }
  }
  else if(option==2){
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double n2 = Norm2(data.VecAt[i]);
      data.ComplexField[i][0] = EXP(-n2/(2.*pow(data.edge/2.,2.)))*sin(n2);
      data.ComplexField[i][1] = 0.;
    }    
  }
  else if(option==3){
    #pragma omp parallel for schedule(guided) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double n2 = Norm2(data.VecAt[i]);
      data.ComplexField[i][0] = EXP(-n2/(2.*pow(data.edge/2.,2.)))*sin(n2);
      data.ComplexField[i][1] = 0.;
    }    
  }
  else if(option==4){
    #pragma omp parallel for schedule(auto) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double n2 = Norm2(data.VecAt[i]);
      data.ComplexField[i][0] = EXP(-n2/(2.*pow(data.edge/2.,2.)))*sin(n2);
      data.ComplexField[i][1] = 0.;
    }    
  }
  if(option==5){
    #pragma omp parallel for schedule(runtime) if(data.ompThreads>1)
    for(int i=0;i<data.GridSize;i++){
      double n2 = Norm2(data.VecAt[i]);
      data.ComplexField[i][0] = EXP(-n2/(2.*pow(data.edge/2.,2.)))*sin(n2);
      data.ComplexField[i][1] = 0.;
    }
  }
}

void passDoubleOMPloop(int n, vector<double> &AV, vector<double> &AV2){
    //#pragma omp parallel for schedule(dynamic)
    for(int ss=0;ss<3;ss++){
      //#pragma omp parallel for schedule(dynamic)
      for(int sss=0;sss<3;sss++){
	
	for(int i=0;i<n;i++){
	double val;
	  for(int j=0;j<n;j++){
	    for(int dummy=0;dummy<20;dummy++){
	      val = (double)(dummy+i*i+i)/1000.;
	      int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
	    }
	  }
	}
	
      }
    }
}

void testSWAP(int n, datastruct &data){
  SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
  SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 1.);
  PRINT("testSWAP(integrals of Field1 and Field2): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  
  StartTiming(data);
  for(int i=0;i<n;i++){
      data.TmpField1 = data.TmpField2;
  }
  EndTiming(data);
  PRINT("testSWAP(copy Field2 into Field1): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  
  SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
  SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 1.);
  PRINT("testSWAP(integrals of Field1 and Field2): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  StartTiming(data);
  for(int i=0;i<n;i++){
      data.TmpField1.swap(data.TmpField2);
  }
  EndTiming(data);
  PRINT("testSWAP(swap addresses of Field1 and Field2): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  
  SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
  SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 1.);
  PRINT("testSWAP(integrals of Field1 and Field2): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  StartTiming(data);
  for(int i=0;i<n;i++){
      data.TmpField1.swap(data.TmpField2);
      data.TmpField2.swap(data.TmpField1);
  }
  EndTiming(data);
  PRINT("testSWAP(swap addresses of Field1 and Field2 and swap back): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);  
  
  SetField(data,data.TmpField1, data.DIM, data.EdgeLength, 0.);
  SetField(data,data.TmpField2, data.DIM, data.EdgeLength, 1.);
  PRINT("testSWAP(integrals of Field1 and Field2): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);
  StartTiming(data);
  for(int i=0;i<n;i++){
      data.TmpField1.swap(data.TmpField2);
      data.TmpField1.swap(data.TmpField2);
  }
  EndTiming(data);
  PRINT("testSWAP(swap addresses of Field1 and Field2 twice): Field1=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame)) + " Field2=" + to_string(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame)),data);    
}

void testFFT(datastruct &data){
  int diagQ = 1; if(diagQ) cout << "input field at diagonal locations" << endl; else cout << "input field at scattered locations" << endl;

  int dim = data.DIM, NP = data.EdgeLength, NP2=NP*NP, NP3=NP*NP*NP;
  double L = data.edge, Deltax=L/(double)data.steps, Deltak=2.*PI/L, k0 = -PI/Deltax;
  
  if(dim==1){
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      data.ComplexField.at(i).at(0) = EXP(-(x*x));
      data.ComplexField.at(i).at(1) = 0.;
      cout << "testFFT: " << i << " " << data.ComplexField.at(i).at(0) << " " << data.ComplexField.at(i).at(1) << endl;
    }
  }
  else if(dim==2){
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      for (int j=0;j<NP;j++){
	double y = -L/2.+(double)j*Deltax;
	data.ComplexField.at(i*NP+j).at(0) = EXP(-(x*x+y*y));
	data.ComplexField.at(i*NP+j).at(1) = 0.;
	if( (diagQ && j==i) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "testFFT: " << "[" << i << "][" << j << "] " << data.ComplexField.at(i*NP+j).at(0) << " " << data.ComplexField.at(i*NP+j).at(1) << endl;
      } 
    }
  }
  else if(dim==3){
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      for (int j=0;j<NP;j++){
	double y = -L/2.+(double)j*Deltax;
	for (int k=0;k<NP;k++){
	  double z = -L/2.+(double)k*Deltax;
	  data.ComplexField.at(i*NP2+j*NP+k).at(0) = EXP(-(x*x+y*y+z*z));//isotropic Gaussian
	  //data.ComplexField.at(i*NP2+j*NP+k).at(0) = EXP(-(x*x+2.*y*y+3.*z*z));//anisotropic Gaussian
	  data.ComplexField.at(i*NP2+j*NP+k).at(1) = 0.;
	  if( (diagQ && j==i && k==j) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP))) && k==NP-(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "testFFT: " << "[" << i << "][" << j << "][" << k << "] " << data.ComplexField.at(i*NP2+j*NP+k).at(0) << " " << data.ComplexField.at(i*NP2+j*NP+k).at(1) << endl;
	}
      } 
    }
  }
  
  cout << "compute fft" << endl;
  fftParallel(FFTW_FORWARD, data);
  
  if(dim==1){
    for (int i=0;i<NP;i++){
      double k = k0+(double)i*Deltak;
      cout << "testFFT: " << i << " " << data.ComplexField.at(i).at(0) << " " << data.ComplexField.at(i).at(1) << " " << SQRTPI*EXP(-k*k/4.) << endl;
    }
  }
  else if(dim==2){  
    for (int i=0;i<NP;i++){
      for (int j=0;j<NP;j++){
	double kx = k0+(double)i*Deltak, ky = k0+(double)j*Deltak;
	if( (diagQ && j==i) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "testFFT: " << "[" << i << "][" << j << "] " << data.ComplexField.at(i*NP+j).at(0) << " " << data.ComplexField.at(i*NP+j).at(1) << " " << PI*EXP(-(kx*kx+ky*ky)/4.) << endl;
      } 
    }
  }
  else if(dim==3){  
    for (int i=0;i<NP;i++){
      for (int j=0;j<NP;j++){
	for (int k=0;k<NP;k++){
	  double kx = k0+(double)i*Deltak, ky = k0+(double)j*Deltak, kz = k0+(double)k*Deltak;
	  if( (diagQ && j==i && k==j) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP))) && k==NP-(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "testFFT: " << "[" << i << "][" << j << "][" << k << "] " << data.ComplexField.at(i*NP2+j*NP+k).at(0) << " " << data.ComplexField.at(i*NP2+j*NP+k).at(1) << " " << pow(PI,1.5)*EXP(-(kx*kx+ky*ky+kz*kz)/4.) << endl;
	  //if( (diagQ && j==i && k==j) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP))) && k==NP-(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "[" << i << "][" << j << "][" << k << "] " << data.ComplexField.at(i*NP2+j*NP+k).at(0) << " " << data.ComplexField.at(i*NP2+j*NP+k).at(1) << " vs " << pow(PI,1.5)*EXP(-(kx*kx/4.+ky*ky/8.+kz*kz/12.))/sqrt(6.) << " 0" << endl;
	}
      } 
    }
  }
  
  cout << "compute inv_fft" << endl;
  fftParallel(FFTW_BACKWARD, data);

  if(dim==1){
    double check=0.;
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      check += pow(data.ComplexField.at(i).at(0) - EXP(-(x*x)),2.);
      check += pow(data.ComplexField.at(i).at(1) - 0.,2.);
      cout << i << " " << data.ComplexField.at(i).at(0) << " " << data.ComplexField.at(i).at(1) << endl;
    }
    cout << "check: " << sqrt(check/(double)NP) << " = MachinePrecision ?" << endl;
  }
  else if(dim==2){
    double check=0.;
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      for (int j=0;j<NP;j++){
	double y = -L/2.+(double)j*Deltax;
	check += pow(data.ComplexField.at(i*NP+j).at(0) - EXP(-(x*x+y*y)),2.);
	check += pow(data.ComplexField.at(i*NP+j).at(1) - 0.,2.);
	if( (diagQ && j==i) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "[" << i << "][" << j << "] " << data.ComplexField.at(i*NP+j).at(0) << " " << data.ComplexField.at(i*NP+j).at(1) << endl;
      }
    }
    cout << "check: " << sqrt(check/(double)NP2) << " = MachinePrecision ?" << endl;
  }
  else if(dim==3){
    double check=0.;
    for (int i=0;i<NP;i++){
      double x = -L/2.+(double)i*Deltax;
      for (int j=0;j<NP;j++){
	double y = -L/2.+(double)j*Deltax;
	for (int k=0;k<NP;k++){
	  double z = -L/2.+(double)k*Deltax;
	  check += pow(data.ComplexField.at(i*NP2+j*NP+k).at(0) - EXP(-(x*x+y*y+z*z)),2.);
	  check += pow(data.ComplexField.at(i*NP2+j*NP+k).at(1) - 0.,2.);
	  if( (diagQ && j==i && k==j) || (!diagQ && j==(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP))) && k==NP-(i+1)*(int)(pow(0.75*(double)NP,log( (double)NP/(double)(i+1) ) / log(0.75*(double)NP)))) ) cout << "[" << i << "][" << j << "][" << k << "] " << data.ComplexField.at(i*NP2+j*NP+k).at(0) << " " << data.ComplexField.at(i*NP2+j*NP+k).at(1) << endl;
	}
      }
    }
    cout << "check: " << sqrt(check/(double)NP3) << " = MachinePrecision ?" << endl;
  }  
  
  FillComplexField(1,data);
  //data.MONITOR = true;
  StartTiming(data);
  double test;
  for(int i=0;i<=100;i++){
    test = 0.; for(int j=0;j<data.GridSize;j++) test += data.ComplexField[j][0]; cout << "testFFT: " << test << endl;
    CopyComplexFieldToReal(data,data.TmpField1, data.ComplexField, 0, data.DIM, data.EdgeLength);
    test = 0.; for(int j=0;j<data.GridSize;j++) test += data.TmpField1[j]; cout << "testFFT: " << test << endl;
    fftParallel(FFTW_FORWARD,data);
    fftParallel(FFTW_BACKWARD,data);
    CopyComplexFieldToReal(data,data.TmpField2, data.ComplexField, 0, data.DIM, data.EdgeLength);
    if(i==0 || i%10==0 || i==100) PRINT("testFFT(" + to_string(i) + "): " + to_string_with_precision(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField1, data.frame),16) + " " + to_string_with_precision(Integrate(data.ompThreads,data.method, data.DIM, data.TmpField2, data.frame),16),data);
  }
  EndTiming(data);
  //data.MONITOR = false;
  //usleep(10*sec);
}

void report_num_threads(int level){
    #pragma omp single
    {
         printf("Level %d: number of threads in the team - %d\n", level, omp_get_num_threads());
    }
}

void process_idle_threads(vector<int> &thread_active) {

  #pragma omp parallel num_threads((int)sqrt((double)omp_get_max_threads()))
  {
    int inner_thread_id = omp_get_thread_num();

    #pragma omp for schedule(static)
    for (int i = 0; i < 1000; i++) {
      // Simulate computation using the idle threads
      simulate_work(i * 100000, (i + 1 + inner_thread_id) * 100000);
    }

    // Count idle threads of the outer loop
    int idle_threads = 0;
    #pragma omp critical
    {
      for (int t = 0; t < thread_active.size(); t++) {
        if (thread_active[t] == 0) idle_threads++;
      }
      printf("Inner thread %d: %d outer threads are idle\n", inner_thread_id, idle_threads);
    }
  }
}

void testOMP(int n, datastruct &data){

  	int max_threads = omp_get_max_threads();
  	cout << "omp_get_nested() " << omp_get_nested() << endl;
  	string max_threads_str = to_string(omp_get_max_threads());
    cout << "max_threads_str.c_str() " << max_threads_str.c_str() << endl;
  	const char* value;
    value = std::getenv("SUNW_MP_MAX_POOL_THREADS");
    cout << "SUNW_MP_MAX_POOL_THREADS: " << value << endl;
    value = std::getenv("SUNW_MP_MAX_NESTED_LEVELS");
    cout << "SUNW_MP_MAX_NESTED_LEVELS: " << value << endl;

    //omp_set_dynamic(0);

      int total_iterations = 1000;  // Total iterations for the outer loop
      vector<int> thread_active(max_threads,0);  // Track thread activity (0 = idle, 1 = active)

      // Outer parallel region
      #pragma omp parallel shared(thread_active) num_threads((int)sqrt((double)omp_get_max_threads()))
      {
        // Outer loop with static scheduling
        #pragma omp for schedule(static)
        for (int i = 0; i < total_iterations; i++) {
          // Mark thread as active
          int thread_id = omp_get_thread_num();
          thread_active[thread_id] = 1;

          // Simulate work in the outer loop
          if (i % 10 == 0) {
            simulate_work(0, 1000*(i+1)); // Simulate a longer task
          }

          // Call the nested function
          process_idle_threads(thread_active);

          // Mark thread as idle after completing work
          thread_active[thread_id] = 0;

          // #pragma omp critical
          // {
          //   printf("---> Outer thread %d marked idle\n", thread_id);
          // }
        }
      }



      //       int threads_per_level = (int)(0.5*(double)omp_get_max_threads());
      //       #pragma omp parallel num_threads(threads_per_level)
      //       {
      //         #pragma omp for schedule(dynamic)//Outer Loop
      //         for(...){
      //           #pragma omp parallel num_threads(threads_per_level)
      //           {
      //             #pragma omp for schedule(dynamic)
      //             for(...){
      //
      //             }
      //           }
      //         }
      //       }

      // int max_threads_outer_loop = max_threads / 2;
      // std::cout << "Maximum available threads: " << max_threads << std::endl;
      // vector<vector<double>> res(max_threads_outer_loop);
      // // Parallel region for the outer loop
      // #pragma omp parallel num_threads(max_threads_outer_loop) // Use half of the available threads for the outer loop
      // {
      //   int outer_thread_id = omp_get_thread_num();
      //   int outer_threads = omp_get_num_threads();
      //
      //   // Display the number of threads used in the outer parallel region
      //   #pragma omp single
      //   {
      //     std::cout << "Outer region threads: " << outer_threads << std::endl;
      //   }
      //
      //   #pragma omp for schedule(dynamic)
      //   for (int i = 0; i < max_threads_outer_loop; ++i) { // Outer loop
      //     res[i].resize(10+10*max_threads,0.);
      //     #pragma omp critical
      //     {
      //     	std::cout << "Outer loop iteration " << i << " handled by thread " << outer_thread_id << std::endl;
      //     }
      //     // Inner parallel region with remaining threads
      //     #pragma omp parallel num_threads(max_threads - outer_threads) // Use remaining threads for inner loop
      //     {
      //       int inner_thread_id = omp_get_thread_num();
      //       int inner_threads = omp_get_num_threads();
      //
      //       // Display the number of threads used in the inner parallel region
      //       #pragma omp single
      //       {
      //         std::cout << "  Inner region threads: " << inner_threads << " (==" << max_threads - outer_threads << "?)" << std::endl;
      //       }
      //
      //       #pragma omp for schedule(dynamic)
      //       for(int j=0;j<10+10*omp_get_thread_num();j++) { // Inner loop
      //         #pragma omp critical
      //         {
      //         	std::cout << "    Inner loop iteration " << j << " handled by thread " << inner_thread_id << std::endl;
      //         }
      //          	res[i][j] = HG1F2(5./3.,4./3.,8./3,100.+1./(double)(j+1+omp_get_thread_num()));
      //       }
      //     }
      //   }
      // }
      // for (int i = 0; i < max_threads_outer_loop; ++i) cout << accumulate(res[i].begin(),res[i].end(),0.) << endl;
      // usleep(3000000);
      //
      //


    // #pragma omp parallel
    // {
    //   #pragma omp for
    //   for (int i = 0; i < n; i++) {
    //     int num_threads = omp_get_max_threads();
    //     std::vector<double> res(num_threads, 0.0);
    //
    //     #pragma omp taskloop
    //     for (int t = 0; t < num_threads; t++) {
    //       if (t == 0) {
    //         #pragma omp critical
    //         std::cout << "omp_get_num_threads() " << omp_get_num_threads() << std::endl;
    //       }
    //       for (int j = 0; j < 10; j++) {
    //         res[t] += HG1F2( 5.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, 100.0 + 1.0 / ((double)(j + 1 + t)) );
    //       }
    //     }
    //
    //     #pragma omp critical
    //     std::cout << partial_vec_to_str_with_precision(res, 0, 4, 16) << std::endl;
    //   }
    // }



    // #pragma omp parallel
    // {
    //   	#pragma omp for
    // 	for(int i=0;i<n;i++){
    //       	vector<double> res(omp_get_max_threads(),0.);
    //       	#pragma omp parallel
    //       	{
    //         	cout << "omp_get_num_threads() " <<  omp_get_num_threads() << endl;
    //         	#pragma omp taskloop
    //         	for(int i=0;i<omp_get_max_threads();i++){
    //           		if(i==0) cout << "omp_get_thread_num()  " << omp_get_thread_num() << endl;
    //           		for(int j=0;j<10;j++) res[i] += HG1F2(5./3.,4./3.,8./3,100.+1./(double)(j+1+omp_get_thread_num()));
    //         	}
    //       	}
    //       	cout << partial_vec_to_str_with_precision(res,0,4,16) << endl;
    // 	}
    // }

/*
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(5)
    {
        report_num_threads(1);
        #pragma omp parallel num_threads(2)
        {
            report_num_threads(2);
            #pragma omp parallel num_threads(2)
            {
                report_num_threads(3);
            }
        }
    }
  
  int outer = 2; cout << "outer " << outer << endl;
  int inner = (data.ompThreads-2)/outer; cout << "inner " << inner << endl;

  StartTiming(data);
  vector<double> res(2);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<outer;i++){
    cout << i << endl;
    //report_num_threads(1);
    int size = 10000000;
    vector<double> out(size);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int j=0;j<size/2;j++){
      //if(i==0 &&j==0) report_num_threads(2);
      out[j] = data.RN(data.MTGEN);
      out[size-1-j] = -out[j];
    }
    res[i] = out[0]+out[size-1];
  }
  cout << "Nested=true: " << vec_to_str(res) << endl;
  EndTiming(data);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<outer;i++){
    cout << i << endl; //if(i==0) report_num_threads(1);
    int size = 10000000;
    vector<double> out(size);
    //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int j=0;j<size/2;j++){
      out[j] = data.RN(data.MTGEN);
      out[size-1-j] = -out[j];
    }
    res[i] = out[0]+out[size-1];
  }
  cout << "Nested=false: " << vec_to_str(res) << endl;
  EndTiming(data);  
  //usleep(10*sec);*/
  
/*
In opt.cpp:
Assign proper number of threads in second omp level, depending on threads in first omp level:
vector<int> threadsInLevel(2)
  */

//   StartTiming(data);
//   //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<data.S;s++){
//     GetDensity(s,data);
//     data.TmpAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
//   }
//   EndTiming(data);//0.43sec for S=2 steps=1024 on 10 cores
//   
//   StartTiming(data);
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<data.S;s++){
//     GetDensity(s,data);
//     data.TmpAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, data.Den[s], data.frame);
//   }
//   EndTiming(data);//1.56sec for S=2 steps=1024 on 10 cores -> be careful when calling parallelized routines within omp loops!!!
//   
//   vector<double> AV(n*n),AV2(n*n);
//   
//   StartTiming(data);
//   //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<3;s++){
//     //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int ss=0;ss<3;ss++){
//       //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//       for(int sss=0;sss<3;sss++){
// 	
// 	for(int i=0;i<n;i++){
// 	double val;
// 	  for(int j=0;j<n;j++){
// 	    for(int dummy=0;dummy<20;dummy++){
// 	      val = (double)(dummy+i*i+i)/1000.;
// 	      int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
// 	    }
// 	  }
// 	}
// 	
//       }
//     }
//   }
//   PRINT("testOMP: " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV, data.frame)) + " " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV2, data.frame)),data);
//   EndTiming(data);//9sec on 10 cores
//   
//   StartTiming(data);
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<3;s++){
//     //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int ss=0;ss<3;ss++){
//       //#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//       for(int sss=0;sss<3;sss++){
// 	
// 	for(int i=0;i<n;i++){
// 	double val;
// 	  for(int j=0;j<n;j++){
// 	    for(int dummy=0;dummy<20;dummy++){
// 	      val = (double)(dummy+i*i+i)/1000.;
// 	      int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
// 	    }
// 	  }
// 	}
// 	
//       }
//     }
//   }
//   PRINT("testOMP: " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV, data.frame)) + " " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV2, data.frame)),data);
//   EndTiming(data);//3sec on 10 cores
// 
//   StartTiming(data);
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<3;s++){
//     #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//     for(int ss=0;ss<3;ss++){
//       #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//       for(int sss=0;sss<3;sss++){
// 	
// 	for(int i=0;i<n;i++){
// 	double val;
// 	  for(int j=0;j<n;j++){
// 	    for(int dummy=0;dummy<20;dummy++){
// 	      val = (double)(dummy+i*i+i)/1000.;
// 	      int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
// 	    }
// 	  }
// 	}
// 	
//       }
//     }
//   }
//   PRINT("testOMP: " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV, data.frame)) + " " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV2, data.frame)),data);
//   EndTiming(data);//3sec on 10 cores -> OMP cannot handle nested loops
//   
//   StartTiming(data);
//   #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
//   for(int s=0;s<3;s++){
//     passDoubleOMPloop(n,AV,AV2);
//   }
//   PRINT("testOMP: " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV, data.frame)) + " " + to_string(Integrate(data.ompThreads,data.method, data.DIM, AV2, data.frame)),data);
//   EndTiming(data);//3sec on 10 cores -> OMP cannot assign additional threads in parallelized functions called within an omp loop!!!   
//   
//   StartTiming(data); FillComplexField(1,data); EndTiming(data);
//   StartTiming(data); FillComplexField(2,data); EndTiming(data);
//   StartTiming(data); FillComplexField(3,data); EndTiming(data);
//   StartTiming(data); FillComplexField(4,data); EndTiming(data);
//   StartTiming(data); FillComplexField(5,data); EndTiming(data);
//   
//   StartTiming(data); testFFT(data); EndTiming(data);
 
}

void testCPU(int n, datastruct &data){
  PRINT("testCPUspeed; summation of n=" + to_string(n) + "^4 simple terms:",data);
  double res = 0.;
  StartTiming(data);
  for(int i=0;i<n*n*n*n;i++){
    res += (double)i;
  }
  EndTiming(data);
  res = 0.;
  StartTiming(data);
  for(int i=0;i<n*n*n*n;i++){
    res += 1./sqrt(1.+(double)i);
  }
  EndTiming(data);
  res = 0.;
  StartTiming(data);
  for(int i=0;i<n*n*n*n;i++){
    res += gsl_sf_bessel_I0(log(1.+(double)i));
  }
  EndTiming(data);  
  
  PRINT("testCPUspeed(n=" + to_string(n) + "^3):",data);
  vector<double> AV(n*n),AV2(n*n);
  
  PRINT("2D loop, nested; omp, chopped",data); StartTiming(data);
  int CHOP = data.ompThreads; cout << "testCPU: " << CHOP << endl;
  int tasksPerThread=(int)((double)n/(double)CHOP);
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int chop=0;chop<CHOP;chop++){
    int istart = chop*tasksPerThread, iend = (chop+1)*tasksPerThread;
    for(int i=istart;i<iend;i++){
      double val;
      for(int j=0;j<n;j++){
	for(int dummy=0;dummy<20;dummy++){
	  val = (double)(dummy+i*i+i)/1000.;
	  int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
	}
      }
    }
  }
  EndTiming(data);  

  PRINT("2D loop, nested; omp",data); StartTiming(data);
  //...probably does the chopping automatically...
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<n;i++){
    double val;
    for(int j=0;j<n;j++){
      for(int dummy=0;dummy<20;dummy++){
	val = (double)(dummy+i*i+i)/1000.;
	int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
      }
    }
  }
  EndTiming(data);  
  PRINT("2D loop, nested; serial",data); StartTiming(data);
  for(int i=0;i<n;i++){
    double val;
    for(int j=0;j<n;j++){
      for(int dummy=0;dummy<20;dummy++){
	val = (double)(dummy+i*i+i)/1000.;
	int N = i*n+j; AV[N] = val; AV2[N] = 2.*val;
      }
    }
  }
  EndTiming(data);
  
}

void testBoxBoundaryQ(datastruct &data){
  cout << "testBoxBoundaryQ..." << endl;
  bool success = true;
  for(int i=0;i<data.EdgeLength;i++){
    for(int j=0;j<data.EdgeLength;j++){
      for(int k=0;k<data.EdgeLength;k++){
	int index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+k;
	bool test = BoxBoundaryQ(index,data);
	if(!test && (i==0 || i==data.steps || j==0 || j==data.steps || k==0 || k==data.steps)){ success = false; cout << i << " " << j << " " << k << " " << test << endl; }
	else if(test && (i!=0 && i!=data.steps && j!=0 && j!=data.steps && k!=0 && k!=data.steps)){ success = false; cout << i << " " << j << " " << k << " " << test << endl; }
      }
    }
  }
  if(success) cout << "...ok!" << endl; else cout << "...not ok!!!!!!" << endl;
}

void testKD(datastruct &data){
  	struct KDintegrationParams KDip;
    KDip.num_outer_threads = 1;
    KDip.num_inner_threads = data.ompThreads;

  	//BEGIN USER INPUT
  	double scale = 1.0e+6;
    //a,b in [0,1] for A in [-inf,inf] and B in [0,inf]
    //a=const
    // double amin = GetKDa(1.0e+7,scale);//A=-10^7,-10^6,-10^5,0,10^5,10^6,10^7?
    // double amax = amin;
    // double bmin = 0.;
    // double bmax = 1.;
    //b=const
   //  double bmin = GetKDb(1.0e+7,scale);//B=10^-5,10^-4,10^-1,10^0,...10^7
   //  double bmax = bmin;
  	// double amin = 0.;
  	// double amax = 1.;
    //zoom: a=const
    // double amin = GetKDa(0.,scale);
    // double amax = amin;
    // double bmin = 0.;//0.5;//0.9999;//
    // double bmax = 0.001;//0.505;//0.999901;//
    // double amin = GetKDa(3.0e+4,scale);
    // double amax = amin;
    // double bmin = 0.5;
    // double bmax = 0.505;
    //zoom: b=const
    // double amin = 0.495;
    // double amax = 0.505;
    // double bmin = GetKDb(1.0e+4,scale);//B=10^-4,10^2,10^3,10^4,10^7
    // double bmax = bmin;
    double amin = 0.9999;
    double amax = 0.99995;
    double bmin = GetKDb(1.0e+7,scale);//B=10^-4,10^2,10^5,10^6,10^7
    double bmax = bmin;
  	int points = 10000;
  	int NumSamples = points/100;
    double BoundarySamplingFreq = 30.;//the larger the more points near the interval borders [0,1] of a & b
    //END USER INPUT

    cout << "testKD:" << endl;

  	double RELERR = data.InternalAcc, ABSERR;

    // double amin = GetKDa(1.,scale);
    // double amax = amin;
    // double bmin = 0.;
    // double bmax = 1.;

    // StartTimer("K1",data);
    // cout << GetKDA(0.2,scale) << " " << GetKDB(1.0e-10,scale) << " " << KD(1,GetKDA(0.2,scale),GetKDB(1.0e-10,scale),ABSERR,RELERR,KDip) << endl;
    // cout << KD(1,0.,GetKDB(0.001,scale),ABSERR,RELERR,KDip) << endl;
    // cout << KD(1,0.,0.003372003904,ABSERR,RELERR,KDip) << endl;
    // EndTimer("K1",data);
    // StartTimer("K3",data);
    // cout << GetKDA(0.2,scale) << " " << GetKDB(1.0e-10,scale) << " " << KD(3,1.0e+7,1.0e+7,ABSERR,RELERR,KDip) << endl;
    // cout << KD(3,0.,GetKDB(0.001,scale),ABSERR,RELERR,KDip) << endl;
    // EndTimer("K3",data);
    // SleepForever();

    double a,b,A,B,res,pmin,pmax;
    string FileNameSuffix = "", TimeStamp = YYYYMMDD() + "_" + hhmmss();
    if(bmax-bmin<MP){//b=const
      	cout << "@ b = " << bmin << " <-> B = " << GetKDB(bmin,scale) << endl;
      	pmin = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*(1./(double)points-0.5)) );
      	pmax = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*(0.5)) );
        cout << "        amin = " << pmin << " <-> minA = " << GetKDA(pmin,scale) << endl;
        cout << "        amax = " << pmax << " <-> maxA = " << GetKDA(pmax,scale) << endl;
        FileNameSuffix += "_b=" + to_string_with_precision(bmin,14);
    }
    else{//a=const
      	cout << "@ a = " << amin << " <-> A = " << GetKDA(amin,scale) << endl;
      	pmin = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*(1./(double)points-0.5)) );
      	pmax = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*(0.5)) );
        cout << "        bmin = " << pmin << " <-> minB = " << GetKDB(pmin,scale) << endl;
        cout << "        bmax = " << pmax << " <-> maxB = " << GetKDB(pmax,scale) << endl;
        FileNameSuffix += "_a=" + to_string_with_precision(amin,14);
    }

  	for(int d=1;d<=3;d++){
      	StartTimer("testKD",data);
      	string FileName = "mpDPFT_testK" + to_string(d) + "_" + TimeStamp + FileNameSuffix + ".dat";
        vector<vector<double>> Out(0);
        vector<double> out(2);
    	for(int i=1;i<=points;i++){
          	if(amax-amin<MP) a = amin;
          	else a = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*((double)i/(double)points-0.5)) );
            A = GetKDA(a,scale);
            if(ABS(a-GetKDa(A,scale))>MP) cout << "testKD Error " << a << " " << GetKDa(A,scale) << endl;
      		for(int j=1;j<=points;j++){
              	if(bmax-bmin<MP) b = bmin;
              	else b = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*((double)j/(double)points-0.5)) );
                B = GetKDB(b,scale);
                if(ABS(b-GetKDb(B,scale))>MP) cout << "testKD Error " << b << " " << GetKDb(B,scale) << " " << endl;
                //cout << d << " " << i << " " << j << " " << a << " " << b << " " << A << " " << B << " " << res << endl;
                res = KD(d,A,B,ABSERR,RELERR,KDip);
                //cout << "-> K" + to_string(d) + " = "<< res << endl;
                if(bmax-bmin<MP) break; //b=const
                else{
                  	bool collect = false;
                  	if(Out.size()==0) collect = true;
                    else if(b>Out.back()[0]+MP) collect = true;//ensure enough separation for alglib interpolator
                    if(collect) Out.push_back({b,res});
                    if(j%(points/10)==0) cout << d << " " << j << " " << vec_to_str_with_precision(Out.back(),16) << endl;
                }
      		}
            if(amax-amin<MP) break; //a=const
            else{
              	bool collect = false;
              	if(Out.size()==0) collect = true;
              	else if(a>Out.back()[0]+MP) collect = true;//ensure enough separation for alglib interpolator
              	if(collect) Out.push_back({a,res});
                if(i%(points/10)==0) cout << d << " " << i << " " << vec_to_str_with_precision(Out.back(),16) << endl;
            }
        }
        MatrixToFile(Out,FileName,16);

        cout << " interpolate and validate in [" << pmin << "," << pmax << "]" << endl;
        spline1dinterpolant InterpolKD = GetSpline1D(FileName,true);
        double absvar = 0., relvar = 0., AbsDiffmax = 0., RelDiffmax = 0.;
        for(int i=0;i<NumSamples;i++){
          	double param = pmin + 0.5 * (pmax-pmin) * ( 1. + tanh(BoundarySamplingFreq*(data.RNpos(data.MTGEN)-0.5)) );
            double kd, ip = spline1dcalc(InterpolKD,param);
          	if(amax-amin<MP) kd = KD(d,GetKDA(amin,scale),GetKDB(param,scale),ABSERR,RELERR,KDip);
          	else kd = KD(d,GetKDA(param,scale),GetKDB(bmin,scale),ABSERR,RELERR,KDip);
            double absdiff = ABS(kd-ip), reldiff = RelDiff(kd,ip);
          	if(absdiff>AbsDiffmax && reldiff>RelDiffmax){//if absdiff is small, then reldiff does not matter, and if reldiff is small, then absdiff does not matter.
              	AbsDiffmax = absdiff;
                RelDiffmax = reldiff;
              	cout << "param = " << param << " K" + to_string(d) + " = " << kd << " Interpol = " << ip << " absdiff = " << absdiff << " reldiff = " << reldiff << endl;
            }
            absvar += absdiff*absdiff;
            relvar += reldiff*reldiff;
        }
        absvar = sqrt(absvar/(double)NumSamples);
        relvar = sqrt(relvar/(double)NumSamples);
        cout << " absvar = " << absvar << endl << " relvar = " << relvar << endl << endl;
        EndTimer("testKD",data);
  	}

  	//does not run in the background - contrary to intention...
  	// Absolute path to the folder where the script is located
  	const char* folder_path = "/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts/";
    // Change the directory and then execute the script
    std::string command = "cd " + std::string(folder_path) + " && ./mpScript_testKD.sh &";
    // Call the bash script
    if (system(command.c_str()) != 0) {
      std::cerr << "Error: The script failed to execute properly." << std::endl;
    } else {
      std::cout << "Script executed successfully!" << std::endl;
    }

    //...but this alternative does not work at all
    // pid_t pid = fork(); // Create a new process (fork)
    // if (pid == 0) {// Child process: pid == 0
    //   execl("./home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts/mpScript_testKD.sh", "mpScript_testKD.sh", nullptr); // Execute the script
    //   std::cerr << "Failed to execute the script\n"; // Only reached if execl fails
    // }
}

void testFitFunction(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr){// where x is a position on X-axis and c is adjustable parameter
  datastruct *data = (datastruct *)ptr;
  //cout << data->Abundances[0] << endl;

  //func = 0.5*c[0]*pow(ABS(x[0]),c[0]);
  func = 0.5*c[0]*(1.+pow(ABS(x[0]),c[0])); cout << c[0] << " " << func << endl;
}

// void testALGfit(datastruct &data){
//
//     double X[1][2]; X[0][0] = 0;
//     real_2d_array x; x.setcontent(1,2,&(X[0][0]));
//     real_1d_array y = "[0]";//Target values / refData
//     real_1d_array c = "[0.3]";
//     real_1d_array bndl = "[0.0]";
//     real_1d_array bndu = "[1000.0]";
//
//     double epsx = 0;
//     ae_int_t maxits = 0;
//     ae_int_t info;
//     lsfitstate state;
//     lsfitreport rep;
//     double diffstep = 1;
//
//     lsfitcreatef(x, y, c, diffstep, state);
//     lsfitsetbc(state, bndl, bndu);
//     lsfitsetcond(state, epsx, maxits);
//     lsfitfit(state, testFitFunction, NULL, (void*)&data);
//     lsfitresults(state, info, c, rep);
//     printf("%d\n", int(info));
//     printf("%s\n", c.tostring(1).c_str());
//
// //     int L = 3;
// //     double X[L][2]; for(int i=0;i<L;i++){ X[i][0] = -1.+(double)i; }
// //     real_2d_array x; x.setcontent(L,2,&(X[0][0]));
// //     real_1d_array y = "[2,0,2]";//Target values / refData
// //     real_1d_array c = "[0.3]";
// //     real_1d_array bndl = "[0.0]";
// //     real_1d_array bndu = "[1000.0]";
// //
// //     double epsx = 0.000001;
// //     ae_int_t maxits = 0;
// //     ae_int_t info;
// //     lsfitstate state;
// //     lsfitreport rep;
// //     double diffstep = 0.0001;
// //
// //     lsfitcreatef(x, y, c, diffstep, state);
// //     lsfitsetbc(state, bndl, bndu);
// //     lsfitsetcond(state, epsx, maxits);
// //     lsfitfit(state, testFitFunction, NULL, (void*)&data);
// //     lsfitresults(state, info, c, rep);
// //     printf("%d\n", int(info));
// //     printf("%s\n", c.tostring(1).c_str());
//
//     //usleep(10*sec);
// }

void testisfinite(void){
    //std::cout << std::boolalpha
              std::cout << "isfinite(NaN) = " << std::isfinite(NAN) << '\n'
              << "isfinite(-NaN) = " << std::isfinite(-NAN) << '\n'
              << "isfinite(Inf) = " << std::isfinite(INFINITY) << '\n'
              << "isfinite(-Inf) = " << std::isfinite(-INFINITY) << '\n'
              << "isfinite(0.0) = " << std::isfinite(0.0) << '\n'
              << "isfinite(exp(800)) = " << std::isfinite(std::exp(800.)) << '\n'
              << "isfinite(exp(-800)) = " << std::isfinite(std::exp(-800.)) << " " << exp(-800.) << '\n'              
              << "isfinite(DBL_MIN/2.0) = " << std::isfinite(DBL_MIN/2.0) << '\n';  
    usleep(1*sec);
}

void testPCA(datastruct &data){//be aware: inputMatrix is internally accessed AND MODIFIED!!!
  
  //begin USERINPUT
  int pcaType = 1;//--- 0: full pca --- 1: truncated pca
  int DIM = 300, NumVecs = (int)3.33e+6, ReduceToDim = 2;// 10^9 data values ->40 seconds with pcaType==1
  double eps = 1.0e-3;
  ae_int_t maxits = 0;
  //end USERINPUT
  
  vector<vector<double>> pca(DIM); for(int eigvec=0;eigvec<DIM;eigvec++) pca[eigvec].resize(DIM);  
  real_2d_array inputMatrix; inputMatrix.setlength(NumVecs,DIM);
  alglib::ae_int_t info;// integer status code
  alglib::real_1d_array eigValues;// scalar values that describe variances along each eigenvector
  alglib::real_2d_array eigVectors;// targeted orthogonal basis as unit eigenvectors  
  
  for(int i=0;i<NumVecs;i++) for(int j=0;j<DIM;j++) if(j==0 || j==1) inputMatrix[i][j] = (double)i+0.1*data.RN(data.MTGEN); else inputMatrix[i][j] = 0.001*data.RN(data.MTGEN);

  StartTiming(data);
  if(pcaType==0) pcabuildbasis(inputMatrix, NumVecs, DIM, eigValues, eigVectors);
  else if(pcaType==1) pcatruncatedsubspace(inputMatrix, NumVecs, DIM, ReduceToDim, eps, info, eigValues, eigVectors);
  EndTiming(data);

  for(int eigvec=0;eigvec<=ReduceToDim+1;eigvec++){
    for(int coor=0;coor<DIM;coor++) pca[eigvec][coor] = eigVectors[coor][eigvec];
    cout << "testPCA eigenvector" << eigvec << " (pcatruncatedsubspace) = {" << vec_to_str(pca[eigvec]) << "}" << endl;
  }
  usleep(100*sec); 

}

double testRBF_f(double x, double y, vector<vector<double>> &params){
  double res = 0.;
  for(int i=0;i<params.size();i++){
    res += params[i][3]*EXP(-0.5*((x-params[i][0])*(x-params[i][0])+(y-params[i][1])*(y-params[i][1]))/params[i][2]);
  }
  return res;
}

void testRBF(datastruct &data){
  
    int DIM = 2;
    int funcDIM = 1;//scalar function 
    int NumVecs = 100;//10;   
    int Nx = 10, Ny = 10;
    double SpatialExtent = 10., averageDistance = SpatialExtent/pow((double)NumVecs,1./((double)DIM)), baseRadius = sqrt(2.)*SpatialExtent,/*baseRadius = 4.0*averageDistance,*/ v;
    double smoothing = 1.0e-3;//default smoothing choice: 1.0e-4...1.0e-3 --- put zero to omit smoothing
    int NumLayers = (int)(log(2.*baseRadius/averageDistance)/log(2.))+2;
    bool zeroAsymptote = false;
    cout << "testRBF, NumLayers = " << NumLayers << ", baseRadius = " << baseRadius << endl;    
    
    //test2
    vector<vector<double>> params(100);
    for(int i=0;i<params.size();i++){//100 Gaussians
      params[i].push_back(SpatialExtent*data.RN(data.MTGEN));//x_peak
      params[i].push_back(SpatialExtent*data.RN(data.MTGEN));//y_peak
      params[i].push_back(pow(0.1*SpatialExtent,2.));
      //params[i].push_back(pow(SpatialExtent*data.RN(data.MTGEN),2.));//variance
      params[i].push_back(data.RN(data.MTGEN));//amplitude
    }
    ofstream testRBF_function, testRBF_dataset, testRBF_interpol;
    testRBF_function.open("testRBF_function.dat");
    testRBF_dataset.open("testRBF_dataset.dat");
    testRBF_interpol.open("testRBF_interpol.dat");
    for(double x=-10.;x<10.;x+=0.1) for(double y=-10.;y<10.;y+=0.1) testRBF_function << x << " " << y << " " << testRBF_f(x,y,params) << "\n";
    testRBF_function.close();     

    rbfmodel model;
    rbfcreate(DIM, funcDIM, model);

    real_2d_array rawdataset, dataset;
    rawdataset.setlength(NumVecs,DIM+funcDIM);
    vector<double> xVec(NumVecs);
    vector<vector<double>> tmpdataset(0);
    for(int i=0;i<NumVecs;i++){    
      //test1
//       rawdataset[i][0] = floor(SpatialExtent*data.RNpos(data.MTGEN));//(double)(i);
//       rawdataset[i][1] = floor(SpatialExtent*data.RNpos(data.MTGEN));//floor(sqrt((double)(i)));//(double)(i);
//       rawdataset[i][2] = floor(10.+90.*data.RNpos(data.MTGEN));/*(double)(i);*/
      //test2
      rawdataset[i][0] = SpatialExtent*data.RN(data.MTGEN);
      rawdataset[i][1] = SpatialExtent*data.RN(data.MTGEN);
      rawdataset[i][2] = testRBF_f(rawdataset[i][0],rawdataset[i][1],params); 
      
      cout << to_string_with_precision(rawdataset[i][0],4) << " " << to_string_with_precision(rawdataset[i][1],4) << " " << to_string_with_precision(rawdataset[i][2],4) << endl;
      xVec[i] = rawdataset[i][0];
    }     
    cout << endl;
    vector<double> pt(3);
    for(auto i: sort_indices(xVec)){
      pt = {{rawdataset[i][0],rawdataset[i][1],rawdataset[i][2]}};
      bool AlreadyAddedQ = false;
      if(tmpdataset.size()>0){
	int previous = tmpdataset.size()-1;
	while(ABS(pt[0]-tmpdataset[previous][0])<MP){
	  if(ABS(pt[1]-tmpdataset[previous][1])<MP) AlreadyAddedQ = true;
	  if(previous==0 || AlreadyAddedQ) break;
	  else previous--;
	}
      }
      if(!AlreadyAddedQ){ tmpdataset.push_back(pt); cout << vec_to_str(pt) << endl; }
    }
    int NewNumVecs = tmpdataset.size();
    if(NewNumVecs!=NumVecs){
      cout << "duplicates removed, print 2D-matrix-indices and func values (number of samples " << NumVecs << "->" << NewNumVecs << ")" << endl;
      for(int i=0;i<NewNumVecs;i++) cout << vec_to_str(tmpdataset[i]) << endl;
    }
    dataset.setlength(NewNumVecs,DIM+funcDIM);
    for(int i=0;i<NewNumVecs;i++){   
      dataset[i][0] = tmpdataset[i][0];
      dataset[i][1] = tmpdataset[i][1];
      dataset[i][2] = tmpdataset[i][2];
      //test2
      testRBF_dataset << dataset[i][0] << " " << dataset[i][1] << " " << dataset[i][2] << "\n";
    }
    rbfsetpoints(model, dataset);

    rbfreport rep;
    rbfsetalgohierarchical(model, baseRadius, NumLayers, smoothing);
    if(zeroAsymptote) rbfsetzeroterm(model);
    else rbfsetconstterm(model);
    rbfbuildmodel(model, rep);
    
    //test1
//     for(int i=0;i<Nx;i++){
//       for(int j=0;j<Ny;j++) cout << (int)(rbfcalc2(model, (double)i, (double)j)) << " ";
//       //for(int j=0;j<Ny;j++) cout << to_string_with_precision(rbfcalc2(model, (double)i, (double)j),3) << " ";
//       cout << endl;
//     }    
//     cout << endl;
//     for(double x=1.;x<1000.;x*=2.) cout << x << " " << (int)(rbfcalc2(model, x, sqrt(x))) << " " << (int)(rbfcalc2(model, x, x)) << endl;
    
    //test2
    for(double x=-10.;x<10.;x+=0.1) for(double y=-10.;y<10.;y+=0.1) testRBF_interpol << x << " " << y << " " << rbfcalc2(model, x, y) << "\n";
    testRBF_dataset.close();
    testRBF_interpol.close();

    usleep(100*sec); 
}

void testcec14(datastruct &data, taskstruct &task){
	int dim = 10, funcID = 9;
	vector<double> f(1), x(dim);
  
	for(int i=0;i<dim;i++) x[i] = 1./((double)(1+i)); cec14_test_func(&x[0], &f[0], dim, 1, funcID, 0);  cout << std::setprecision(16) << "MIT: f = " << f[0] << endl;
	for(int i=0;i<dim;i++) x[i] = 0.0000000000000000; cec14_test_func(&x[0], &f[0], dim, 1, funcID, 0);  cout << std::setprecision(16) << "MIT: f = " << f[0] << endl;

	//cec14
	OPTstruct opt;
	SetDefaultOPTparams(opt);
	opt.reportQ = 1;
	opt.function = -2014+9;//custom objective function, defined in Plugin_OPT.cpp
	
	vector<vector<double>> optimizer(1);
	double optimum;
	opt.D = 20;
	opt.epsf = 1.0e-12;
	opt.threads = data.ompThreads;
	opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.;
	opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
	opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);    
	for(int i=0;i<opt.D;i++){
		opt.SearchSpaceLowerVec[i] = opt.SearchSpaceMin;
		opt.SearchSpaceUpperVec[i] = opt.SearchSpaceMax;
	}
	//opt.UpdateSearchSpaceQ = 2;

	//start PSO:
	SetDefaultPSOparams(opt);
	//begin adjust parameters
	opt.pso.runs = 100*opt.D;
	opt.pso.increase = 20.;//increase initial swarm size by a factor of X {in EXP(log(X)/((double)opt.pso.runs))} towards final run
	opt.pso.InitialSwarmSize = opt.D;
	//opt.pso.SwarmDecayRate = 1.7;
	InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
	opt.pso.VarianceCheck = 10*opt.D;
	//opt.pso.MaxLinks = 3;
	//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
	//opt.pso.elitism = 0.1;
	//opt.pso.PostProcessQ = 1;
	opt.pso.loopMax = (int)5.0e+5;
	//opt.pso.reseed = 0.5;
	//opt.pso.CoefficientDistribution = 1;
	//opt.pso.alpha = 0.;
	//opt.BreakBadRuns = 1;
	//end adjust parameters
	PSO(opt);
	optimizer[0] = opt.pso.bestx;
	optimum = opt.pso.bestf;
	//end PSO
	
	TASKPRINT(opt.control.str(),task,0);
	string FileName = "mpDPFT_ObjFunc_DFTe_QPot_bestf_" + to_string_with_precision(GetFuncVal(0,optimizer[0],opt.function,opt),16) + ".dat";
	MatrixToFile(optimizer,FileName,16);
	TASKPRINT("min{f} = " + to_string_with_precision(optimum,16) + " @ x = " + vec_to_str_with_precision(optimizer[0],16) + "\n --> stored in " + FileName,task,0);  
	  
}

void testOptimizers(datastruct &data, taskstruct &task){

  OPTstruct opt;
  SetDefaultOPTparams(opt);
  opt.reportQ = 1;
  opt.function = -100;//custom objective function, defined in Plugin_OPT.cpp
  
	//begin template
// 	vector<vector<double>> optimizer(1);
// 	double optimum;
// 	opt.D = 31;
// 	opt.epsf = 1.0e-12;
// 	opt.threads = data.ompThreads;
//  	opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;	
// 	//opt.SearchSpaceMin = -0.512; opt.SearchSpaceMax = 0.512;
//     opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
//     opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D); 	
// 	
// 	//start PSO:
// 	SetDefaultPSOparams(opt);
// 	//begin adjust parameters
// 	opt.pso.runs = 10000;
// 	opt.pso.increase = 100.;//increase initial swarm size by this factor toward final run
// 	opt.pso.SwarmDecayRate = 1.7;
// 	opt.pso.InitialSwarmSize = opt.D;
// 	InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
// 	opt.pso.VarianceCheck = max(10,min(50,opt.D));
// 	//opt.pso.MaxLinks = 3;
// 	//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
// 	//opt.pso.elitism = 0.1;
// 	//opt.pso.PostProcessQ = 1;
// 	opt.pso.loopMax = (int)1.0e+7;
// 	//opt.pso.reseed = 0.25;
// 	//opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
// 	//opt.pso.alpha = 0.;
// 	//opt.BreakBadRuns = true;
// 	opt.pso.TargetMinEncounters = opt.pso.runs;
// 	//end adjust parameters
// 	PSO(opt);
// 	optimizer[0] = opt.pso.bestx;
// 	optimum = opt.pso.bestf;
// 	//end PSO	
// 	
// 	TASKPRINT(opt.control.str(),task,0);
// 	string FileName = "mpDPFT_ObjFunc_bestf_" + to_string_with_precision(ObjFunc(optimizer[0],opt),16) + ".dat";
// 	MatrixToFile(optimizer,FileName,16);
// 	TASKPRINT("min{f} = " + to_string_with_precision(optimum,16) + " @ x = " + vec_to_str_with_precision(optimizer[0],16) + "\n --> stored in " + FileName,task,0);  
	//end template
	


}

void test1pExDFTN2(datastruct &data, taskstruct &task){
	//1pEx-DFT for atoms/ions, no frills
	OPTstruct opt;
	SetDefaultOPTparams(opt);
	opt.reportQ = 1;
	opt.function = -100;//custom objective function, defined in Plugin_OPT.cpp
  
	vector<vector<double>> optimizer(1);
	double optimum;
	int L=14;
	opt.D = 2*L;
	opt.epsf = 1.0e-7;
	opt.threads = data.ompThreads;
	opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 2.*PI;
    opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
    opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D); 	
	
	opt.ex = TASK.ex;
	opt.ex.FreeIndices.clear(); opt.ex.FreeIndices.resize(0); for(int i=0;i<L;i++) opt.ex.FreeIndices.push_back(i);
	opt.ex.Iabcd.clear(); opt.ex.Iabcd.resize(0);
	ReadVec("TabFunc_Hint.dat", opt.ex.Iabcd);
	cout << "test1pExDFTN2 " << opt.ex.Iabcd.size() << endl;	
	
	//start PSO:
	SetDefaultPSOparams(opt);
	//begin adjust parameters
	opt.pso.runs = 10;
	opt.pso.increase = 1.;//increase initial swarm size by this factor toward final run
	opt.pso.InitialSwarmSize = 10*opt.D;
	InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
	opt.pso.VarianceCheck = opt.D;
	//opt.pso.MaxLinks = 3;
	//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
	//opt.pso.elitism = 0.1;
	//opt.pso.PostProcessQ = 1;
	opt.pso.loopMax = (int)5.0e+5;
	//opt.pso.reseed = 0.5;
	//opt.pso.CoefficientDistribution = 1;
	//opt.pso.alpha = 0.;
	//opt.BreakBadRuns = 1;
	//end adjust parameters
	PSO(opt);
	optimizer[0] = opt.pso.bestx;
	optimum = opt.pso.bestf;
	//end PSO	
	
	TASKPRINT(opt.control.str(),task,0);
	string FileName = "mpDPFT_ObjFunc_DFTe_QPot_bestf_" + to_string_with_precision(ObjFunc(optimizer[0],opt),16) + ".dat";
	MatrixToFile(optimizer,FileName,16);
	TASKPRINT("min{f} = " + to_string_with_precision(optimum,16) + " @ x = " + vec_to_str_with_precision(optimizer[0],16) + "\n --> stored in " + FileName,task,0); 	
}

void testALGintegrator(datastruct &data, taskstruct &task){
	//ToDo
}

void testHydrogenicHint(datastruct &data, taskstruct &task){
	data.FLAGS.ChemistryEnvironment = 0;
	data.FLAGS.GetOrbitals = 0;
	data.FLAGS.InitializeDensities = 0;
	data.FLAGS.Export = 0;
	
	GetQuantumNumbers(TASK.ex);
	
	int L=14, NumTestElements = 100;
	
	bool SingularityCompensation = true;//false;//
	
	TASK.ex.Iabcd.clear(); TASK.ex.Iabcd.resize(0);
	ReadVec("TabFunc_Hint.dat", TASK.ex.Iabcd);
	
	vector<vector<int>> Indices(0);
	//pick random indices:
	Indices.resize(NumTestElements); for(int i=0;i<Indices.size();i++) for(int j=0;j<4;j++) Indices[i].push_back((int)((double)L*rn())); Indices[0] = {{0,0,0,0}};
	//all S states:
	//vector<int> s = {{0,1,5}}; for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++){ Indices.push_back({{s[i],s[j],s[k],s[l]}}); cout << vec_to_str(Indices[Indices.size()-1]) << endl; }
	
	for(int i=0;i<Indices.size();i++){
		int a = Indices[i][0];
		int b = Indices[i][1];
		int c = Indices[i][2];
		int d = Indices[i][3];
	
		fab(data.TmpComplexField,a,b,data,TASK);
		fab(data.ComplexField,c,d,data,TASK);
		
		vector<double> Ref(0), Imf(0);
		double ReSC, ImSC;//Ingredient -0.5*f''(k=0) for SingularityCompensation (SC)
		if(SingularityCompensation){
			Ref.resize(data.GridSize);
			Imf.resize(data.GridSize);
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){
				double r2 = Norm2(data.VecAt[i]);
				Ref[i] = r2*data.ComplexField[i][0];
				Imf[i] = r2*data.ComplexField[i][1];
			}
			ReSC = -0.5*Integrate(data.ompThreads, data.method, data.DIM, Ref, data.frame);
			ImSC = -0.5*Integrate(data.ompThreads, data.method, data.DIM, Imf, data.frame);
		}
		
		fftParallel(FFTW_FORWARD, data);
		double Ref0tilde = data.ComplexField[data.CentreIndex][0], Imf0tilde = data.ComplexField[data.CentreIndex][1];//Ingredient f0tilde for SC
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			double k2 = data.Norm2kVecAt[i];
			if(i!=data.CentreIndex){
				if(SingularityCompensation){
					double fac = EXP(-k2);
					data.ComplexField[i][0] -= Ref0tilde*fac;
					data.ComplexField[i][0] -= Imf0tilde*fac;
				}
				data.ComplexField[i][0] /= k2;
				data.ComplexField[i][1] /= k2;
			}
			else if(SingularityCompensation){
				data.ComplexField[i][0] += ReSC;
				data.ComplexField[i][1] += ImSC;
			}
			data.ComplexField[i][0] *= 4.*PI;
			data.ComplexField[i][1] *= 4.*PI;
		}
		fftParallel(FFTW_BACKWARD, data);
		
		if(SingularityCompensation){
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++){
				double r = Norm(data.VecAt[i]), CompensationFac;
				if(i!=data.CentreIndex) CompensationFac = gsl_sf_erf(0.5*r)/r;
				else CompensationFac = 1./SQRTPI;
				data.ComplexField[i][0] += Ref0tilde*CompensationFac;
				data.ComplexField[i][1] += Imf0tilde*CompensationFac;
			}
		}
		
		vector<double> integrand(data.GridSize);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<data.GridSize;i++){
			integrand[i] = data.TmpComplexField[i][0]*data.ComplexField[i][0] - data.TmpComplexField[i][1]*data.ComplexField[i][1];
		}
	
		double A = Integrate(data.ompThreads, data.method, data.DIM, integrand, data.frame), B = TASK.ex.Iabcd[a*L*L*L+b*L*L+c*L+d];
		if(ABS(A)>1.0e-6 || ABS(B)>1.0e-6){
			TASKPRINT(to_string(i+1) + "/" + to_string(Indices.size()) + " testHydrogenicHint(" + to_string(a) + "," + to_string(b) + "," + to_string(c) + "," + to_string(d) + ") = " + to_string_with_precision(A,12),TASK,1);
			TASKPRINT(to_string(i+1) + "/" + to_string(Indices.size()) + "       TabFunc_Hint(" + to_string(a) + "," + to_string(b) + "," + to_string(c) + "," + to_string(d) + ") = " + to_string_with_precision(B,12),TASK,1);
		}
		else TASKPRINT(to_string(i+1) + "/" + to_string(Indices.size()) + "                  @(" + to_string(a) + "," + to_string(b) + "," + to_string(c) + "," + to_string(d) + ") < 1.0e-6",TASK,1);
	
	}
	TASKPRINT("testHydrogenicHint finished.",TASK,1);
	task.controltaskfile << TASK.controltaskfile.str();
}

void fab(vector<vector<double>> &TargetField, int a, int b, datastruct &data, taskstruct &task){
	double Z=1.;
	#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 
	for(int i=0;i<data.GridSize;i++){
		vector<double> sc = SphericalCoor(data.VecAt[i]);
		double r = sc[0], theta = sc[1], phi = sc[2], rho = 2.*Z*r;
		vector<int> nlma = TASK.ex.QuantumNumbers[a], nlmb = TASK.ex.QuantumNumbers[b];
		int na = nlma[0], la = nlma[1], ma = nlma[2], nb = nlmb[0], lb = nlmb[1], mb = nlmb[2];
		double dna = (double)na, dla = (double)la, xa = rho/dna, dnb = (double)nb, dlb = (double)lb, xb = rho/dnb;

		double Ra = sqrt(POW(2.*Z/dna,3)*EXP((double)gsl_sf_lnfact((unsigned int)(na-la-1))-(double)gsl_sf_lnfact((unsigned int)(na+la))-xa)/(2.*dna)) * POW(xa,la) * boost::math::laguerre((unsigned int)(na-la-1),(unsigned int)(2*la+1),xa);
		double Rb = sqrt(POW(2.*Z/dnb,3)*EXP((double)gsl_sf_lnfact((unsigned int)(nb-lb-1))-(double)gsl_sf_lnfact((unsigned int)(nb+lb))-xb)/(2.*dnb)) * POW(xb,lb) * boost::math::laguerre((unsigned int)(nb-lb-1),(unsigned int)(2*lb+1),xb);

		mb *= -1;//because of the complex conjugate spherical harmonic
		double Ya = sqrt(EXP((double)gsl_sf_lnfact((unsigned int)(la-ma))-(double)gsl_sf_lnfact((unsigned int)(la+ma))) * (2*la+1)/(4.*PI)) * boost::math::legendre_p(la,ma,cos(theta));
		double Yb = sqrt(EXP((double)gsl_sf_lnfact((unsigned int)(lb-mb))-(double)gsl_sf_lnfact((unsigned int)(lb+mb))) * (2*lb+1)/(4.*PI)) * boost::math::legendre_p(lb,mb,cos(theta));
		
		double msign = -1.;
		if(mb%2==0) msign = 1.;
		double qab = Ra*Rb*msign*Ya*Yb;
		
		TargetField[i][0] = qab*cos((ma+mb)*phi);
		TargetField[i][1] = qab*sin((ma+mb)*phi);
	}	
}

/* routines for KD: */

double tweeze_phi(double x, double a, double b, bool left, KDparams &p){
	double midpoint = 0.5*(a+b);
	if(RelDiff(a,b)<p.epsTweeze) return midpoint;
	bool overshot = false;
	if(KDphi(p.alpha,p.beta,midpoint)>x) overshot = true;
	if( (left && overshot) || (!left && !overshot) ) return tweeze_phi(x,midpoint,b,left,p);
	else return tweeze_phi(x,a,midpoint,left,p);
}

double K1integrand(double x, KDparams &p, taskstruct &task){
	double alpha=p.alpha; double beta=p.beta;
	double s1, s2, s1s2contrib = 0., s0 = 1./sqrt(2.*(sqrt(alpha*alpha+beta*beta*beta)+alpha)), phi0 = KDphi(alpha, beta, s0);
	
	if(x>phi0){
		if(p.epsTweeze>0.){//bisection variant
			double s1a = 0.5*s0;
			while(KDphi(alpha,beta,s1a)<x) s1a *=0.5;
			double s1b = 2.*s1a;
			double s2b = 2.*s0;
			while(KDphi(alpha,beta,s2b)<x) s2b *=2.;
			double s2a = 0.5*s2b;
			s1 = tweeze_phi(x,s1a,s1b,true,p);
			s2 = tweeze_phi(x,s2a,s2b,false,p);
		}
		else{
			double alpha2 = alpha*alpha, alpha3 = alpha2*alpha, beta3 = beta*beta*beta, beta3inv = 1./beta3, x2 = x*x, x4 = x2*x2;
			//Alex' PhD thesis (4.49)
// 			double a = alpha3/(4.*beta3*beta3*beta3)+9.*x*x/(8.*beta3*beta3)-3.*alpha/(4.*beta3*beta3);
// 			double b = POW(alpha2/(4.*beta3*beta3)+1./(4.*beta3),3);
// 			double c = 0.5*a;
// 			double d = 0.25*a*a-b; if(d<0.) d = 0.; else d = sqrt(d); 
// 			double m = pow(c+d,0.3333333333333333)+pow(c-d,0.3333333333333333)-alpha/beta3;
// 			double P = 2.*m; if(P<0.) P = 0.; else P = -sqrt(P);
// 			double Q = 3.*alpha/(2.*beta3)+m+P*3.*x/(4.*m*beta3);//what if m=0?
// 			double root = P*P-4.*Q; if(root<0.) root = 0.; else root = sqrt(root);
// 			s1 = 0.5*(-P-root);
// 			s2 = 0.5*(-P+root);
// 			if(!(std::isfinite(s1) && std::isfinite(s2))){
// 				cout << alpha << " " << beta << " " << a << " " << b << " " << c << " " << d << " " << m << " " << P << " " << Q << " " << root << " " << s1 << " " << s2 << endl;
// 				usleep(100000);
// 			}
			//analytical variant, see K1.nb
			//real version
// 			double expr7 = 2.*alpha3-6.*alpha*beta3+9.*beta3*x*x;
// 			double arg = beta3*(36.*(alpha3-3.*alpha*beta3)*x*x+81.*beta3*x*x*x*x-4.*(-3.*alpha2+beta3)*(-3.*alpha2+beta3));
// 			if(arg>1.0e-32) expr7 += sqrt(arg);
// 			double expr6 = 1.0e-16;
// 			if(expr7>1.0e-48) expr6 = pow(expr7,0.3333333333333333);
// 			double expr5 = (alpha2+beta3)/expr6;
// 			double expr4 = max(1.0e-32,beta3inv*(-4.*alpha+2.5198420997897464*expr5+1.5874010519681994*expr6));
// 			double sqrtexpr4 = sqrt(expr4);
// 			double expr3 = 8.485281374238571*x*beta3inv/sqrtexpr4;
// 			double expr2 = -4.*alpha*beta3inv-1.2599210498948732*beta3inv*expr5-0.7937005259840998*beta3inv*expr6;
// 			double expr1 = 0.35355339059327373*sqrtexpr4;
// 			double expr2m3 = expr2-expr3;//max(1.0e-32,expr2-expr3);
// 			double expr2p3 = expr2+expr3;//max(1.0e-32,expr2+expr3);
// 			double sqrtexpr2m3 = sqrt(expr2m3);
// 			double sqrtexpr2p3 = sqrt(expr2p3);
// 			vector<double> sVec = {{-expr1-0.5*sqrtexpr2m3, -expr1+0.5*sqrtexpr2m3, expr1-0.5*sqrtexpr2p3, expr1+0.5*sqrtexpr2p3}};
// 			vector<double> s(0);
// 			for(int i=0;i<4;i++) if(std::isfinite(sVec[i]) && sVec[i]>0.) s.push_back(ABS(sVec[i]));
			
			//complex version
			vector<std::complex<double>> sVec(4);
// 			sVec[0] = -0.35355339059327373*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),0.5) - 0.5*pow(-4.*alpha*beta3inv - 1.2599210498948732*beta3inv*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) - 0.7937005259840998*beta3inv*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333) - 8.485281374238571*x*beta3inv*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),-0.5),0.5);
// 
// 			sVec[1] = -0.35355339059327373*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),0.5) + 0.5*pow(-4.*alpha*beta3inv - 1.2599210498948732*beta3inv*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) - 0.7937005259840998*beta3inv*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333) - 8.485281374238571*x*beta3inv*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),-0.5),0.5);
// 
// 			sVec[2] = 0.35355339059327373*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),0.5) - 0.5*pow(-4.*alpha*beta3inv - 1.2599210498948732*beta3inv*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) - 0.7937005259840998*beta3inv*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333) + 8.485281374238571*x*beta3inv*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),-0.5),0.5);
// 
// 			sVec[3] = 0.35355339059327373*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),0.5) + 0.5*pow(-4.*alpha*beta3inv - 1.2599210498948732*beta3inv*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) - 0.7937005259840998*beta3inv*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333) + 8.485281374238571*x*beta3inv*pow(beta3inv*(-4.*alpha + 2.5198420997897464*(alpha2 + beta3)*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),-0.3333333333333333) + 1.5874010519681994*pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + pow(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*pow(-3.*alpha2 + beta3,2)),0.5),0.3333333333333333)),-0.5),0.5);
			
			std::complex<double> v0 = (-3.*alpha2 + beta3)*(-3.*alpha2 + beta3);
			std::complex<double> v1 = pow(2.*alpha3 - 6.*alpha*beta3 + 9.*beta3*x2 + sqrt(beta3*(36.*(alpha3 - 3.*alpha*beta3)*x2 + 81.*beta3*x4 - 4.*v0)),0.3333333333333333);
			std::complex<double> v2 = (alpha2 + beta3)/v1;
			std::complex<double> v3 = -4.*alpha*beta3inv - 1.2599210498948732*beta3inv*v2 - 0.7937005259840998*beta3inv*v1;
			std::complex<double> v4 = 1.5874010519681994*v1;
			std::complex<double> v5 = sqrt(beta3inv*(-4.*alpha + 2.5198420997897464*v2 + v4));
			std::complex<double> v6 = 0.35355339059327373*v5;
			std::complex<double> v7 = 8.485281374238571*x*beta3inv/v5;
			std::complex<double> v8 = sqrt(v3 - v7);
			std::complex<double> v9 = sqrt(v3 + v7);
			sVec[0] = -v6 - 0.5*v8;
			sVec[1] = -v6 + 0.5*v8;
			sVec[2] = v6 - 0.5*v9;
			sVec[3] = v6 + 0.5*v9;

			vector<double> s(0);
			for(int i=0;i<4;i++) if(std::isfinite(sVec[i].real()) && sVec[i].real()>0. && ABS(sVec[i].imag())<1.0e-10) s.push_back(sVec[i].real());
			
			//cout << s.size() << " " << alpha << " " << beta << " " << x << " " << vec_to_str(s); usleep(1000000);
			if(s.size()!=2){
				#pragma omp critical
				{
					string ss = "empty";
					if(s.size()>0) ss = vec_to_str(s);
					TASKPRINT("s = " + ss,task,0);
					p.Error = true;
					s1 = s0;
					s2 = s0;
				}
			}
			else{
				s1 = min(s[0],s[1]);
				s2 = max(s[0],s[1]);
			}
		}
		s1s2contrib = -((x-alpha*s1-POW(beta*s1,3)/3.)-(x-alpha*s2-POW(beta*s2,3)/3.));
	}

	return (POS(x)+s1s2contrib)*sin(x);
}

double GetPartialK1(double a, double b, KDparams &p, taskstruct &task){
	//integrates for range a<x<b, with subdiv intervals of size dx;
	int bp = max(1,(int)(0.25*(double)p.subdiv)), Lf = 4*bp+1;
	double dx=(b-a)/((double)(Lf-1)); 
	vector<double> f(0);
	for(int d=0;d<Lf;d++){
		double x = a+(double)d*dx;
		f.push_back(K1integrand(x,p,task));
	}
	p.FuncEvals += (double)Lf;
	return BooleRule1D(1,f,Lf,a,b,bp);
}

double GetBisectionK1(double a, double b, int iter, double prevInt, KDparams &p, taskstruct &task){
	double midpoint = 0.5*(a+b);
	//calculate bisection values
	double A = GetPartialK1(a,midpoint,p,task);
	double B = GetPartialK1(midpoint,b,p,task);
	double C = A+B;
	double absDiff = ABS(prevInt-C);
	double relDiff = RelDiff(prevInt,C);
	//set flags for controlled termination
	bool resolutionLimit = false, iterLimit = false, TargetRelAccuracyReached = false, TargetAbsAccuracyReached = false, distinguishabilityLimit = false, alert = false;
	if(b-a<p.minDelta) resolutionLimit = true;
	if(iter==p.maxIter) iterLimit = true;
	if(absDiff<p.relCrit*min(ABS(prevInt),ABS(C))) TargetRelAccuracyReached = true;
	if(absDiff<p.absCrit) TargetAbsAccuracyReached = true;
	if(relDiff<p.distinguishabilityThreshold) distinguishabilityLimit = true;
	if(relDiff>max(0.01,sqrt(p.relCrit))) alert = true;
	//process bisection result
	if(!std::isfinite(A) || !std::isfinite(B)){ if(omp_get_thread_num()==0) TASKPRINT("GetPartialIntegral: Integral not finite!",task,p.printQ); return prevInt; }
	if( iterLimit || (iter>p.minIter && (resolutionLimit || TargetRelAccuracyReached || TargetAbsAccuracyReached)) || distinguishabilityLimit){
		if(alert){
			if(omp_get_thread_num()==0) if(resolutionLimit) TASKPRINT("GetPartialIntegral: minDelta (b-a<" + to_string_with_precision(p.minDelta,16) + ") reached!",task,p.printQ);
			if(omp_get_thread_num()==0) if(iterLimit) TASKPRINT("GetPartialIntegral: maxIter=" + to_string(p.maxIter) + " reached --- final successive section values " + to_string_with_precision(prevInt,16) + " --> " + to_string_with_precision(C,16),task,p.printQ);
		}
		//if(distinguishabilityLimit) cout << "GetPartialIntegral: successive section values numerically indistinguishable " << prevInt << " --> " << C << endl;
		return C;
	}
	double refinedA = GetBisectionK1(a,midpoint,iter+1,A,p,task);
	double refinedB = GetBisectionK1(midpoint,b,iter+1,B,p,task);
	return refinedA+refinedB;
}

double getK1(KDparams &p, datastruct &data, taskstruct &task){
	double alpha=p.alpha; double beta=p.beta; int IntLimitCount=p.IntLimitCount; double epsIntLimits=p.epsIntLimits; double bStepScale=p.bStepScale; double relCrit=p.relCrit; double absCrit=p.absCrit; int subdiv=p.subdiv; int printQ=p.printQ;
	if(omp_get_thread_num()==0){
		TASKPRINT("... KDparams: alpha=" + to_string_with_precision(alpha,16) + " beta=" + to_string_with_precision(beta,16) + " IntLimitCount=" + to_string(IntLimitCount) + " epsIntLimits=" + to_string(epsIntLimits) + " relCrit=" + to_string_with_precision(relCrit,16) + " absCrit=" + to_string_with_precision(absCrit,16) + " subdiv=" + to_string(subdiv),task,printQ);
	}
	p.FuncEvals = 0.;
	
	//find expedient integration limits a & b
	int successCounterPos = 0;
	double s0 = 1./sqrt(2.*(sqrt(alpha*alpha+beta*beta*beta)+alpha)), phi0 = KDphi(alpha, beta, s0);
	double a = min(0.,phi0);
	double b = p.UpperLimit;
	if(p.UpperLimit==1.23456789){
		b = 1.;
		bool IntegrationLimitsFound = false;
		while(!IntegrationLimitsFound){
			if(successCounterPos<IntLimitCount){//upper limit not found yet
				if(abs(K1integrand(b,p,task))<epsIntLimits) successCounterPos++;
				else successCounterPos = 0.;
				b *= bStepScale;
			}		
			if(successCounterPos==IntLimitCount){
				IntegrationLimitsFound = true;
				b /= bStepScale;
			}
		}
	}
	b = 2.*PI*(floor(b/(2.*PI))+1.);
	
	//compute total integral between a and b
	double BaseLength = p.BaseScale*2.*PI;
	double ab = 0.; while(ab<a) ab += BaseLength;
	while((b-ab)/BaseLength>2147483646.){
		BaseLength *= 2.;
		if(omp_get_thread_num()==0) TASKPRINT("... integration base exceeds 2147483647 ParallelizedIntervals --> increase BaseLength to " + to_string(BaseLength),task,printQ);
	}
	int ParallelizedIntervals = (int)((b-ab)/BaseLength+0.5);
	if(omp_get_thread_num()==0) TASKPRINT("... get total integral on " + to_string(ParallelizedIntervals) + " intervals of length " + to_string(BaseLength) + " <-> integration limits [" + to_string(a) + "," +  to_string(b) + "]",task,printQ);
	//first get integral between a and ab
	double result = GetBisectionK1(a,ab,0,1.0e+300,p,task);
	vector<double> Result(ParallelizedIntervals);
	#pragma omp parallel for schedule(dynamic) if(p.Threads>1)
	for(int i=0;i<ParallelizedIntervals;i++) Result[i] = GetBisectionK1(ab+(double)i*BaseLength,ab+(double)(i+1)*BaseLength,0,1.0e+300,p,task);
	//add integral between ab and b
	result += accumulate(Result.begin(),Result.end(),0.);
	result *=  p.prefactor;
	
	//report
	if(printQ>1) for(int i=0;i<ParallelizedIntervals;i++) cout << "Result[" << ab+(double)i*BaseLength << "," <<  ab+(double)(i+1)*BaseLength << "] = " << Result[i] << endl;
	if(ParallelizedIntervals>100){
		string LastBisectionResults = "... sequence of last 100 bisection results {IntervalMidpoint,K1}:\n{";
		for(int i=ParallelizedIntervals-100;i<ParallelizedIntervals;i++){
			if(i>ParallelizedIntervals-100) LastBisectionResults += ",";
			LastBisectionResults += "{" + to_string_with_precision(ab+((double)i+0.5)*BaseLength,16) + "," + to_string_with_precision(Result[i],16) + "}";
		}
		LastBisectionResults += "}";
		if(omp_get_thread_num()==0) TASKPRINT(LastBisectionResults,task,printQ);
	}
	if(omp_get_thread_num()==0){
		TASKPRINT("... K1(alpha=" + to_string_with_precision(alpha,16) + ",beta=" + to_string_with_precision(beta,16) + ") = " + to_string_with_precision(result,16) + " [from " + to_string_with_precision(p.FuncEvals,16) + " integrand evaluations]\n",task,printQ);
	}
	
	return result;
}

void GetKD(int D, double alpha, double beta, KDparams &p, datastruct &data, taskstruct &task){
	string reldiffstring = "", KD_from_Plugin_KD = "";
	if(omp_get_thread_num()==0){
		StartTimer("GetKD",data);
		cout << "GetKD: alpha = " << alpha << "; beta = " << beta << endl;
	}
	if(beta<p.betaThreshold){
		if(alpha>0.){
			double sa = sqrt(alpha), Bessel;
			if(sa>2.0e+12) Bessel = sqrt(2./(PI*sa))*cos(sa-0.5*data.DDIM*PI-0.25*PI);
			else Bessel = gsl_sf_bessel_Jn(D,sa);
			p.Result = POW(sa,D) * Bessel;
		}
		else p.Result = 0.;
	}
	else{
		p.alpha = alpha;
		p.beta = beta;
		double prevres = 2.*p.Result, UpperLimit = p.UpperLimit;
		while(RelDiff(p.Result,prevres)>p.RelCrit){
			prevres = p.Result;
			if(D==1) p.Result = getK1(p, data, task);
			p.UpperLimit *= 4.;
			if(omp_get_thread_num()==0) cout << "GetKD: Intermediate result = " << p.Result << endl;
		}
		p.UpperLimit = UpperLimit;
		reldiffstring = " --- relative deviation to previous result = " + to_string_with_precision(RelDiff(p.Result,prevres),16);
	}
	if(omp_get_thread_num()==0){
		EndTimer("GetKD",data);
		double ABSERR;
        struct KDintegrationParams KDip;
		if(p.CompareKD) KD_from_Plugin_KD = "\n  KD_from_Plugin_KD = \n" + to_string_with_precision(KD(data.DIM,alpha,beta,ABSERR,data.InternalAcc,KDip),16);
		TASKPRINT("GetKD: Final result [from " + to_string_with_precision(p.FuncEvals,16) + " integrand evaluations]" + reldiffstring + "\n  K" + to_string(D) + "(alpha=" + to_string_with_precision(alpha,16) + ",beta=" + to_string_with_precision(beta,16) + ") = \n" + to_string_with_precision(p.Result,16) + KD_from_Plugin_KD + "\n",task,1);
	}
}

void testK1(datastruct &data, taskstruct &task){
	KDparams KDparameters;
	
	//BEGIN USER INPUT
	KDparameters.alpha = 1.;
	KDparameters.beta = 1.;	
	KDparameters.RelCrit = 1.0e-2;//2.0e-3;//
	KDparameters.UpperLimit = 1.0e+1;//1.0e+3;//
	KDparameters.IntLimitCount = 10;
	KDparameters.relCrit = 1.0e-1;//1.0e-3;//
	KDparameters.absCrit = 1.0e-3;//1.0e-6;//
	KDparameters.minIter = 2;
	KDparameters.maxIter = 6;//
	KDparameters.betaThreshold = MachinePrecision;//1.0e-6;//
	//KDparameters.epsTweeze = 1.0e-12;
	KDparameters.epsIntLimits = 1.0e+300;//1.0e-2;
	KDparameters.Threads = data.ompThreads;
	KDparameters.prefactor = 2./PI;
	KDparameters.BaseScale = 1.;
	KDparameters.bStepScale = 1.23456789;
	KDparameters.minDelta = 1.0e-16;
	KDparameters.distinguishabilityThreshold = 1.0e-14;
	KDparameters.printQ = 0;
	KDparameters.CompareKD = true;
	data.Print = 1;
	//END USER INPUT
	
	//Production runs
	GetKD(1, 1.,1., KDparameters, data, task);
	GetKD(1, 1.,2., KDparameters, data, task);
	GetKD(1, 2.,1., KDparameters, data, task);
	GetKD(1, 2.,2., KDparameters, data, task);
	GetKD(1, 20.,2., KDparameters, data, task);
	GetKD(1, 2.,20., KDparameters, data, task);
	GetKD(1, 20.,20., KDparameters, data, task);
	GetKD(1, 20.,50., KDparameters, data, task);
	GetKD(1, 50.,20., KDparameters, data, task);
	GetKD(1, 50.,50., KDparameters, data, task);
	GetKD(1, 100.,50., KDparameters, data, task);
	GetKD(1, 50.,100., KDparameters, data, task);
	GetKD(1, 100.,100., KDparameters, data, task);
	GetKD(1, -100.,1., KDparameters, data, task);
	GetKD(1, -100.,50., KDparameters, data, task);
	GetKD(1, -1000.,50., KDparameters, data, task);//problematic
	GetKD(1, -10000.,50., KDparameters, data, task);//problematic
	GetKD(1, -10000.,10000., KDparameters, data, task);//problematic

}

// Helper function that executes a command and captures its output.
string exec(const char* cmd) {
    array<char, 128> buffer;
    string result;
    // Open a pipe to run the command.
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        throw runtime_error("popen() failed!");
    }
    // Read the output from the command.
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

int testIAMIT(vector<double> &x, datastruct &data, taskstruct &task){
  	cout << "testIAMIT:" << endl;

    vector<double> Sum(data.ompThreads);
    // Parallel region using OpenMP.
    #pragma omp parallel for schedule(static) if(data.ompThreads>1)
    for (int i = 0; i < data.ompThreads; ++i) {
        // Build the command string with the Python script and its arguments.
        ostringstream cmd;
        cmd << "python3 /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts/Project_ItaiArad_MIT/noisy-DM-PEPS-sim.py";
        for (double num : x) {
            cmd << " " << num;
        }
        try {
            // Execute the command and capture its output.
            string output = exec(cmd.str().c_str());
            // Convert the output string to a double.
            double sum = stod(output);
            // Use an OpenMP critical section for safe output to the console.
            #pragma omp critical
            {
                cout << "Thread " << omp_get_thread_num() << " - Sum: " << sum << endl;
                Sum[omp_get_thread_num()] = sum+(double)i;
            }
        } catch (const exception &e) {
            #pragma omp critical
            {
                cerr << "Error in thread " << omp_get_thread_num() << ": " << e.what() << endl;
            }
        }
    }

    cout << vec_to_str(Sum) << endl;

    return 0;


}

// R O U T I N E S    F O R  ---  OPTIMIZATION  ---  MACHINE LEARNING  --- NEURAL NETWORKS

void ParetoFront(datastruct &data, taskstruct &task){
	vector<vector<double>> ParetoFront(0);
	
	//BEGIN USER INPUT
	//Choose optimizer identifier opt_ID:
	//CGD(100) --- PSO(101) --- LCO(102) --- AUL(103) --- GAO(104) --- cSA(105) --- CMA(106)
	int opt_ID = 106;
	
	//List of optimization tasks --- function identifier func_ID for specific OptimizationProblem (GetFuncVal, see Plugin_OPT.cpp for details):
	//0: Template --- -2014+i: cec14 test functions (with indices i=1...30) --- -440: ForestGEO --- -100: custom objective function (defined in Plugin_OPT.cpp) --- -44: ForestGEO --- -20: NN --- -11: -2: 1pEx-DFT (combined minimization) --- -1: 1pEx-DFT (separated minimization) --- 1-9: Reserved slots --- 10-99: Test problems: 10: ChemicalProcess --- 100-999: Problem zoo: 100: MutuallyUnbiasedBases -- 101: ShiftedSphere -- 102: SineEnvelope -- 103: Rana -- 104: UnconstrainedRana -- 105: UnconstrainedEggholder -- 200: NYFunction -- 201: QuantumCircuitIA -- 300: ConstrainedRastrigin --- 1000-9999: DFTe
	
	//Uncomment one of the following tasks or create a new one

    //Task Misc
	//ParetoFront.push_back(Optimize(0, opt_ID, 0, data, task));
	//ParetoFront.push_back(Optimize(-2014+5, opt_ID, 0, data, task));
	//for(int func_ID=-2014+1;func_ID<=-2014+16;func_ID++) ParetoFront.push_back(Optimize(func_ID, opt_ID, 0, data, task));
	//for(int func_ID=-2014+1;func_ID<=-2014+22;func_ID++) ParetoFront.push_back(Optimize(func_ID, opt_ID, 0, data, task));
	//for(int func_ID=-2014+23;func_ID<=-2014+30;func_ID++) ParetoFront.push_back(Optimize(func_ID, opt_ID, 0, data, task));
	//for(int func_ID=-2014+1;func_ID<=-2014+30;func_ID++) ParetoFront.push_back(Optimize(func_ID, opt_ID, 0, data, task));
    //ParetoFront.push_back(Optimize(10, opt_ID, 0, data, task));
	//for(int func_ID=101;func_ID<=104;func_ID++) ParetoFront.push_back(Optimize(func_ID, opt_ID, 0, data, task));
    //for(int i=0;i<9;i++) ParetoFront.push_back(Optimize(105, opt_ID, 2*i, data, task));
    ParetoFront.push_back(Optimize(201, opt_ID, 0, data, task));
    //for(int i=0;i<5;i++) ParetoFront.push_back(Optimize(300, opt_ID, i, data, task));

	//Task NYFunction:
    //single:
    //ParetoFront.push_back(Optimize(200, opt_ID, 0, data, task));
    //repetitive:
    // for(int rep=1;rep<=20;rep++){
    //   	TASKPRINT("ParetoFront: REPETITION " + to_string(rep),task,1);
    //   	ParetoFront.push_back(Optimize(200, opt_ID, 0, data, task));
    // }
    //production:
    // string InputFileName = findMatchingFile("TabFunc_NYFunction_",".dat");
    // task.Aux = 1.;
    // task.refData = vector<vector<double>>(ReadMat(InputFileName));
    // ParetoFront.resize(task.refData.size());
    // int C = 1/*#FuncEvals*/ + 1/*optimum*/ + 5/*optimizer*/ + task.refData[0].size()/*frequencies*/ + task.refData[0].size()/*probabilities*/ + 2*4/*scalarproducts of tetrahedron measurements*/ + 3/*scr,ccr,lcr*/;
    // for(int c=0;c<ParetoFront.size();c++){
    //   ParetoFront[c].clear();
    //   ParetoFront[c].resize(C);
    // }
    // #pragma omp parallel for schedule(static) if(data.ompThreads>1)
    // for(int c=0;c<ParetoFront.size();c++){
    //   	if(omp_get_thread_num()==0) TASKPRINT("ParetoFront: input vector #" + to_string((c+1)*data.ompThreads) + " (estimate)",task,1);
    //     ParetoFront[c] = Optimize(200, opt_ID, c, data, task);
    // }
    // string OutputFileName = "mpDPFT_" + InputFileName.substr(0, InputFileName.length() - 4) + "_" + data.AuxString + "_" + YYYYMMDD() + "_" + hhmmss() + ".dat";
    // cout << "OutputFileName " << OutputFileName << endl;
    // MatrixToFile(ParetoFront,OutputFileName,14);
	
	// //prepare optimization task in Optimize(...) below
	
	//in Plugin_OPT.cpp:
	// - Ensure that ConstrainXXX are handled problem-specifically, see ConstrainPSO, ConstrainGAO, ConstrainCMA, etc.
	// - supply code of opt.function in GetFuncVal
	
	//END USER INPUT
	
	MatrixToFile(ParetoFront,"mpDPFT_ParetoFront.dat",14);
}

vector<double> Optimize(int func_ID, int opt_ID, int aux, datastruct &data, taskstruct &task){
	
	OPTstruct opt;
	SetDefaultOPTparams(opt);
	opt.threads = data.ompThreads;
	vector<double> paretofront;
	
	opt.reportQ = 1;
	opt.function = func_ID;
	
	if(opt.function==0){//template (copy all & delete / comment what is not needed)
		//BEGIN USER INPUT template
		opt.D = 10;
		opt.epsf = 1.0e-6;
		opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 100;
			opt.pso.increase = 10.;//increase initial swarm size by this factor toward final run
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = 10*opt.D;
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = max(10,min(50,opt.D));
			opt.pso.MaxLinks = 3;
			//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
			//opt.pso.elitism = 0.1;
			//opt.pso.PostProcessQ = 1;
			opt.pso.loopMax = (int)1.0e+7;
			//opt.pso.reseed = 0.25;
			opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
			//opt.pso.alpha = 0.;
			//opt.BreakBadRuns = 1;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
		else if(opt_ID==104){//GAO
			//begin adjust parameters
			//opt.UpdateSearchSpaceQ = 2;
			opt.evalMax = 1.0e+6;//start low
			opt.gao.runs = 20*opt.D;
			opt.gao.popExponent = 0.3;//trade-off between large populations (popExponent->1) and many generations (popExponent->0)
			//opt.gao.RandomSearch = true;//use GAO as random-search optimizer with exactly opt.evalMax function evaluations (no further inputs are necessary)
			SetDefaultGAOparams(opt);
			opt.gao.mutationRate = 0.02;//in [0.0,0.2]	
			opt.gao.mutationRateDecay = 1.0;//in [-0.2,1.0]
			opt.gao.mutationStrength = 10.;//in [0,100]
			opt.gao.mutationStrengthDecay = 1.0;//in [-0.2,1.0]
			opt.gao.invasionRate = 0.02;//in [0,0.05]
			opt.gao.invasionRateDecay = 1.0;//in [-0.2,1.0]
			opt.gao.crossoverRate = 0.8;//in [0.2,1.0]
			opt.gao.crossoverRateDecay = 0.;//in [0.0,1.0]
			opt.gao.dispersalStrength = 1.5;//in [0,runs]
			opt.gao.dispersalStrengthDecay = -(double)opt.gao.runs;//in [-runs,0]	
			//opt.gao.HyperParamsDecayQ = false;
			//opt.gao.HyperParamsSchedule = 2;	
			//opt.PickRandomParamsQ = true;	
			opt.gao.ShrinkPopulationsQ = true;
			//end adjust parameters
			GAO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.gao.bestf);	
			paretofront.insert(paretofront.end(), opt.gao.bestx.begin(), opt.gao.bestx.end());
		}
		else if(opt_ID==106){//CMA
			opt.epsf = 1.0e-6;
			//opt.UpdateSearchSpaceQ = 2;
			opt.BreakBadRuns = 1;//0: don't --- 1: only in case of severe violations --- 2: aggressive termination policy
			SetDefaultCMAparams(opt);
			opt.cma.runs = opt.threads;
			opt.cma.popExponent = 5;//population prop to 2^popExponent
			opt.cma.CheckPopVariance = 0.2;//percentage of best populations to assess for termination criterion
			opt.cma.PickRandomParamsQ = true;
			opt.cma.DelayEigenDecomposition = true;
			opt.cma.elitism = true;
			opt.cma.WeightScenario = 1;//0: constant weights --- 1: standard positive weights
			opt.cma.Constraints = 2;//0: No Contraints --- 1: Box Constraints --- 2: Penalty (smooth square of box repair distance) --- 3: Penalty (sharp sqrt of box repair distance) --- 4: Penalty (constant 1.0e+100)
			CMA(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.currentBestf);	
			paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());			
		}
		//END USER INPUT template
	}
	else if(opt.function>=-2014+1 && opt.function<=-2014+30){//cec14 test functions (with indices i=1...30)
		opt.D = 10;//20;//2;//50;//
		opt.epsf = 1.0e-6;
		opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.epsf = 1.0e-9;
			opt.pso.runs = 100*opt.D;
			opt.pso.increase = 100.;
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = 1*opt.D;
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = 10*opt.D;
			opt.pso.loopMax = (int)1.0e+6;
			opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
		else if(opt_ID==104){//GAO
			//begin adjust parameters
			opt.UpdateSearchSpaceQ = 0;
			opt.evalMax = 1.0e+9;//start low
			opt.gao.runs = 20*opt.D;
			opt.gao.popExponent = 0.4;
			SetDefaultGAOparams(opt);
			opt.gao.HyperParamsDecayQ = true;
			opt.gao.HyperParamsSchedule = 0;
			opt.PickRandomParamsQ = false;
			opt.gao.ShrinkPopulationsQ = false;
			//end adjust parameters
			GAO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.gao.bestf);
			paretofront.insert(paretofront.end(), opt.gao.bestx.begin(), opt.gao.bestx.end());
		}
		else if(opt_ID==106){//CMA
			opt.epsf = 1.0e-14;//1.0e-6;//
			opt.BreakBadRuns = 1;
			SetDefaultCMAparams(opt);
			opt.cma.runs = 100/*100000*/;//opt.D;//10*opt.D;
			opt.cma.popExponent = 5;//10;
			opt.cma.muRatio = 0.5;//0.9;//
			//opt.cma.CheckPopVariance = 0.1;
			opt.cma.VarianceCheck = 10*opt.D;
			opt.cma.DelayEigenDecomposition = true;
			opt.cma.elitism = true;
			opt.cma.WeightScenario = 2;//1;
			opt.cma.Constraints = 2;
			CMA(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.currentBestf);
			paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
// 			opt.TestMode = 1;
// 			cout << GetFuncVal( 0, opt.currentBestx, opt.function, opt ) << endl;
// 			SleepForever();
		}
	}
	else if(opt.function==10){//ChemicalProcess (Himmelblau...pdf)
      	opt.D = 7;//{Y1,Y2,Y3,C1,B2,B3,BP}
      	vector<double> X(opt.D);

      	opt.NumEC = 1;
        opt.NumIC = 3;
        opt.PenaltyMethod = {0.5,1.0e+6,0.,0.};//seems to work best
        //opt.PenaltyMethod = {1.5,1.0e+2,0.,0.1};
        //opt.PenaltyMethod = {2.5,1.0e+4,0.,0.1};
        //opt.PenaltyMethod = {3.5,1.0,0.,0.};

        //opt.anneal = 1.0e-4;

      	opt.epsf = 1.0e-10;
        opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = PI;
      	opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      	opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);


        if(opt_ID==101){//PSO
          SetDefaultPSOparams(opt);
          //begin adjust parameters
          opt.pso.runs = 100*opt.D;
          opt.pso.increase = 10.;
          opt.pso.SwarmDecayRate = 1.2;
          opt.pso.InitialSwarmSize = 100*opt.D;
          InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
          opt.pso.VarianceCheck = opt.D;
          opt.pso.loopMax = 200;
          //opt.pso.reseed = 0.25;
          opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
          //opt.BreakBadRuns = 1;
          opt.pso.TargetMinEncounters = opt.pso.runs;
          //end adjust parameters
          PSO(opt);
          X = opt.pso.bestx;
          paretofront.push_back(opt.nb_eval);
          paretofront.push_back(opt.pso.bestf);
        }
        else if(opt_ID==106){//CMA
        	//opt.UpdateSearchSpaceQ = 2;
        	opt.BreakBadRuns = 1;//0: don't --- 1: only in case of severe violations --- 2: aggressive termination policy
        	SetDefaultCMAparams(opt);
            opt.cma.runs = 1000*opt.D;
            opt.cma.popExponent = 4;//6;//
            opt.cma.VarianceCheck = opt.D;
            //opt.stallCheck = 10*opt.D;
            opt.cma.CheckPopVariance = 0.05;
            opt.cma.PopulationDecayRate = 2.;
            opt.cma.elitism = true;//false;//
        	//opt.cma.PickRandomParamsQ = true;
        	//opt.cma.DelayEigenDecomposition = true;
        	opt.cma.WeightScenario = 2;//0: constant weights --- 1: standard positive weights
        	opt.cma.Constraints = 0;//0: No Contraints --- 1: Box Constraints --- 2: Penalty (smooth square of box repair distance) --- 3: Penalty (sharp sqrt of box repair distance) --- 4: Penalty (constant 1.0e+100)
        	CMA(opt);
        	paretofront.push_back(opt.nb_eval);
        	paretofront.push_back(opt.currentBestf);
            X = opt.currentBestx;
      	}

      	vector<double> EC(opt.NumEC);
        vector<double> IC(opt.NumIC);
        GetECIC(EC,IC,X,opt);
        TASKPRINT("                  f (without penalty terms) = " + to_string(ChemicalProcess(X,opt,-1)) + " @",task,1);
      	X = x2X_ChemicalProcess(X);
      	TASKPRINT("Y1 = " + to_string(X[0]),task,1);
        TASKPRINT("Y2 = " + to_string(X[1]),task,1);
        TASKPRINT("Y3 = " + to_string(X[2]),task,1);
        TASKPRINT("C1 = " + to_string(X[3]),task,1);
        TASKPRINT("B2 = " + to_string(X[4]),task,1);
        TASKPRINT("B3 = " + to_string(X[5]),task,1);
        TASKPRINT("BP = " + to_string(X[6]),task,1);
        for(int i=0;i<opt.NumEC;i++) TASKPRINT("   equality constraint ( satisfied if  ~0 ): " + to_string_with_precision(EC[i],16),task,1);
        for(int i=0;i<opt.NumIC;i++) TASKPRINT(" inequality constraint ( satisfied if >=0 ): " + to_string_with_precision(IC[i],16),task,1);

        paretofront.insert(paretofront.end(), X.begin(), X.end());
    }
	else if(opt.function==100){//mutually unbiased bases
		opt.AuxParams.resize(3);
		//BEGIN USER INPUT
		opt.AuxParams[0] = 1.0e+3; //PenaltyScale
		opt.AuxParams[1] = 4.; //number of bases
		opt.AuxParams[2] = 4.; //number of vectors==dimensions  
		//END USER INPUT
		int NB = (int)(opt.AuxParams[1]+0.5), NV = (int)(opt.AuxParams[2]+0.5), dim = NV;
		opt.D = NB*NV*dim*2;
		opt.epsf = 1.0e-8;
		opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);    
		for(int i=0;i<opt.D;i++){
			opt.SearchSpaceLowerVec[i] = opt.SearchSpaceMin;
			opt.SearchSpaceUpperVec[i] = opt.SearchSpaceMax;
		}
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 10;
			//opt.pso.increase = 10.;
			//opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = opt.D;
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.loopMax = (int)1.0e+6;
			//opt.pso.reseed = 0.5;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
		else if(opt_ID==104){//GAO
			//begin adjust parameters
			//opt.UpdateSearchSpaceQ = 2;
			opt.evalMax = 1.0e+6;//start low
			opt.gao.runs = 20*opt.D;
			opt.gao.popExponent = 0.3;
			//opt.gao.RandomSearch = true;
			SetDefaultGAOparams(opt);
			//opt.gao.HyperParamsDecayQ = false;
			//opt.gao.HyperParamsSchedule = 2;
			//opt.PickRandomParamsQ = true;
			opt.gao.ShrinkPopulationsQ = true;
			//end adjust parameters
			GAO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.gao.bestf);
			paretofront.insert(paretofront.end(), opt.gao.bestx.begin(), opt.gao.bestx.end());
		}
		else if(opt_ID==106){//CMA
			opt.epsf = 1.0e-12;
			//opt.UpdateSearchSpaceQ = 2;
			opt.BreakBadRuns = 1;//0: don't --- 1: only in case of severe violations --- 2: aggressive termination policy
			SetDefaultCMAparams(opt);
			opt.cma.runs = 5*opt.D;
			opt.cma.popExponent = 7;//population prop to 2^popExponent
			opt.cma.CheckPopVariance = 0.1;//percentage of best populations to assess for termination criterion
			opt.cma.DelayEigenDecomposition = true;
			opt.cma.elitism = true;
			opt.cma.WeightScenario = 2;
			opt.cma.Constraints = 2;
			CMA(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.currentBestf);	
			paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());			
		}		
	}
	else if(opt.function==102){//SineEnvelope
		opt.D = 5;
		opt.epsf = 1.0e-6;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		
		opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.;
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.epsf = 1.0e-14;//opt.epsf = 1.0e-6;
			opt.pso.runs = 10*opt.D;
			opt.pso.increase = 10.;
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = 10*opt.D;
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = 10*opt.D;
			opt.pso.loopMax = (int)1.0e+6;
			opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			if(opt.function==104) for(int d=0;d<opt.D;d++) opt.pso.bestx[d] = SigmoidX(opt.pso.bestx[d],-512.,512.);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
		else if(opt_ID==106){//CMA
			opt.epsf = 1.0e-14;
			opt.BreakBadRuns = 1;
			opt.ReportX = true;
			SetDefaultCMAparams(opt);
            opt.printQ = 1;
			opt.cma.runs = 20*opt.D;//10*opt.D;//5*opt.D;//   /* overkill//14 digits accuracy//10 digits accuracy// */
			opt.cma.popExponent = 7;//6;//5;//
			opt.cma.VarianceCheck = 20*opt.D;//10*opt.D;//5*opt.D;//
			opt.stallCheck = 100*opt.D;
			opt.cma.CheckPopVariance = 0.5;
			opt.cma.PopulationDecayRate = 0.;//3.;//4.;//
			//opt.cma.PickRandomParamsQ = true;
			//opt.cma.DelayEigenDecomposition = true;
			opt.cma.elitism = true;
			opt.cma.WeightScenario = 2;
			if(opt.function==104 || opt.function==200) opt.cma.Constraints = 0;
			else opt.cma.Constraints = 2;
			CMA(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.currentBestf);
			paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
		}
	}
	else if(opt.function==103){//Rana
      opt.D = 7;
      opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
      opt.SearchSpaceMin = -512.; opt.SearchSpaceMax = 512.;

      if(opt_ID==101){//PSO
        SetDefaultPSOparams(opt);
        //begin adjust parameters
        opt.epsf = 1.0e-14;//opt.epsf = 1.0e-6;
        opt.pso.runs = 10*opt.D;
        opt.pso.increase = 10.;
        opt.pso.SwarmDecayRate = 1.7;
        opt.pso.InitialSwarmSize = 10*opt.D;
        InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
        opt.pso.VarianceCheck = 10*opt.D;
        opt.pso.loopMax = (int)1.0e+6;
        opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
        opt.pso.TargetMinEncounters = opt.pso.runs;
        //end adjust parameters
        PSO(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.pso.bestf);
        paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
      }
      else if(opt_ID==106){//CMA
        opt.epsf = 1.0e-14;
        opt.BreakBadRuns = 1;
        opt.ReportX = true;
        SetDefaultCMAparams(opt);
        opt.printQ = 1;
        opt.cma.runs = 100*opt.D;
        opt.cma.popExponent = 5;
        opt.cma.VarianceCheck = 10*opt.D;
        opt.stallCheck = 100*opt.D;
        opt.cma.CheckPopVariance = 0.1;
        opt.cma.PopulationDecayRate = 3.;
        //opt.cma.PickRandomParamsQ = true;
        //opt.cma.DelayEigenDecomposition = true;
        opt.cma.elitism = true;
        opt.cma.WeightScenario = 2;
        opt.cma.Constraints = 1;
        CMA(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.currentBestf);
        paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
      }
    }
	else if(opt.function==104){//UnconstrainedRana
      opt.D = 5;
      opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
      opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.;

      if(opt_ID==101){//PSO
        SetDefaultPSOparams(opt);
        //begin adjust parameters
        opt.epsf = 1.0e-14;//opt.epsf = 1.0e-6;
        opt.pso.runs = 10*opt.D;
        opt.pso.increase = 10.;
        opt.pso.SwarmDecayRate = 1.7;
        opt.pso.InitialSwarmSize = 10*opt.D;
        InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
        opt.pso.VarianceCheck = 10*opt.D;
        opt.pso.loopMax = (int)1.0e+6;
        opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
        opt.pso.TargetMinEncounters = opt.pso.runs;
        //end adjust parameters
        PSO(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.pso.bestf);
        if(opt.function==104) for(int d=0;d<opt.D;d++) opt.pso.bestx[d] = SigmoidX(opt.pso.bestx[d],-512.,512.);
        paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
      }
      else if(opt_ID==106){//CMA
        opt.epsf = 1.0e-14;
        opt.BreakBadRuns = 1;
        opt.ReportX = true;
        SetDefaultCMAparams(opt);
        opt.printQ = 1;
        opt.cma.runs = 100*opt.D;
        opt.cma.popExponent = 5;
        opt.cma.VarianceCheck = 10*opt.D;
        opt.stallCheck = 100*opt.D;
        opt.cma.CheckPopVariance = 0.1;
        opt.cma.PopulationDecayRate = 3.;
        //opt.cma.PickRandomParamsQ = true;
        //opt.cma.DelayEigenDecomposition = true;
        opt.cma.elitism = true;
        opt.cma.WeightScenario = 1;
        opt.cma.Constraints = 0;
        CMA(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.currentBestf);
        if(opt.function==104) for(int d=0;d<opt.D;d++) opt.currentBestx[d] = SigmoidX(opt.currentBestx[d],-512.,512.);
        paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
      }
    }
    else if(opt.function==105){//UnconstrainedEggholder
      	opt.D = 5;
      	opt.inflate = 1;
      	opt.D *= opt.inflate;
      	opt.lowerBound = -512./(double)opt.inflate;
      	opt.upperBound = +512./(double)opt.inflate;
      	opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      	opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
      	opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.;

		if(opt_ID==101){//PSO
          	SetDefaultPSOparams(opt);
            //begin adjust parameters
            RandomizeOptLocation(opt);
        	opt.homotopy = 1;
        	opt.printQ = 1;
        	opt.pso.epsf = 1.0e-14;
        	opt.pso.runs = 10*aux;
        	opt.pso.increase = 10.;
        	opt.pso.SwarmDecayRate = 1.7;
        	opt.pso.InitialSwarmSize = 100*opt.D;
        	InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
        	opt.pso.VarianceCheck = 10*opt.D;
        	opt.pso.loopMax = (int)1.0e+3;
        	opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
        	opt.pso.TargetMinEncounters = opt.pso.runs;
        	//end adjust parameters
        	PSO(opt);
        	paretofront.push_back(opt.nb_eval);
        	paretofront.push_back(opt.pso.bestf);
        	if(opt.function==104) for(int d=0;d<opt.D;d++) opt.pso.bestx[d] = SigmoidX(opt.pso.bestx[d],-512.,512.);
        	paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
        }
      	if(opt_ID==106){//CMA
        	SetDefaultCMAparams(opt);
            //begin adjust parameters
        	RandomizeOptLocation(opt);
            opt.epsf = 1.0e-14;
        	opt.BreakBadRuns = 1;
        	opt.ReportX = true;
        	opt.homotopy = 1;
        	opt.printQ = 1;
        	opt.cma.runs = (int)POW(2.,aux)*10*opt.D/opt.inflate;
        	opt.cma.generationMax = 1000*opt.D;
        	opt.cma.popExponent = (int)(4.*data.RNpos(data.MTGEN));
        	opt.cma.VarianceCheck = 10*opt.D/opt.inflate;
        	opt.stallCheck = 100*opt.D/opt.inflate;
        	opt.cma.CheckPopVariance = 0.1;
        	opt.cma.PopulationDecayRate = 0.;
        	//opt.cma.PickRandomParamsQ = true;
        	//opt.cma.DelayEigenDecomposition = true;
        	opt.cma.elitism = true;
        	opt.cma.WeightScenario = 2;
        	opt.cma.Constraints = 0;
            //end adjust parameters
        	CMA(opt);
        	paretofront.push_back(opt.nb_eval);
        	paretofront.push_back(opt.currentBestf);
        	vector<double> y(opt.currentBestx);
        	if(opt.Rand) ShiftRot(y,opt);
        	for(int d=0;d<opt.D;d++) y[d] = SigmoidX(y[d],opt.lowerBound,opt.upperBound);
        	vector<double> z(opt.D/opt.inflate,0.); for(int j=0;j<z.size();j++) for(int i=0;i<opt.inflate;i++) z[j] += y[opt.inflate*j+i];
        	paretofront.insert(paretofront.end(), z.begin(), z.end());
      	}
    }
	else if(opt.function>100 && opt.function<200){//ShiftedSphere, SineEnvelope, Rana's function, NYFunction, etc.
      opt.D = 5;
      opt.epsf = 1.0e-6;
      opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);


      if(opt.function==101 || opt.function==104/*UnconstrainedRana*/){ opt.SearchSpaceMin = -1.; opt.SearchSpaceMax = 1.; }
      else if(opt.function==102/*SineEnvelope*/){ opt.SearchSpaceMin = -100.; opt.SearchSpaceMax = 100.; }
      else if(opt.function==103){ opt.SearchSpaceMin = -512.; opt.SearchSpaceMax = 512.; }
      else if(opt.function==200){//NYFunction

        opt.AuxIndex = aux;

        //first, provide "TabFunc_NYFunctionLikelihood_XXX.dat" (the ParetoFront data, containing the maximum likelihood estimators L_max(D)) produced with opt.SubFunc==0 --- then switch to opt.SubFunc>0 here and re-run (no further manual adjustments needed anywhere else).
        //BEGIN USER INPUT
        opt.SubFunc = 0;//0: minimize the negative Loglikelihood --- 1: minimize the least-squares difference between probabilities and relative frequencies (opt.AuxParams) --- 2: minimize the negative fidelity --- 3: experiment
        //END USER INPUT
        #pragma omp critical
        {
          if(opt.SubFunc==0) data.AuxString = "Likelihood";
          else if(opt.SubFunc==1) data.AuxString = "LeastSquares";
          else if(opt.SubFunc==2) data.AuxString = "Fidelity";
          else if(opt.SubFunc==3) data.AuxString = "Experiment";
        }


        opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 2.*PI;

        // 			opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
        // 			for(int i=0;i<opt.D;i++) opt.SearchSpaceLowerVec[i] = 0.;
        // 			opt.SearchSpaceUpperVec[0] = PI;
        // 			opt.SearchSpaceUpperVec[1] = PI;
        // 			opt.SearchSpaceUpperVec[2] = PI;
        // 			opt.SearchSpaceUpperVec[3] = PI;
        // 			opt.SearchSpaceUpperVec[4] = 2.*PI;
        // 			opt.SearchSpaceUpperVec[0] = PI/2.;
        // 			opt.SearchSpaceUpperVec[1] = PI/2.;
        // 			opt.SearchSpaceUpperVec[2] = PI;
        // 			opt.SearchSpaceUpperVec[3] = PI;
        // 			opt.SearchSpaceUpperVec[4] = 2.*PI;

        opt.AuxParams.clear();

        //use cases
        //opt.AuxParams = {805., 1593., 836., 803., 1705., 851., 841., 1732., 834.};//case1: expected f_min = ?
        //opt.AuxParams = {842.,854.,1615.,799.,800.,1714.,892.,813.,1671.};//case2: expected f_min = 2.1377889507324106
        //opt.AuxParams = {654.,558.,1169.,967.,2313.,521.,1260.,1152.,1406.};//case3: expected f_min = 2.0945855608529884
        //opt.AuxParams = {666.8670613173642, 542.267850618758, 1197.916666666669, 975.7351656601277, 2352.755640130231, 557.5922586427129, 1169.8977730225154, 1167.4765092510274, 1369.4910746906237};//case3b: expected f_min = 2.0945855608529884
        //opt.AuxParams = {1284.3882311377308, 382.2784355289398, 1614.5833333333364, 2031.596817348357, 893.4573530598154, 1335.102843552896, 579.0467062444684, 454.2324566807027, 1425.3138231137752};//case4: expected f_min = 2.074295099297655
        //opt.AuxParams = {618, 594, 1176, 995, 2316, 562, 1241, 1175, 1323};//case5: expected f_min = ?
        //opt.AuxParams = {1307, 144, 712, 473, 103, 756, 612, 133, 760};//case6: expected f_min = [1.8155203662194874 or 1.922065651222289]
        //opt.AuxParams = {1444, 148, 617, 519, 95, 777, 600, 123, 677};//case7: expected f_min = [1.8151322728878325 or 1.926019361486259]
        //opt.AuxParams = {1342, 169, 667, 490, 99, 750, 594, 103, 786};//case8: expected f_min = [1.8193364393587443 or 1.9260567253422933]
        if(task.Aux>0.5){//from TabFunc_NYFunction_XXX.dat
          opt.threads = 1;
          opt.reportQ = 0;
          opt.AuxParams = task.refData[opt.AuxIndex];
        }

        opt.AuxVal = accumulate(opt.AuxParams.begin(),opt.AuxParams.end(),0.);
        // 			TASKPRINT("opt.AuxParams=" + vec_to_CommaSeparatedString_with_precision(opt.AuxParams,16),task,1);
        // 			VecFact(opt.AuxParams,1./opt.AuxVal);//ToDo: why does this not work??????????
        // 			TASKPRINT("opt.AuxParams=" + vec_to_CommaSeparatedString_with_precision(opt.AuxParams,16),task,1);
        for(int i=0;i<opt.AuxParams.size();i++) opt.AuxParams[i] /= opt.AuxVal;
        if(task.Aux<0.5) TASKPRINT("opt.AuxParams=" + vec_to_CommaSeparatedString_with_precision(opt.AuxParams,16),task,1);

        //begin test ground

        //Camilla's order
        //vector<double> x = {/*a=*/PI/4.,/*f0=*/0.,/*f1=*/0.,/*t0=*/0.,/*t1=*/PI/2.};//case1(a)
        //vector<double> x = {2.331710128012299,2.595433049161852,5.776511197655469,0.5264971896014186,0.5205234056820061};//case1(b)
        //TASKPRINT("case1: {a,f0,f1,t0,t1}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        //Pranjal's order
        // 			vector<double> x = {/*theta1*/0., /*theta2*/PI/2., /*phi1*/0., /*phi2*/0., /*alpha*/PI/4.};//case2
        // 			TASKPRINT("case2: {theta1,theta2,phi1,phi2,alpha}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        // 			vector<double> x = {/*theta1*/PI/6., /*theta2*/PI/2.+PI/6., /*phi1*/PI/4., /*phi2*/PI/2., /*alpha*/PI/6.};//case3
        // 			TASKPRINT("case3: {theta1,theta2,phi1,phi2,alpha}={PI/6.,PI/2.+PI/6.,PI/4.,PI/2.,PI/6.}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        // 			x = {0.5235987755982988,2.094395102393195,0.7853981633974483,1.570796326794897,0.5235987755982988};
        // 			TASKPRINT("case3: {theta1,theta2,phi1,phi2,alpha}={0.5235987755982988,2.094395102393195,0.7853981633974483,1.570796326794897,0.5235987755982988}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        // 			x = {2.6072033366248,4.1823366657766,3.9196739422848,1.5574885302675,0.54017337241981};
        // 			TASKPRINT("case3(MIT, old code): {theta1,theta2,phi1,phi2,alpha}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        // 			vector<double> x = {0.5343893173506,1.0407440166693,0.77808130199747,4.6990812421347,2.6014192818704};
        // 			TASKPRINT("case3(MIT, old code): {theta1,theta2,phi1,phi2,alpha}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        //             vector<double> x = {1.33073649,0.17024566,-2.9826044,-0.93498342,0.21936886};
        //             TASKPRINT("case6(MIT): {theta1,theta2,phi1,phi2,alpha}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
        //
        //             usleep(10000000);
        //end test ground
      }


      if(opt_ID==101){//PSO
        SetDefaultPSOparams(opt);
        //begin adjust parameters
        opt.epsf = 1.0e-14;//opt.epsf = 1.0e-6;
        opt.pso.runs = 10*opt.D;
        opt.pso.increase = 10.;
        opt.pso.SwarmDecayRate = 1.7;
        opt.pso.InitialSwarmSize = 10*opt.D;
        InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
        opt.pso.VarianceCheck = 10*opt.D;
        opt.pso.loopMax = (int)1.0e+6;
        opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
        opt.pso.TargetMinEncounters = opt.pso.runs;
        //end adjust parameters
        PSO(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.pso.bestf);
        if(opt.function==104) for(int d=0;d<opt.D;d++) opt.pso.bestx[d] = SigmoidX(opt.pso.bestx[d],-512.,512.);
        paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
      }
      else if(opt_ID==106){//CMA
        opt.epsf = 1.0e-14;
        opt.BreakBadRuns = 1;
        opt.ReportX = true;
        SetDefaultCMAparams(opt);
        opt.printQ = -2;
        opt.cma.runs = 20*opt.D;//10*opt.D;//5*opt.D;//   /* overkill//14 digits accuracy//10 digits accuracy// */
        opt.cma.popExponent = 7;//6;//5;//
        opt.cma.VarianceCheck = 20*opt.D;//10*opt.D;//5*opt.D;//
        opt.stallCheck = 100*opt.D;
        opt.cma.CheckPopVariance = 0.5;
        opt.cma.PopulationDecayRate = 0.;//3.;//4.;//
        //opt.cma.PickRandomParamsQ = true;
        //opt.cma.DelayEigenDecomposition = true;
        opt.cma.elitism = true;
        opt.cma.WeightScenario = 2;
        if(opt.function==104 || opt.function==200) opt.cma.Constraints = 0;
        else opt.cma.Constraints = 2;
        CMA(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.currentBestf);
        if(opt.function==104) for(int d=0;d<opt.D;d++) opt.currentBestx[d] = SigmoidX(opt.currentBestx[d],-512.,512.);
        paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
        if(opt.function==200){
          paretofront.insert(paretofront.end(), opt.AuxParams.begin(), opt.AuxParams.end());
          opt.finalcalc = true;
          double dummy = NYFunction(opt.currentBestx,opt);
          paretofront.insert(paretofront.end(), opt.AuxVec.begin(), opt.AuxVec.end());
        }
      }

      // 		if(opt.function==200 && task.Aux<0.5){
      // 			//shift t0,t1,f0,f1,a into [0,2Pi]
      // // 			for(int i=2;i<2+opt.D;i++){
      // // 				double m = paretofront[i]/(2.*PI);
      // // 				if(m>1.){
      // // 					int n = (int)m;
      // // 					paretofront[i] -= (double)n*2.*PI;
      // // 				}
      // // 				else if(m<0.){
      // // 					int n = (int)m+1;
      // // 					paretofront[i] += (double)n*2.*PI;
      // // 				}
      // // 			}
      // // 			//shift t0,t1,a into [0,Pi]
      // // 			if(paretofront[2]>PI) paretofront[2] -= PI;
      // // 			if(paretofront[3]>PI) paretofront[3] -= PI;
      // // 			if(paretofront[6]>PI) paretofront[6] -= PI;
      // 			vector<double> x = opt.currentBestx;
      //
      // 			//vector<double> y = {PI/6.,PI/2.+PI/6.,PI/4.,PI/2.,PI/6.};//case3b
      // 			vector<double> y = {PI/12.,PI/2.-PI/12.,PI/4.,PI/2.,PI/3.};//case4
      //
      // 			if(Norm(VecDiff(x,y))<1.0e-6){
      // 				TASKPRINT("Bingo!!!" + vec_to_CommaSeparatedString_with_precision(x,16),task,1);
      // 				usleep(10000000);
      // 			}
      // 			TASKPRINT("NYFunction: final check {theta1,theta2,phi1,phi2,alpha}=" + vec_to_CommaSeparatedString_with_precision(x,16) + "\n ---> NYFunction = " + to_string_with_precision(NYFunction(x,opt),16),task,1);
      // 		}
    }
    else if(opt.function==201){//Itai Arad's quantum circuit, QuantumCircuitIA
      	opt.D = 12;
      	opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      	opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
      	opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 2.*PI;

      	if(opt_ID==106){//CMA
        	SetDefaultCMAparams(opt);
            //begin adjust parameters
            opt.epsf = 1.0e-14;
        	opt.BreakBadRuns = 1;
        	opt.ReportX = true;
        	opt.homotopy = 1;
        	opt.printQ = 1;
        	opt.cma.runs = 1;//(int)POW(2.,aux)*10*opt.D;
        	opt.cma.generationMax = 50*opt.D;
        	opt.cma.popExponent = 0;//(int)(4.*data.RNpos(data.MTGEN));
        	opt.cma.VarianceCheck = opt.D;//10*opt.D;
        	opt.stallCheck = 100*opt.D;
        	opt.cma.CheckPopVariance = 0.5;
        	opt.cma.PopulationDecayRate = 0.;
        	//opt.cma.PickRandomParamsQ = true;
        	//opt.cma.DelayEigenDecomposition = true;
        	opt.cma.elitism = true;
        	opt.cma.WeightScenario = 2;
        	opt.cma.Constraints = 0;
            //end adjust parameters
        	CMA(opt);
        	paretofront.push_back(opt.nb_eval);
        	paretofront.push_back(opt.currentBestf);
        	paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
      	}
    }
	else if(opt.function==300){//Rastrigin constrained on polytope

      opt.inflate = 10;
      opt.AuxVal = 3.;//equals sum_i n_i
      opt.AuxParams.clear(); opt.AuxParams.resize(5);
      opt.AuxParams[0] = 16.;//number of local minima in each dimension, see ConstrainedRastrigin.nb, (for some strange reason 10. is problematic for some D)
      opt.AuxParams[1] = 0.;//lower bound of variables
      opt.AuxParams[2] = 2./(double)opt.inflate;//upper bound of variables
      opt.AuxParams[3] = 10.;//scale of Sigmoid
      opt.AuxParams[4] = 1.0e-8;//threshold for acceptable violation of equality constraints and variable bounds
      opt.D = 6*opt.inflate;

      opt.AuxMat.clear(); opt.AuxMat.resize(0);

      opt.epsf = 1.0e-20;
      opt.SearchSpaceMin = -3.5; opt.SearchSpaceMax = 3.5;//see ConstrainedRastrigin.nb
      opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
      opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);

      if(opt_ID==101){//PSO
        SetDefaultPSOparams(opt);
        RandomizeOptLocation(opt);
        //begin adjust parameters
        opt.pso.epsf = 1.0e-14;
        opt.pso.runs = 10*opt.D;
        opt.pso.increase = 1.*(double)opt.D;
        opt.pso.SwarmDecayRate = 1.7;
        opt.pso.InitialSwarmSize = 1*opt.D;
        InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
        opt.pso.VarianceCheck = 10*opt.D;
        opt.pso.loopMax = (int)1.0e+6;
        opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
        opt.pso.TargetMinEncounters = opt.pso.runs;
        //end adjust parameters
        PSO(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.pso.bestf);
        if(opt.function==104) for(int d=0;d<opt.D;d++) opt.pso.bestx[d] = SigmoidX(opt.pso.bestx[d],-512.,512.);
        paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
        vector<double> y(opt.pso.bestx);
        if(opt.Rand) ShiftRot(y,opt);
        if(!EqualityConstraintBoundedVariables(y,opt)) TASKPRINT("Optimize: Warning !!! Equality Constraints violated",task,1);
        vector<double> z(opt.D/opt.inflate,0.); for(int j=0;j<z.size();j++) for(int i=0;i<opt.inflate;i++) z[j] += y[opt.inflate*j+i];
        paretofront.insert(paretofront.end(), z.begin(), z.end());
      }
      else if(opt_ID==106){//CMA
        //opt.UpdateSearchSpaceQ = 2;
        opt.BreakBadRuns = 1;
        opt.ReportX = true;
        //opt.anneal = 1.0e-1; opt.AnnealType = 0;
        SetDefaultCMAparams(opt);
        RandomizeOptLocation(opt);
        opt.printQ = 1;
        opt.cma.runs = (10-2*aux)*opt.D;
        //opt.cma.generationMax = 1000*opt.D;
        opt.cma.popExponent = (int)(0.3*(double)aux);
        opt.cma.VarianceCheck = 10*opt.D/opt.inflate;
        opt.stallCheck = 10*opt.D/opt.inflate;//opt.cma.generationMax;//
        opt.cma.CheckPopVariance = 0.1;
        opt.cma.PopulationDecayRate = 3.;
        //opt.cma.PickRandomParamsQ = true;
        //opt.cma.DelayEigenDecomposition = true;
        opt.cma.elitism = true;
        opt.cma.WeightScenario = 2;
        opt.cma.Constraints = 0;//0: No Contraints --- 1: Box Constraints --- 2: Penalty (smooth square of box repair distance) --- 3: Penalty (sharp sqrt of box repair distance) --- 4: Penalty (constant 1.0e+100)
        CMA(opt);
        paretofront.push_back(opt.nb_eval);
        paretofront.push_back(opt.currentBestf);
        paretofront.insert(paretofront.end(), opt.currentBestx.begin(), opt.currentBestx.end());
        vector<double> y(opt.currentBestx);
        if(opt.Rand) ShiftRot(y,opt);
        if(!EqualityConstraintBoundedVariables(y,opt)) TASKPRINT("Optimize: Warning !!! Equality Constraints violated",task,1);
        vector<double> z(opt.D/opt.inflate,0.); for(int j=0;j<z.size();j++) for(int i=0;i<opt.inflate;i++) z[j] += y[opt.inflate*j+i];
        paretofront.insert(paretofront.end(), z.begin(), z.end());
        MatrixToFile(opt.AuxMat,"mpDPFT_AuxMat.dat",8);
      }
    }
	else if(opt.function>=1000 && opt.function<=1001){//DFTe fit to consumer-resource quasi-potential QPot, see DiffEq_quasi-potential_DFTe.nb
		//function --- 1000: Dakos2012 --- 1001: Nolting2016
		if(opt.function==1000) opt.D = 10;
		if(opt.function==1001) opt.D = 7;//10;//
		opt.epsf = 1.0e-7;
		opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		opt.SearchSpaceLowerVec[0] = -0.;//inertia
		opt.SearchSpaceUpperVec[0] = 0.;
		if(opt.function==1000){
			opt.SearchSpaceLowerVec[1] = -20.;
			opt.SearchSpaceUpperVec[1] = 20.;
			for(int i=2;i<opt.D;i++){
				opt.SearchSpaceLowerVec[i] = -12.;
				opt.SearchSpaceUpperVec[i] = 12.;
			}
		}
		if(opt.function==1001){
			for(int i=1;i<opt.D;i++){
				opt.SearchSpaceLowerVec[i] = -3.;
				opt.SearchSpaceUpperVec[i] = 3.;
			}		
		}
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 10000;
			opt.pso.increase = 10.;
			opt.pso.InitialSwarmSize = 10;//10*opt.D;//opt.D;//
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = 10*opt.D;
			opt.pso.loopMax = (int)5.0e+5;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
	}
	else if(opt.function>=1002 && opt.function<=1005){//fit time series data {c,x} to DE/DFTe
		//function --- 1002: Dakos2012, DEfit --- 1003: Dakos2012, DFTefit --- 1004: Nolting2016, DEfit --- 1005: Nolting2016, DFTefit
		bool DFTefit = false;
		if(opt.function==1003 || opt.function==1005) DFTefit = true;
		opt.D = 3;
		if(DFTefit){
			if(opt.function>=1002 && opt.function<=1003) opt.D = 10;
			else if(opt.function>=1004 && opt.function<=1005) opt.D = 7;
		}
		opt.epsf = 1.0e-12;
		//opt.UpdateSearchSpaceQ = 2;
		opt.SearchSpaceMin = 0.; opt.SearchSpaceMax = 0.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);    
		for(int i=0;i<opt.D;i++)
		{
			if(DFTefit && i==0){//mInverse
				opt.SearchSpaceLowerVec[i] = -10.;
				opt.SearchSpaceUpperVec[i] = 10.;
			}		
			else if(opt.function>=1002 && opt.function<=1003){
				if(i==1){//carrying capacity
					opt.SearchSpaceLowerVec[i] = -20.;
					opt.SearchSpaceUpperVec[i] = 20.;
				}
				else{
					opt.SearchSpaceLowerVec[i] = -12.;
					opt.SearchSpaceUpperVec[i] = 12.;
				}
			}
			else if(opt.function>=1004 && opt.function<=1005){
				opt.SearchSpaceLowerVec[i] = -3.;
				opt.SearchSpaceUpperVec[i] = 3.;
			}
		
			if(opt.UpdateSearchSpaceQ==2)
			{
				opt.VariableLowerBoundaryQ.push_back(true);
				opt.VariableUpperBoundaryQ.push_back(true);
			}
		}
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 10; if(DFTefit) opt.pso.runs = 100000;
			opt.pso.increase = 1.;
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = 5*opt.D;
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = 10*opt.D;
			opt.pso.loopMax = (int)5.0e+5;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
	}
	else if(opt.function==-100){//custom objective function (defined in Plugin_OPT.cpp)
		opt.D = 5;
		opt.epsf = 1.0e-12;
		opt.SearchSpaceMin = -512.; opt.SearchSpaceMax = 512.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 10;
			opt.pso.increase = 2.;
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = max(40,opt.D);
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = max(10,min(50,opt.D));
			//opt.pso.MaxLinks = 3;
			//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
			//opt.pso.elitism = 0.1;
			//opt.pso.PostProcessQ = 1;
			opt.pso.loopMax = (int)1.0e+6;
			//opt.pso.reseed = 0.25;
			//opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
			//opt.pso.alpha = 0.;
			//opt.BreakBadRuns = 1;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
	}
	else{//default
		opt.D = 5;
		opt.epsf = 1.0e-12;
		opt.SearchSpaceMin = -512.; opt.SearchSpaceMax = 512.;
		opt.SearchSpaceLowerVec.clear(); opt.SearchSpaceLowerVec.resize(opt.D);
		opt.SearchSpaceUpperVec.clear(); opt.SearchSpaceUpperVec.resize(opt.D);
		
		if(opt_ID==101){//PSO
			SetDefaultPSOparams(opt);
			//begin adjust parameters
			opt.pso.runs = 10;
			opt.pso.increase = 2.;
			opt.pso.SwarmDecayRate = 1.7;
			opt.pso.InitialSwarmSize = max(40,opt.D);
			InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
			opt.pso.VarianceCheck = max(10,min(50,opt.D));
			//opt.pso.MaxLinks = 3;
			//opt.pso.eval_max_init = (int)1.0e+1*opt.D;
			//opt.pso.elitism = 0.1;
			//opt.pso.PostProcessQ = 1;
			opt.pso.loopMax = (int)1.0e+6;
			//opt.pso.reseed = 0.25;
			//opt.pso.CoefficientDistribution = 5; opt.pso.AbsAcc = 0.01;
			//opt.pso.alpha = 0.;
			//opt.BreakBadRuns = 1;
			opt.pso.TargetMinEncounters = opt.pso.runs;
			//end adjust parameters
			PSO(opt);
			paretofront.push_back(opt.nb_eval);
			paretofront.push_back(opt.pso.bestf);
			paretofront.insert(paretofront.end(), opt.pso.bestx.begin(), opt.pso.bestx.end());
		}
		
	}
	
	if(task.Aux<0.5) TASKPRINT(opt.control.str(),task,0);

    cout << "------------------------- return paretofront ------------------------- " << endl;
	
	return paretofront;
}

void testNN(datastruct &data){

  //training data
  int NumInputVars = 2, Dim = NumInputVars+1, NumSamples = 1000, MaxNumWeights0 = min(100,(int)(0.1*(double)NumSamples));
  int MaxNumNeurons1 = (int)((double)MaxNumWeights0/(double)NumInputVars), NumNeurons1 = MaxNumNeurons1;
  double c = 0.5*(double)NumInputVars;
  int MaxNumNeurons2 = (int)(sqrt((double)MaxNumWeights0+c*c)-c), NumNeurons2 = MaxNumNeurons2;
  cout << "testNN --- MaxNumNeurons1 = " << MaxNumNeurons1 << " --- MaxNumNeurons2 = " << MaxNumNeurons2 << endl;
  vector<vector<double>> TrainingData(NumSamples);
  for(int sample=0;sample<NumSamples;sample++){
    for(int i=0;i<NumInputVars;i++) TrainingData[sample].push_back(3.*data.RN(data.MTGEN));
    for(int i=NumInputVars;i<Dim;i++) TrainingData[sample].push_back(1.+exp(-(TrainingData[sample][0]*TrainingData[sample][0]+TrainingData[sample][1]*TrainingData[sample][1])));
    //for(int i=NumInputVars;i<Dim;i++) TrainingData[sample].push_back(TrainingData[sample][0]*TrainingData[sample][1]);
    //for(int i=NumInputVars;i<Dim;i++) TrainingData[sample].push_back(TrainingData[sample][0]+TrainingData[sample][1]);
  }

  MaxNumNeurons1 = 32; MaxNumNeurons2 = (int)sqrt((double)MaxNumNeurons1);
  
  //default NN parameters
  double wstep = 0.005;
  double weightDecay = 1.0e-3;
  int NumRestarts = 0;
  ae_int_t maxits = 100*NumInputVars;//epochs
  
  //run Test:
  int count = 0, NumTests = 10, BestCount = 0;
  vector<double> history;
  vector<vector<double>> NNparams;
  
  bool TrainingComplete = false;
  while(count<NumTests || TrainingComplete){
    int NumHiddenLayers;
    vector<double> params(5);
    
    if(TrainingComplete){//report best network
      count = 0;
      for(auto i: sort_indices(history)){
	count++;
	cout << "\n BestPrediction [MeanRelDiff] = " << history[i] << "\n @ i=" << i << " --- #HiddenLayers = " << NNparams[i][4] << " --- weightDecay = " << NNparams[i][0] << " --- NumRestarts = " << NNparams[i][1] << endl;
	if(count==1) BestCount = i;
	if(count==10) break;
      }
      weightDecay = NNparams[BestCount][0];
      NumRestarts = NNparams[BestCount][1];
      NumNeurons1 = NNparams[BestCount][2];
      NumNeurons2 = NNparams[BestCount][3];
      NumHiddenLayers = NNparams[BestCount][4];
      count = NumTests;
    }
    else{//choose random network parameters
      weightDecay = pow(10.,-6.+0.*data.RNpos(data.MTGEN));
      NumRestarts = 10.+(int)(90.*data.RNpos(data.MTGEN));
      NumNeurons1 = MaxNumNeurons1;//max(2,(int)((0.1+0.9*data.RNpos(data.MTGEN))*(double)MaxNumNeurons1));
      NumNeurons2 = MaxNumNeurons2;//max(2,(int)((0.1+0.9*data.RNpos(data.MTGEN))*(double)MaxNumNeurons2));
      params[0] = weightDecay; params[1] = (double)NumRestarts; params[2] = (double)NumNeurons1; params[3] = (double)NumNeurons2;
    }
    
    //create trainer
    mlptrainer trn;
    mlpcreatetrainer(NumInputVars, 1, trn);

    // specify training set, test set, and training parameters
    vector<ptrdiff_t> TrainingSetIndices, TestSetIndices;
    for(int i=0;i<NumSamples;i++) if(data.RNpos(data.MTGEN)<0.2) TestSetIndices.push_back(i); else TrainingSetIndices.push_back(i);
    int NumTrainingSamples = TrainingSetIndices.size();
    int NumTestSamples = TestSetIndices.size();
    real_2d_array xy; xy.setlength(NumTrainingSamples,Dim);
    real_2d_array xyTest; xyTest.setlength(NumTestSamples,Dim);
    for(int i=0;i<NumTrainingSamples;i++) for(int j=0;j<Dim;j++) xy[i][j] = TrainingData[TrainingSetIndices[i]][j];
    for(int i=0;i<NumTestSamples;i++) for(int j=0;j<Dim;j++) xyTest[i][j] = TrainingData[TestSetIndices[i]][j];
    mlpsetdataset(trn, xy, NumTrainingSamples);
    mlpsetdecay(trn, weightDecay);
    mlpsetcond(trn, wstep, maxits);

    //create NN with linear output
    vector<multilayerperceptron> MLP(3);
    mlpcreate0(NumInputVars, 1, MLP[0]);
    mlpcreate1(NumInputVars, NumNeurons1, 1, MLP[1]);
    mlpcreate2(NumInputVars, NumNeurons2, NumNeurons2, 1, MLP[2]);
    mlpreport rep;
    modelerrors mdlerrs;

    real_1d_array y; y.setlength(1);
    vector<double> output(3);
    if(!TrainingComplete) cout << endl << "NN " << count << ": weightDecay = " << weightDecay << " --- NumRestarts " << NumRestarts << endl;
    int mStart = 0, mEnd = 2;
    if(TrainingComplete){ mStart = NumHiddenLayers; mEnd = mStart; }
    for(int m=mStart;m<=mEnd;m++){
      cout << "--- Result for #HiddenLayers = " << m; if(m==1) cout << " --- NumHiddenNeurons = " << NumNeurons1; else if(m==2) cout << " --- NumHiddenNeurons/layer = " << NumNeurons2;
      if(TrainingComplete) cout << " @ NN " << BestCount << " (retrained)";
      cout << endl;
      mlptrainnetwork(trn, MLP[m], NumRestarts, rep);//train network
      vector<double> RelDiffVec;
      for(int i=0;i<NumTestSamples;i++){//validate network on test set
	real_1d_array xTest; xTest.setlength(NumInputVars);
	for(int j=0;j<NumInputVars;j++) xTest[j] = xyTest[i][j];
	mlpprocess(MLP[m], xTest, y);
	RelDiffVec.push_back(RelDiff(y[0],xyTest[i][NumInputVars])); 
      }
      output[m] = VecAv(RelDiffVec);
      cout << "MeanRelDiff (MaxRelDiff) on test data set = " << output[m] << " (" << *max_element(RelDiffVec.begin(),RelDiffVec.end()) << ")" << endl;
      
      //export array of NN-prediction
      int L = 100;
      vector<vector<double>> NNpred(L);
      for(int l=0;l<L;l++){
	NNpred[l].resize(L);
	real_1d_array xTest; xTest.setlength(2);
	xTest[0] = -3.+(double)l/(double)L*6;
	for(int k=0;k<L;k++){
	  xTest[1] = -3.+(double)k/(double)L*6;
	  mlpprocess(MLP[m], xTest, y);
	  NNpred[l][k] = y[0];
	}
      }
      MatrixToFile(NNpred,"NN_prediction.dat",16);
    }
    for(auto i: sort_indices(output)){
      params[4] = i;//optimum NumHiddenLayers
      history.push_back(output[i]);
      break;
    }
    NNparams.push_back(params);
    count++;
    if(TrainingComplete) TrainingComplete = false;
    if(count==NumTests) TrainingComplete = true;
  }

  cout << "testNN complete" << endl;
  usleep(1000000000);

}

void OptimizeNN(datastruct &data){

  OPTstruct opt;
  SetDefaultOPTparams(opt);
  opt.reportQ = 0;
  opt.printQ = 1;
  opt.ExportIntermediateResults= true;
  opt.UpdateSearchSpaceQ = 1;
  opt.reportQ = 2;
  opt.threads = (int)omp_get_max_threads();//1;
  opt.function = -20;
  opt.SearchSpaceLowerVec.clear();
  opt.SearchSpaceUpperVec.clear();
  opt.VariableLowerBoundaryQ.clear();
  opt.VariableUpperBoundaryQ.clear();  
  
  //par[0]: NumWeights --- total number of links within the NN
  opt.SearchSpaceLowerVec.push_back(2.); opt.SearchSpaceUpperVec.push_back(100.);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(true);
  //par[1]: NumRestarts --- specified number of random restarts (>=0.) are performed, best network is chosen after training
  opt.SearchSpaceLowerVec.push_back(1.); opt.SearchSpaceUpperVec.push_back(1.);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(false);
  //par[2]: MaxNumRepeats --- number of repeats for determining mean and variance of network error
  opt.SearchSpaceLowerVec.push_back(100.); opt.SearchSpaceUpperVec.push_back(100.);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(false);
  //par[3]: WeightChangeStopCriterion --- stop if change of weights falls below WeightChangeStopCriterion
  opt.SearchSpaceLowerVec.push_back(0.); opt.SearchSpaceUpperVec.push_back(0.);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(true);
  //par[4]: epochs --- maximum number of training iterations in each NN instance
  opt.SearchSpaceLowerVec.push_back(1000.); opt.SearchSpaceUpperVec.push_back(1000.);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(true);
  //par[5]: WeightDecay --- increase to penalize NN complexity (avoid overfitting), alglib-recommendation: 0.001 ... 100
  opt.SearchSpaceLowerVec.push_back(0.); opt.SearchSpaceUpperVec.push_back(0.01);
  opt.VariableLowerBoundaryQ.push_back(false); opt.VariableUpperBoundaryQ.push_back(true);
  
  opt.D = opt.SearchSpaceLowerVec.size();
  cout << "OptimizeNN: Initial search space" << endl;
  for(int d=0;d<opt.D;d++) cout << std::setprecision(16) << "[ " << opt.SearchSpaceLowerVec[d] << " " << opt.SearchSpaceUpperVec[d] << "]" << endl;
  
  SetDefaultNNparams(2,TASK.NN);
  TASK.NN.par.clear(); TASK.NN.par.resize(opt.D);

  SetDefaultPSOparams(opt);
  opt.pso.InitialSwarmSize = opt.threads;
  InitializeSwarmSizeDependentVariables(opt.pso.InitialSwarmSize,opt);
  opt.pso.loopMax = 100;
  opt.pso.epsf = 1.0e-6;
  //opt.pso.AlwaysUpdateSearchSpace = true;
  PSO(opt);
  cout << "NN optimized:  bestf = " << to_string_with_precision(opt.pso.bestf,16) << " --- bestx = " << vec_to_str_with_precision(opt.pso.bestx,16) << endl;
  
  //export network to enable external NNprocess(), can only be done safely in single-thread region (hence, ReproductionQ->true)
  cout << "export NN:" << endl;
  TASK.NN.print = 2;
  TASK.NN.ReproductionQ = true;
  double bestf = ProduceNNinstance(opt.pso.bestx);
  cout << "NNprocess({{0,0}},TASK.NN) = " << NNprocess({{0,0}},TASK.NN) << endl;
  
  usleep(2000000000);
}

void SetDefaultNNparams(int dim, MLPerceptron &NN){
  InitRandomNumGeneratorNN(NN);
  
  //data input
  NN.TrainingDataType = 0;// --- 0: scalar objective function (in-code)
  NN.NumInputVars = dim;//dimensionality of support of scalar objective function
  NN.NumTrainingSamples = 1000;//training samples
  NN.NumValidationSamples = 1000;//validation samples
  NN.NumSamples = NN.NumTrainingSamples + NN.NumValidationSamples;
  NN.TestSetFraction = (double)NN.NumValidationSamples/((double)NN.NumSamples);  
  
  //default NN parameters
  NN.NumHiddenLayers = 2;
  
  //validation- and statistical parameters
  NN.TargetCounter = 5;
  NN.TargetSigmaFraction = 0.5;//target of fluctation of mean network error, hit TargetCounter times consecutively
  
  //control parameters
  NN.print = 0;//--- 0: nothing --- 1: only final result --- 2: all & export NN_prediction to file
  NN.ReproductionQ = false;//false for training, true for reproduction with one instance of par
  
  //auxilliary variables
  int NumSeeds = 10;
  NN.Seeds.clear();
  for(int d=0;d<NN.NumInputVars;d++){//for each dimension...
    for(int i=0;i<NumSeeds;i++){//...produce NumSeeds random centres of Gaussians...
      vector<double> seed(0);
      for(int j=0;j<NN.NumInputVars;j++) seed.push_back(-5.+10.*TASK.NN.RNpos(TASK.NN.MTGEN));
      NN.Seeds.push_back(seed);
    }
  }
  cout << "SetDefaultNNparams: " << endl;
  for(int i=0;i<NN.Seeds.size();i++) cout << vec_to_str(NN.Seeds[i]) << " " << NNtestFunction(NN.Seeds[i],NN) << endl;
}

double NNtestFunction(vector<double> &x, MLPerceptron &NN){
  //(shifted) isotropic Gaussian
  return 10.*exp(-ScalarProduct(x,x));
  
  //summed random-centre Gaussians
//   double res = 0.;
//   for(int i=0;i<NN.Seeds.size();i++){
//     vector<double> y = VecDiff(x,NN.Seeds[i]);//...evaluate them at the focal position x...
//     double sign = 1.; if(i%2==0) sign = -1.;
//     res += sign*exp(-ScalarProduct(y,y));//...and add everything up
//   }
//   return res;
  
  //white noise
  //return 1.+NN.RNpos(NN.MTGEN);
}

double NNprocess(vector<double> x, MLPerceptron &NN){
  real_1d_array y; y.setlength(1);
  mlpprocess(NN.mlp, VecTOreal_1d_array(x), y);
  return y[0]*NN.spread+NN.shift;
}

double ProduceNNinstance(vector<double> par){
  TASK.NN.par = par; 

  //load training data
  int dim = TASK.NN.NumInputVars, Dim = dim+1, S = TASK.NN.NumSamples;
  vector<vector<double>> Data(S); for(int s=0;s<S;s++) Data[s].resize(Dim);
  vector<vector<double>> DataTransposed(Dim); for(int i=0;i<Dim;i++) DataTransposed[i].resize(S);
  if(TASK.NN.TrainingDataType==0){
    for(int s=0;s<S;s++){
      for(int i=0;i<dim;i++){
	Data[s][i] = -5.+10.*TASK.NN.RNpos(TASK.NN.MTGEN);
	DataTransposed[i][s] = Data[s][i];
      }
      double f = NNtestFunction(Data[s],TASK.NN);
      Data[s][dim] = f;
      DataTransposed[dim][s] = f;
    }
  }
  
  //standardize sample function values to [0,1]
  double shift = *min_element(DataTransposed[dim].begin(),DataTransposed[dim].end());
  double spread = *max_element(DataTransposed[dim].begin(),DataTransposed[dim].end()) - shift;
  for(int s=0;s<S;s++){
    Data[s][dim] = (Data[s][dim]-shift)/spread;
    DataTransposed[dim][s] = (DataTransposed[dim][s]-shift)/spread;
  }
  
  //split samples into training and test set
  vector<ptrdiff_t> TrainingSetIndices, TestSetIndices;
  for(int i=0;i<S;i++) if(TASK.NN.RNpos(TASK.NN.MTGEN)<TASK.NN.TestSetFraction) TestSetIndices.push_back(i); else TrainingSetIndices.push_back(i);
  int NumTestSamples = TestSetIndices.size();
  int NumTrainingSamples = S - NumTestSamples;
  real_2d_array xy; xy.setlength(NumTrainingSamples,Dim);
  real_2d_array xyTest; xyTest.setlength(NumTestSamples,Dim);
  for(int i=0;i<NumTrainingSamples;i++) for(int j=0;j<Dim;j++) xy[i][j] = Data[TrainingSetIndices[i]][j];
  for(int i=0;i<NumTestSamples;i++) for(int j=0;j<Dim;j++) xyTest[i][j] = Data[TestSetIndices[i]][j];  
  
  //fetch network parameters
  int NumWeights = (int)par[0];
  int NumRestarts = (int)par[1];
  int MaxNumRepeats = (int)par[2]; if(TASK.NN.ReproductionQ) MaxNumRepeats = 1;
  double WeightChangeStopCriterion = par[3];
  int epochs = (int)par[4];
  double WeightDecay = par[5];
  
  //build NN
  int NumNeurons = 0;
  if(TASK.NN.NumHiddenLayers==1) NumNeurons = NumWeights; else if(TASK.NN.NumHiddenLayers==2) NumNeurons = (int)sqrt((double)NumWeights)+1;
  if(TASK.NN.print>1) cout << "NN: NumNeurons = " << NumNeurons << " --- on NumHiddenLayers = " << TASK.NN.NumHiddenLayers << endl;
  int r, targetCounter = 0;
  vector<double> history(0);
  for(r=0;r<MaxNumRepeats;r++){
    if(TASK.NN.print>1) cout << endl << "NN " << r+1 << ": NumWeights = " << NumWeights << " --- NumRestarts " << NumRestarts  << " --- MaxNumRepeats " << MaxNumRepeats << " --- WeightChangeStopCriterion " << WeightChangeStopCriterion << " --- epochs " << epochs << " --- WeightDecay " << WeightDecay << endl;
    
    //create trainer
    mlptrainer trn;
    mlpcreatetrainer(dim, 1, trn);
    if(TASK.NN.print>1) cout << "NN initialized..." << endl;
    
    //attach training set, test set, and training parameters
    mlpsetdataset(trn, xy, NumTrainingSamples);
    mlpsetdecay(trn, WeightDecay);
    mlpsetcond(trn, WeightChangeStopCriterion, epochs);
    if(TASK.NN.print>1) cout << "training set specified..." << endl;

    //create NN with linear output
    multilayerperceptron MLP;
    if(TASK.NN.NumHiddenLayers==0) mlpcreate0(dim, 1, MLP);
    else if(TASK.NN.NumHiddenLayers==1) mlpcreate1(dim, NumNeurons, 1, MLP);
    else if(TASK.NN.NumHiddenLayers==2) mlpcreate2(dim, NumNeurons, NumNeurons, 1, MLP);
    mlpreport rep;
    //modelerrors mdlerrs;

    //train network
    mlptrainnetwork(trn, MLP, NumRestarts, rep);
    if(TASK.NN.print>1) cout << "NN trained..." << endl;
    
    //export network to enable external NNprocess(), can only be done safely in single-thread region
    if(TASK.NN.ReproductionQ){
      TASK.NN.shift = shift;
      TASK.NN.spread = spread;
      TASK.NN.mlp = MLP;
    }
      
    //validate network on (scaled/shifted) test set  
    vector<double> RelDiffVec,yVec,zVec;
    for(int i=0;i<NumTestSamples;i++){
      vector<double> xTest(dim);
      for(int j=0;j<dim;j++) xTest[j] = xyTest[i][j];
      real_1d_array Y; Y.setlength(1);
      mlpprocess(MLP, VecTOreal_1d_array(xTest), Y);
      double y = Y[0];
      RelDiffVec.push_back(RelDiff(y,xyTest[i][dim])); 
      yVec.push_back(y);
      zVec.push_back(xyTest[i][dim]);
    }
    double output = 1.-xiOverlap(yVec,zVec);
    if(TASK.NN.print>1) cout << "NN validated..." << endl;
    
    //export comparison between objective function and (unscaled) network prediction
    if(TASK.NN.print==2){
      cout << "Least-squares error (xi) on test data set = " << output << endl << "export arrays to NN_reference.dat and NN_prediction.dat" << endl;
      int L = 100;
      vector<vector<double>> NNpred(L), NNref(L);
      for(int l=0;l<L;l++){
	NNpred[l].resize(L);
	NNref[l].resize(L);
	vector<double> xTest(dim);
	xTest[0] = -5.+10.*(double)l/(double)L;
	for(int k=0;k<L;k++){
	  xTest[1] = -5.+10.*(double)k/(double)L;
	  real_1d_array Y; Y.setlength(1);
	  mlpprocess(MLP, VecTOreal_1d_array(xTest), Y);
	  //NNpred[l][k] = Y[0];
	  NNpred[l][k] = Y[0]*spread+shift;
	  NNref[l][k] = NNtestFunction(xTest,TASK.NN);
	}
      }
      MatrixToFile(NNref,"NN_reference.dat",16);
      MatrixToFile(NNpred,"NN_prediction.dat",16);
    }
    
    //save and process results
    history.push_back(output);
    double mean = VecAv(history);
    double sigma = sqrt(VecVariance(history));
    if(TASK.NN.print>1 && history.size()>1){ cout << "mean = " << mean << " sigma = " << sigma << endl; }
    if(abs(output-mean)<TASK.NN.TargetSigmaFraction*sigma) targetCounter++;
    if(targetCounter==TASK.NN.TargetCounter) break;
  }
  if(TASK.NN.print>0 && r==MaxNumRepeats && !TASK.NN.ReproductionQ) cout << "ProduceNNinstance: Warning !!! MaxNumRepeats = " << MaxNumRepeats << " reached, but expected network error not determined with target confidence." << endl;
  
  //print best result
  double ExpectedRelativeNetworkError = VecAv(history);
  for(auto i: sort_indices(history)){
    if(TASK.NN.print>0){ cout << endl << "BestPrediction = " << history[i]; if(!TASK.NN.ReproductionQ) cout << " @ NN " << i << " (out of " << r << " networks)"; cout << endl; } 
    break;
  }  
  if(TASK.NN.print>0){
    cout << "ProduceNNinstance completed, expected relative network error = " << ExpectedRelativeNetworkError;
    if(history.size()>1) cout << " [StandardDev = " << sqrt(VecVariance(history)) << "]";
    cout << " from NN parameters = " << vec_to_str_with_precision(par,16) << endl;
    cout << "spread = " << spread << "shift = " << shift << endl;
  }
  if(TASK.NN.ReproductionQ) cout << "NN ready for NNprocess()" << endl;
  
  return ExpectedRelativeNetworkError;
}

void InitRandomNumGeneratorNN(MLPerceptron &NN){
    double MaxInt = (double)(std::numeric_limits<int>::max()), Now = ABS((double)chrono::high_resolution_clock::now().time_since_epoch().count() / 1.0e+10), now = (double)((int)Now), seed = 1.0e+10*(Now-now);
    if(seed<1.0e-6) seed = (double)rand();
    while(seed>=MaxInt) seed /= 2.; while(seed<1.0e+06) seed *= 2.;
    mt19937_64 MTGEN((int)seed);
    uniform_real_distribution<double> RNpos(0.,1.0);
    NN.MTGEN = MTGEN;
    NN.RNpos = RNpos;
}




// H A N D L I N G   R O U T I N E S

int ManualSCloopBreakQ(datastruct &data){//open mpDPFTmanualSCloopBreakQ.dat during runtime, manually change "0" to "1" and save file in order to break SCloop
  ifstream infile;
  infile.open("mpDPFTmanualSCloopBreakQ.dat");
  string line;
  int Q;
  getline(infile, line); istringstream issx(line); issx >> Q;
  infile.close();
  if(Q){
    if(Q==1) PRINT("---------------------------------------SCloop--Manually--Terminated---------------------------------------",data);
    if(Q==2) PRINT("---------------------------------------TASKloop--Manually--Terminated---------------------------------------",data);
    ofstream ManualSCloopBreakQfile;
    ManualSCloopBreakQfile.open("mpDPFTmanualSCloopBreakQ.dat");
    ManualSCloopBreakQfile << 0 << "\n";
    ManualSCloopBreakQfile.close();
    data.Print = 2;
  }
  return Q;
}

void abortQ(datastruct &data){
  if(data.ABORT){
    PRINT("---------------------------------------mpDPFT--Terminated---------------------------------------",data);
    WriteControlFile(data);
    exit(EXIT_FAILURE);
  }
}


// R O U T I N E S    F O R    O U T P U T

void WriteControlFile(datastruct &data){
  ofstream ControlFile;
  string FileName = "mpDPFT_Control.dat";
  ControlFile.open (FileName.c_str());  
  ControlFile << data.controlfile.str();
  ControlFile.close();
}

void StartTimer(string descriptor, datastruct &data){
  if(data.Print==2){ t0 = Time::now(); PRINT("StartTimer for " + descriptor + ":",data); }
}

void EndTimer(string descriptor, datastruct &data){
  if(data.Print==2){ t1 = Time::now(); fsec floatsec = t1 - t0; PRINT("EndTimer for " + descriptor + ": " + to_string(floatsec.count()) + " seconds",data); }
}

void StartTiming(datastruct &data){
  t0 = Time::now();
  PRINT("StartTiming",data);
}

void EndTiming(datastruct &data){
  t1 = Time::now();
  fsec floatsec = t1 - t0;
  PRINT("EndTiming: " + to_string(floatsec.count()) + " seconds",data);
}

void PRINT(string str,datastruct &data){
  if(data.ExitSCcount>0) cout << std::setprecision(16) << str << endl; else cout << std::setprecision(16) << str << "\n";
  data.controlfile << std::setprecision(16) << str << "\n";
  cout.flush();
}

void TASKPRINT(string str,taskstruct &task, int coutQ){
  //cout << "TASKPRINT1" << endl;
  if(coutQ==0) task.controltaskfile << std::setprecision(16) << str << "\n";
  else if(coutQ==1){
    //cout << "TASKPRINT2" << endl;
    cout << std::setprecision(16) << str << endl;
    //cout << "TASKPRINT3" << endl;
    task.controltaskfile << std::setprecision(16) << str << "\n";
    //cout << "TASKPRINT4" << endl;
  }
  else if(coutQ==2){
    cout << std::setprecision(16) << str;
    task.controltaskfile << std::setprecision(16) << str;
  }
  cout.flush();
}

void COUTalongX(string str,vector<double> &Field,datastruct &data){
  for(int i=0;i<data.EdgeLength;i++){
    int index = i;
    if(data.DIM==2) index = i*data.EdgeLength+data.steps/2;
    else if(data.DIM==3) index = i*data.EdgeLength*data.EdgeLength+data.steps/2*data.EdgeLength+data.steps/2;
    cout << str << "[" << i << "] = " << Field[index] << endl;
  }
}

string YYYYMMDD(void){//TimeStamp
  char out[9];
  time_t t=time(NULL);
  strftime(out, sizeof(out), "%Y%m%d", localtime(&t));
  //cout << "Today: " << out << endl;
  string strout = out;
  return strout;
}

// string hhmmss(void){
//   time_t curr_time;
//   curr_time = time(NULL);
//   tm *tm_local = localtime(&curr_time);
//   return to_string(tm_local->tm_hour) + to_string(tm_local->tm_min) + to_string(tm_local->tm_sec);
// }

string hhmmss(void) {
    time_t curr_time = time(NULL);
    tm *tm_local = localtime(&curr_time);  
    ostringstream oss;
    oss << setw(2) << setfill('0') << tm_local->tm_hour
        << setw(2) << setfill('0') << tm_local->tm_min
        << setw(2) << setfill('0') << tm_local->tm_sec;
    return oss.str();
}

void VecToFile(vector<double> &vec, string FileName){
  ofstream outFile;
  outFile.open(FileName);
  outFile << std::setprecision(16);
  for(int i=0;i<vec.size();i++) outFile << vec[i] << "\n";
  outFile.close();
}

void MatrixToFile(vector<vector<double>> &matrix, string FileName, int prec){
  ofstream outFile;
  outFile.open(FileName);
  outFile << std::setprecision(prec);
  for(int i=0;i<matrix.size();i++){
    for(int j=0;j<matrix[i].size();j++){
      outFile << matrix[i][j];
      if(j<matrix[i].size()-1) outFile << " ";
    }
    if(i<matrix.size()-1) outFile << "\n";
  }
  outFile.close();
}

void GetFieldStats(int PrintFieldStatsQ, int OmitWallQ, datastruct &data){
  data.GlobalDenMin = data.Den[0][0], data.GlobalDenMax = data.Den[0][0];
  for(int s=0;s<data.S;s++){
    double test, denmin = data.Den[s][data.CentreIndex], denmax = denmin, envmin = data.Env[s][data.CentreIndex], envmax = envmin, Vmin = data.V[s][data.CentreIndex], Vmax = Vmin;
    for(int i=0;i<data.GridSize;i++){
      test = data.Den[s][i];
      if( test<denmin && (!data.FLAGS.KD || ( data.FLAGS.KD && (data.KD.ExportCleanDensity<MP || ABS(test)<data.KD.ExportCleanDensity) ) ) ) denmin = test;
      else if( test>denmax && (!data.FLAGS.KD || ( data.FLAGS.KD && (data.KD.ExportCleanDensity<MP || ABS(test)<data.KD.ExportCleanDensity) ) ) ) denmax = test;
      test = data.Env[s][i];
      if(test<envmin) envmin = test;
      else if(data.Environments[s]==0 || ( (!OmitWallQ && test>envmax) || (OmitWallQ && test>envmax && WithinWallsQ(data.VecAt[i],data)) ) ) envmax = test;
      test = data.V[s][i];
      if(test<Vmin) Vmin = test;
      else if( (!OmitWallQ && test>Vmax) || (OmitWallQ && test>Vmax && WithinWallsQ(data.VecAt[i],data)) ) Vmax = test;
    }
    data.DenMin[s] = denmin; data.DenMax[s] = denmax;
    data.EnvMin[s] = envmin; data.EnvMax[s] = envmax;
    data.VMin[s] = Vmin; data.VMax[s] = Vmax; 
    if(denmin<data.GlobalDenMin) data.GlobalDenMin = denmin;
    if(denmax>data.GlobalDenMax) data.GlobalDenMax = denmax;
  }
  if(PrintFieldStatsQ){
    data.txtout << "**** FieldStats ****\\\\"; cout << "**** FieldStats ****" << endl;
    data.txtout << "DenMin = " << vec_to_str(data.DenMin) << "\\\\"; cout << "DenMin = " << vec_to_str(data.DenMin) << endl;
    data.txtout << "DenMax = " << vec_to_str(data.DenMax) << "\\\\"; cout << "DenMax = " << vec_to_str(data.DenMax) << endl;
    data.txtout << "EnvMin = " << vec_to_str(data.EnvMin) << "\\\\"; cout << "EnvMin = " << vec_to_str(data.EnvMin) << endl;
    data.txtout << "EnvMax = " << vec_to_str(data.EnvMax) << "\\\\"; cout << "EnvMax = " << vec_to_str(data.EnvMax) << endl;
    data.txtout << "VMin = " << vec_to_str(data.VMin) << "\\\\"; cout << "VMin = " << vec_to_str(data.VMin) << endl;
    data.txtout << "VMax = " << vec_to_str(data.VMax) << "\\\\"; cout << "VMax = " << vec_to_str(data.VMax) << endl;
  }
  if(data.GlobalDenMax>data.GlobalTimeDenMax) data.GlobalTimeDenMax = data.GlobalDenMax;
  for(int k=0;k<data.K;k++){
    double test, resmin = data.NonUniformResourceDensities[k][data.CentreIndex], resmax = resmin;
    if(data.Resources[k]<-1. && data.Resources[k]>-2.){ resmin = data.DenMin[k]; resmax = data.DenMax[k]; }
    else{
      for(int i=0;i<data.GridSize;i++){
	test = data.NonUniformResourceDensities[k][i];
	if(test<resmin) resmin = test;
	else if(test>resmax) resmax = test;
      }
    }
    data.ResMin[k] = resmin;
    data.ResMax[k] = resmax;
  }
}

bool WithinWallsQ(vector<double> vec, datastruct &data){
  if(ABS(data.Wall)<MP) return true;
  double r = Norm(vec);
  if(data.Wall>MP && r<data.rW) return true;
  bool check = true; for(int i=0;i<vec.size();i++) if(ABS(vec[i])>data.rW) check = false;
  if(data.Wall<-MP && check) return true;
  return false;
}

void GetCutPositions(datastruct &data){
  data.CutPositions.resize(data.EdgeLength);
  if(data.DIM==1){
    for(int i=0;i<data.EdgeLength;i++){
      data.CutPositions[i].resize(2);
      data.CutPositions[i][0] = data.Lattice[i];
      data.CutPositions[i][1] = 0.;
    }
  }
  else{
    vector<double> CutStart = {{data.OutCut[0]},{data.OutCut[1]}};
    vector<double> CutEnd = {{data.OutCut[2]},{data.OutCut[3]}};
    double DeltaX = (CutEnd[0]-CutStart[0])/((double)(data.steps));
    double DeltaY = (CutEnd[1]-CutStart[1])/((double)(data.steps));
    for(int i=0;i<data.EdgeLength;i++){
      data.CutPositions[i].resize(2); // now, data.CutPositions is a vector of length steps, with each component holding a position (X,Y) along the Cut
      data.CutPositions[i][0] = CutStart[0]+(double)i*DeltaX;
      data.CutPositions[i][1] = CutStart[1]+(double)i*DeltaY;
      //cout << data.CutPositions[i][0] << " " << data.CutPositions[i][1] << endl;
    }
  }
  data.CutLine = Norm(VecDiff(data.CutPositions[0],data.CutPositions[data.CutPositions.size()-1]));
}
  
void GetCutDataStream(vector<vector<vector<double>>> &FI, datastruct &data){
	data.CutDataStream.str("");
  	data.Global3DcolumnDenMin = 0.; data.Global3DcolumnDenMax = 0.;
    data.ScaledDenMin = 0.; data.ScaledDenMax = 0.;
  	for(int i=0;i<data.EdgeLength;i++){
    	for(int s=0;s<data.S;s++){
      		for(int p=0;p<3;p++){
              	if( !data.FLAGS.KD || ( data.FLAGS.KD && (data.KD.ExportCleanDensity<MP || ABS(FI[s][p][i])<data.KD.ExportCleanDensity) ) ){
                  	double x = -data.CutLine/2.+(double)i*data.CutLine/((double)data.steps);
              		if(s==0 && p==0) data.CutDataStream << x;//column 1
					data.CutDataStream << " " << FI[s][p][i];//columns (s+1)*3-1->(s+1)*3+1 [for s=0,...,S-1]
					if(p==0){
                      	double ScaleFactor = POW(x,data.DIM-1);
                      	if(ScaleFactor*FI[s][p][i]<data.ScaledDenMin) data.ScaledDenMin = ScaleFactor*FI[s][p][i];
                        else if(ScaleFactor*FI[s][p][i]>data.ScaledDenMax) data.ScaledDenMax = ScaleFactor*FI[s][p][i];
                    }
                }
      		}
        }
    	if(data.DIM==3){
      		data.CutDataStream << " " << data.Lattice[i];//column (data.S)*3+2
      		for(int s=0;s<data.S;s++){
				int j = data.steps/2;
				double ColumnDensity = 0.;
				for(int k=0;k<data.EdgeLength;k++){
	  				int index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+k;
	  				ColumnDensity += data.Den[s][index];
				}
				ColumnDensity *= data.Deltax;
                //cout << ColumnDensity << " " << data.Global3DcolumnDenMax; if(ColumnDensity>data.Global3DcolumnDenMax) cout << " True"; cout << endl;
				if(ColumnDensity>data.Global3DcolumnDenMax) data.Global3DcolumnDenMax = ColumnDensity;
				else if(ColumnDensity<data.Global3DcolumnDenMin) data.Global3DcolumnDenMin = ColumnDensity;
				//cout << data.Global3DcolumnDenMax << endl;
				data.CutDataStream << " " << ColumnDensity;//columns (data.S)*3+3+s [for s=0,...,S-1]
      		}
    	}
    	data.CutDataStream << "\n";
  	}
}



vector<vector<vector<double>>> GetFieldInterpolationsAlongCut(datastruct &data){
  vector<vector<vector<double>>> fi(data.S);
  for(int s=0;s<data.S;s++){
    fi[s].resize(3); data.minmaxfi[s].resize(3);
    for(int p=0;p<3;p++) fi[s][p].resize(data.EdgeLength);
    for(int p=0;p<3;p++){ data.minmaxfi[s][p].resize(2); }
  }

  if(data.DIM>1) GetFieldInterpolations(data);
  GetCutPositions(data);
  
  for(int s=0;s<data.S;s++){
    if(data.DIM==1){
      for(int i=0;i<data.EdgeLength;i++){
	fi[s][0][i] = data.Den[s][i]; //if(data.Den[s][i]<0.) cout << "GetFieldInterpolationsAlongCut: " << fi[s][0][i] << endl;
	fi[s][1][i] = data.Env[s][i];
	fi[s][2][i] = data.V[s][i];
      }
    }
    else{
      for(int p=0;p<3;p++){
	//fi[s][p][0] = spline2dcalc(data.FieldInterpolations[s][p],data.CutPositions[0][0],data.CutPositions[0][1]);
	//unclear why x and y have to be swapped...:
	fi[s][p][0] = spline2dcalc(data.FieldInterpolations[s][p],data.CutPositions[0][1],data.CutPositions[0][0]);
	data.minmaxfi[s][p][0] = fi[s][p][data.steps/2];
	data.minmaxfi[s][p][1] = fi[s][p][data.steps/2];
	for(int i=1;i<data.EdgeLength;i++){
	  //fi[s][p][i] = spline2dcalc(data.FieldInterpolations[s][p],data.CutPositions[i][0],data.CutPositions[i][1]);
	  //unclear why x and y have to be swapped...:
	  fi[s][p][i] = spline2dcalc(data.FieldInterpolations[s][p],data.CutPositions[i][1],data.CutPositions[i][0]);
	  if(fi[s][p][i] < data.minmaxfi[s][p][0]) data.minmaxfi[s][p][0] = fi[s][p][i];
	  if(fi[s][p][i] > data.minmaxfi[s][p][1] && ( ABS(data.Wall)<MP || (ABS(data.Wall)>MP && WithinWallsQ(data.CutPositions[i],data)) ) ) data.minmaxfi[s][p][1] = fi[s][p][i];
	}
      }
    }
  }
  
  //for(int i=0;i<data.EdgeLength;i++) cout << 0 << " fi " << i << " " << fi[0][1][i*data.EdgeLength+(data.EdgeLength-1)/2] << endl;
  
  return fi;
}

void Regularize(vector<int> procedures, vector<double> &Field, double auxVal, datastruct &data){//manually modify field
  for(int p=0;p<procedures.size();p++){
    if(procedures[p]==0){ }//do nothing
    else if(procedures[p]==1){//superimpose Gaussian:
      double a = -30./(data.edge*data.edge);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 	
      for(int i=0;i<data.GridSize;i++){
	Field[i] *= EXP(a*Norm2(data.VecAt[i]));
      }
    }
    else if(procedures[p]==2){//smooth (radial) cutoff of boundary regions (---> 1.0e-17 at edge/2)
      double threshold = auxVal*data.edge, threshold2 = threshold*threshold, a = -2500./(data.edge*data.edge);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++){
		double r = Norm(data.VecAt[i]);
		if(r>threshold) Field[i] *= EXP(a*(r-threshold)*(r-threshold));
      }
    }
    else if(procedures[p]==3){//Fourier filter
      FourierFilter(Field, data);
    }
    else if(procedures[p]==4){//POS
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 	
      for(int i=0;i<data.GridSize;i++){
	Field[i] = POS(Field[i]);
      }
    }
    else if(procedures[p]==5){//POS and renormalize
      double integral = Integrate(data.ompThreads,data.method, data.DIM, Field, data.frame);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1) 	
      for(int i=0;i<data.GridSize;i++){
	Field[i] = POS(Field[i]);
      }
      double integralPOS = Integrate(data.ompThreads,data.method, data.DIM, Field, data.frame);
      if(integralPOS>0.) MultiplyField(data,Field, data.DIM, data.EdgeLength, integral/integralPOS);
    }
    else if(procedures[p]==6){//add radial Wall
      double min = Field[0], max = min; for(int i=1;i<Field.size();i++){ double test = Field[i]; if(test>max) max = test; else if(test<min) min = test; }
      double WallHeight; if(max-min>MP) WallHeight = auxVal*(max-min); else WallHeight = auxVal; if(data.PrintSC==1) PRINT("WallHeight = " + to_string(WallHeight),data);
      double threshold = data.RegularizationThreshold*data.edge, threshold2 = threshold*threshold, a = -2500./(data.edge*data.edge);
      #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
      for(int i=0;i<data.GridSize;i++){
	double r = Norm(data.VecAt[i]);
	if(r>threshold) Field[i] += WallHeight*(1.-EXP(a*(r-threshold)*(r-threshold)));
      }
    }    
  }
}

void GetConsumedResources(datastruct &data){
  for(int k=0;k<data.K;k++){
    data.ConsumedResources[k] = 0.;
    for(int s=0;s<data.S;s++){
      data.ConsumedResources[k] += data.Consumption[k][s]*data.TmpAbundances[s];
    }
  }
}

void StoreMovieData(datastruct &data){
  StartTimer("StoreMovieData",data);
  vector<vector<vector<double>>> FI = GetFieldInterpolationsAlongCut(data); //[s species][p plots][i points on line]
  for(int i=0;i<data.EdgeLength;i++){
    for(int s=0;s<data.S;s++){
      for(int p=0;p<3;p++){
	data.MovieStorage[i].push_back(FI[s][p][i]);
      }
    }
  }
  data.FrameCount++;
  EndTimer("StoreMovieData",data);
}

void StoreSnapshotData(datastruct &data){
	StartTimer("StoreSnapshotData",data);
	ofstream outFile;
	outFile.open("mpDPFT_DynDFTe_Snapshot_Den_" + to_string(data.SCcount) + ".dat");
	outFile << std::setprecision(8);
	for(int i=0;i<data.EdgeLength;i++){
		for(int j=0;j<data.EdgeLength;j++){//for gnuplot pm3d
			int index = i*data.EdgeLength+j;
			outFile << data.VecAt[index][0] << " " << data.VecAt[index][1] << " " << data.Den[0][index] << "\n";
		}
		outFile << "\n";
	}
	outFile.close();
	outFile.open("mpDPFT_DynDFTe_Snapshot_Vec_" + to_string(data.SCcount) + ".dat");
	outFile << std::setprecision(8);
	int Ig = 0, Iv = 0;
	for(int i=0;i<data.GridSize;i++){
			outFile << data.VecAt[i][0] << " " << data.VecAt[i][1] << " " << data.Den[0][i] << " " << TASK.DynDFTe.g[0][i][0] << " " << TASK.DynDFTe.g[0][i][1] << " " << TASK.DynDFTe.v[0][i][0] << " " << TASK.DynDFTe.v[0][i][1] << "\n";
			if(Norm2(TASK.DynDFTe.g[0][i])>Norm2(TASK.DynDFTe.g[0][Ig])){ Ig = i; }
			if(Norm2(TASK.DynDFTe.v[0][i])>Norm2(TASK.DynDFTe.v[0][Iv])){ Iv = i; /*cout << "... ... " << vec_to_str(TASK.DynDFTe.v[0][Iv]) << endl;*/ }
			//if(data.SCcount==1) cout << "StoreSnapshotData: " << i << " " << vec_to_str(TASK.DynDFTe.v[0][i]) << endl;
			//if(data.SCcount==5) cout << "StoreSnapshotData: " << vec_to_str(TASK.DynDFTe.v[0][Iv]) << endl;
	}
	double gScale = 0., vScale = 0.;
	if(Norm(TASK.DynDFTe.g[0][Ig])>MP) gScale = 0.15/Norm(TASK.DynDFTe.g[0][Ig]);
	if(Norm(TASK.DynDFTe.v[0][Iv])>MP) vScale = 0.15/Norm(TASK.DynDFTe.v[0][Iv]);
	TASK.DynDFTe.VectorScale.push_back({{gScale,vScale}});
	//TASK.DynDFTe.VectorScale.push_back({{1.,1.}});
	cout << "StoreSnapshotData: " << Ig << " " << Iv << " " << vec_to_str(TASK.DynDFTe.VectorScale[TASK.DynDFTe.VectorScale.size()-1]);
	outFile.close();	
	EndTimer("StoreSnapshotData",data);
}

void SetStreams(datastruct &data){
	StartTimer("SetStreams",data);
	
  data.epslatexSTART << "\n" << "#!/bin/bash" << "\n" << "GNUPLOT=gnuplot" << "\n" << "OUTPUT=`echo $0 | sed 's/\\.sh/";
  data.square << "/'`" << "\n" << "$GNUPLOT << EOF" << "\n" << "set terminal epslatex dashed size 10,10";
  data.rectangle << "/'`" << "\n" << "$GNUPLOT << EOF" << "\n" << "set terminal epslatex dashed size 10,6.875";
  data.epslatexLineStyles << "\n" << "set output \"tmp_split.tex\"" << "\n" << "xunitsize=0.30" << "\n" << "yunitsize=0.30" << "\n" << "set lmargin 0" << "\n" << "set rmargin 0" << "\n" << "set tmargin 0" << "\n" << "set bmargin 0" << "\n" << "set size xunitsize,yunitsize" << "\n" << "set origin 0,0" << "\n" << "set style line 1 dt 1 lw 6 lc rgb \"#0000ff\"" << "\n" << "set style line 2 dt (5,2) lw 6 lc rgb \"#ffa500\"" << "\n" << "set style line 3 dt (1,1) lw 6 lc rgb \"#00ff00\"" << "\n" << "set style line 4 dt (5,2) lw 6 lc rgb \"#a9a9a9\"" << "\n" << "set style line 5 dt 1 lw 6 lc rgb \"#000000\"" << "\n" << "set style line 6 dt 1 lw 6 lc rgb \"#00ff00\"" << "\n" << "set style line 7 dt 1 lw 2 lc rgb \"#ff0000\"" << "\n" << "set style line 11 dt 1 lw 2 lc rgb \"#ff0000\"" << "\n" << "set style line 12 dt (1,2) lw 2 lc rgb \"#0000ff\"" << "\n" << "set style line 13 dt (4,1) lw 6 lc rgb \"#0000ff\"" << "\n" << "set style line 14 dt (1,2) lw 2 lc rgb \"#00ff00\"" << "\n" << "set style line 15 dt (4,1) lw 6 lc rgb \"#00ff00\"" << "\n" << "set style line 16 dt (4,1) lw 6 lc rgb \"#ffffff\"" << "\n";
  data.epslatexEND << "EOF" << "\n" << "cat tmp_split.eps \\" << "\n" << "  | sed 's/\\(\\/LT2.\\+\\[\\).\\+\\(\\] LC2.\\+def\\)/\\1 2 dl1 3 dl2 \\2/' \\" << "\n" << "  | sed 's/\\(\\/LT1.\\+\\[\\).\\+\\(\\] LC1.\\+def\\)/\\1 6 dl1 6 dl2 \\2/' \\" << "\n" << "  > tmp.eps && mv tmp.eps tmp_split.eps" << "\n" << "cat tmp_split.tex \\" << "\n" << "  | sed 's/\\$\\(1\\)e+00\\$/\\$1\\$/g' \\" << "\n" << "  | sed 's/\\$\\(.\\+\\)e+\\(.\\+\\)\\$/\\$\\1 \\\\cdot 10\\^{\\2}\\$/g' \\" << "\n" << "  | sed 's/\\$\\(1\\)e-0*\\(.\\+\\)\\$/\\$10\\^{-\\2}\\$/g' \\" << "\n" << "  | sed 's/\\$\\(.\\+\\)e-0*\\(.\\+\\)\\$/\\$\\1 \\\\cdot 10\\^{-\\2}\\$/g' \\" << "\n" << "  | sed 's/10\\^{0/10\\^{/g' \\" << "\n" << "  > tmp.tex && mv tmp.tex tmp_split.tex &&\\" << "\n" << "epslatex2epspdf tmp_split $OUTPUT &&\\" << "\n" << "rm -f tmp_split.tex tmp_split.eps &&\\" << "\n" << "echo \"Generated $OUTPUT.eps and $OUTPUT.pdf\" && echo" << "\n";
  data.TeX1 << "\\documentclass[a4paper,10pt]{article}" << "\n" << "\\usepackage[utf8]{inputenc}" << "\n" << "\\usepackage{graphicx}" << "\n" << "\\usepackage{placeins}" << "\n" << "\\textwidth=18cm" << "\n" << "\\textheight=28cm" << "\n" << "\\topmargin=-2cm" << "\n" << "\\oddsidemargin=-1cm" << "\n" << "\\setlength\\parindent{0pt}" << "\n" << "\\pdfminorversion=7" << "\n" << "\\begin{document}" << "\n";
  data.TeX2 << "\n" << "\\end{document}";
  
  data.IncludeGraphicsSTART << "\n" << "\\begin{figure}[htb!]";
  data.includeGraphics << "\n" << "\\includegraphics[width=0.49\\linewidth]{mpDPFT_Plots";
  data.include3Graphics << "\n" << "\\includegraphics[width=0.29\\linewidth]{mpDPFT_Plots";
  data.IncludeGraphicsEND << "\n" << "\\end{figure}";  
  
	data.EnergyNames.clear(); data.EnergyNames.resize(0);
	data.IntermediateEnergies << "#SCcount ";
	for(int s=0;s<data.S;s++){
		data.EnergyNames.push_back("Ekin[" + to_string(s) + "]");
		data.IntermediateEnergies << " " << data.EnergyNames[data.EnergyNames.size()-1];
		data.EnergyNames.push_back("Eenv" + to_string(data.Environments[0]) + "[" + to_string(s) + "]");
		data.IntermediateEnergies << " " << data.EnergyNames[data.EnergyNames.size()-1];
	}
	for(int i=0;i<data.Interactions.size();i++){
		data.EnergyNames.push_back("Eint" + to_string(data.Interactions[i]));
		data.IntermediateEnergies << " " << data.EnergyNames[data.EnergyNames.size()-1];
	}
	data.EnergyNames.push_back("Etot");
	data.IntermediateEnergies << " " << data.EnergyNames[data.EnergyNames.size()-1] << "\n";
	
	EndTimer("SetStreams",data);
}


void ExportData(datastruct &data){
  StartTimer("ExportData",data);
  //PRINT(" ***** get energies ****",data);
  PRINT("FreeRAM1 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
  GetEnergy(data);
  PRINT("FreeRAM3 = " + to_string_with_precision((double)getFreeRAM(data),5),data);

  freeContainerMemory(data.Dx);
  freeContainerMemory(data.Dy);
  freeContainerMemory(data.Dz);
  freeContainerMemory(data.D2x);
  freeContainerMemory(data.D2y);
  freeContainerMemory(data.D2z);
  freeContainerMemory(data.SqrtDenDx);
  freeContainerMemory(data.SqrtDenDy);
  freeContainerMemory(data.SqrtDenDz);
  freeContainerMemory(data.GradSquared);
  freeContainerMemory(data.Laplacian);
  freeContainerMemory(data.SqrtDenGradSquared);
  PRINT("FreeRAM4 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
  
  if(data.FLAGS.Export){
    PRINT(" ***** prepare export ****",data);
    time(&data.Timing2); data.Timing2 = difftime(data.Timing2, data.Timing0);

    if(data.System==1 || data.System==2 || data.FLAGS.ForestGeo){
      //data.fitness = {{1.}};
      //data.fitness = {{1.1},{1.}};
      //data.fitness = {{1.},{2.},{4.}};          
      data.txtout << "zeta " << data.zeta << "\\\\"; //global factor for resource terms, Interaction 2
      data.txtout << "sigma " << data.sigma << "\\\\";
      data.txtout << "kappa " << data.kappa << "\\\\";
      data.txtout << "Consumption:" << "\\\\"; for(int k=0;k<data.K;k++) data.txtout << vec_to_str(data.Consumption[k]) << "\\\\";
      data.txtout << "Amensalism:" << "\\\\"; for(int s=0;s<data.S;s++){ for(int sp=0;sp<data.S;sp++) data.txtout << data.Amensalism[s*data.S+sp] << " "; data.txtout << "\\\\"; }
      data.txtout << "RepMut:" << "\\\\"; for(int s=0;s<data.S;s++){ for(int sp=0;sp<data.S;sp++) data.txtout << data.RepMut[s*data.S+sp] << " "; data.txtout << "\\\\"; }
      data.txtout << "Competition:" << "\\\\"; for(int s=0;s<data.S;s++){ for(int sp=0;sp<data.S;sp++) data.txtout << data.Competition[s*data.S+sp] << " "; data.txtout << "\\\\"; }
      //data.txtout << "Consumption;   Consumed/Provided:" << "\\\\";
      //for(int k=0;k<data.K;k++) data.txtout << vec_to_str(data.Consumption[k]) << ";   " << VecMult(data.Consumption[k],data.Abundances)) << "/" << data.AvailableResources[k] << "\\\\";    
      data.txtout << "\\newpage \n";
    }

    PRINT(" ***** get field stats ****",data);
    data.txtout << "\\newpage \n";
    GetFieldStats(1,0,data);

    if(data.DIM>1){
      PRINT(" ***** get FieldInterpolations ****",data);
      GetFieldInterpolations(data);
    }
    PRINT(" ***** get CutInterpolations ****",data);
    vector<vector<vector<double>>> FI = GetFieldInterpolationsAlongCut(data); //[s species][p plots][i points on line]  
  
    //manually modify densities
    //for(int s=0;s<data.S;s++) Regularize({{1}},data.Den[s],data.RegularizationThreshold,data);

    PRINT(" ***** get consumption ****",data);
    GetConsumedResources(data);

    PRINT("FreeRAM5 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
    if(data.FLAGS.ExportCubeFile){
    	PRINT(" ***** export Cube file ****",data);
    	ofstream DenDataCube;//S-column row-major file of densities;
        double c = 1., c3 = 1., VolumeElement = POW(data.Deltax/c,3);
        if(data.Units==2){//but, actually, switch from Angstrom to Bohr
    		c = 0.52917721067;
    		VolumeElement = POW(data.Deltax/c,3);
        	c3 = c*c*c;
        }
    	DenDataCube.open("mpDPFT_Den_Cube.dat");
    	DenDataCube << std::setprecision(data.OutPrecision);
    	DenDataCube << data.NumberOfNuclei << " " << 0 << " " << 0 << " " << 0 << "\n";
    	DenDataCube << data.steps << " " << data.Deltax/c << " " << 0 << " " << 0 << "\n";
        DenDataCube << data.steps << " " << 0 << " " << data.Deltax/c << " " << 0 << "\n";
        DenDataCube << data.steps << " " << 0 << " " << 0 << " " << data.Deltax/c << " " << "\n";
        if(data.NumberOfNuclei>0){
        	for(int nuc=0;nuc<data.NumberOfNuclei;nuc++){
          		vector<double> NucPosition = {data.NucleiPositions[0][nuc]/c,data.NucleiPositions[1][nuc]/c,data.NucleiPositions[2][nuc]/c};
          		DenDataCube << data.NucleiTypes[nuc] << " " << vec_to_str_with_precision(NucPosition,16) << "\n";
        	}
        }
        else DenDataCube << 0 << " " << 0 << " " << 0 << " " << 0 << "\n";
    	for(int i=0;i<data.steps;i++){//omit the last grid point in each direction
          	for(int j=0;j<data.steps;j++){
              	for(int k=0;k<data.steps;k++){
                  	int index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+k;
          			DenDataCube << max(1.0e-14,c3*data.Den[0][index]*VolumeElement) << "\n";
                }
            }
        }
        PRINT("FreeRAM6 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
    	DenDataCube.close();
    }
    else{
      	PRINT(" ***** export Den ****",data);
      	ofstream DenData;//S-column row-major file of densities;
      	double conversion = 1.;
      	//double c = 0.52917721067, conversion = c*c*c;//to switch from Angstrom to Bohr:
      	DenData.open("mpDPFT_Den.dat");
      	DenData << std::setprecision(data.OutPrecision);
      	for(int i=0;i<data.GridSize;i++){ for(int s=0;s<data.S;s++){ if(s==data.S-1) DenData << conversion*data.Den[s][i] << "\n"; else DenData << conversion*data.Den[s][i] << " "; } }
      	DenData.close();
    }

    PRINT("FreeRAM7 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
  
  	if(data.DIM==3 && data.steps>400){
      	PRINT(" ***** skip V ****",data);
    }
    else{
    	PRINT(" ***** export V ****",data);
    	ofstream VData;//to continue calculation via InterpolVQ
    	VData.open("mpDPFT_V.dat");
    	VData << std::setprecision(data.OutPrecision);
    	vector<double> position(data.DIM);
    	for(int i=0;i<data.GridSize;i++){
      		position = data.VecAt[i];
      		for(int j=0;j<data.DIM;j++) VData << position[j] << " ";
      		for(int s=0;s<data.S;s++) VData << data.V[s][i] << " ";
      		VData << "\n";
        }
    	VData.close();
    }

    PRINT("FreeRAM8 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
  
    vector<double> pos(data.DIM);
    double posxinit;
    if(data.DIM>1){
      PRINT(" ***** export ContourData ****",data);
      //cout << "data.Env[0][data.CentreIndex] = " << data.Env[0][data.CentreIndex] << endl;
      int index;
      ofstream ContourData;//for 2D contour plots with gnuplot
      ContourData.open("mpDPFT_ContourData.dat");
      ContourData << std::setprecision(data.OutPrecision);
      VecAtIndex(0,data,pos); posxinit = pos[0];
      for(int i=0;i<data.EdgeLength;i++){
	for(int j=0;j<data.EdgeLength;j++){
	  if(data.DIM==2) index = i*data.EdgeLength+j;
	  else if(data.DIM==3) index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+data.steps/2;//x-y-plane    
	  pos = data.VecAt[index];
	  if(pos[0]>posxinit){//to provide correct data file structure for gnuplot pm3d
	    ContourData << "\n";
	    posxinit = pos[0];
	  }
	  ContourData << pos[0] << " "  << pos[1] << " " << Norm(pos);//columns 1 2 3
	  for(int s=0;s<data.S;s++) ContourData  << " " << data.Den[s][index] << " " << data.Env[s][index] << " " << data.V[s][index]; //columns (s+1)*3+1->(s+1)*3+3 [for s=0,...,S-1]
	  ContourData << "\n";
	}
      }	
      ContourData << "\n";
      ContourData.close();

      freeContainerMemory(data.V);
      
      if(data.K>0){

      PRINT(" ***** export ResourceData ****",data);  
      ofstream ResourceData;//for 2D contour plots with gnuplot
      ResourceData.open("mpDPFT_ResourceData.dat");
      ResourceData << std::setprecision(data.OutPrecision);
      VecAtIndex(0,data,pos); posxinit = pos[0];
      for(int i=0;i<data.EdgeLength;i++){
	for(int j=0;j<data.EdgeLength;j++){
	  if(data.DIM==2) index = i*data.EdgeLength+j;
	  else if(data.DIM==3) index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+data.steps/2;//x-y-plane    
	  pos = data.VecAt[index];
	  if(pos[0]>posxinit){//to provide correct data file structure for gnuplot pm3d
	    ResourceData << "\n";
	    posxinit = pos[0];
	  }
	  ResourceData << pos[0] << " "  << pos[1] << " " << Norm(pos);//columns 1 2 3
	  for(int k=0;k<data.K;k++){//columns 4->4+K-1
	    if(data.Resources[k]<-1. && data.Resources[k]>-2.) ResourceData  << " " << data.Den[k][index];
	    else ResourceData  << " " << data.NonUniformResourceDensities[k][index];
	  }
	  ResourceData << "\n";
	}	
      }
      ResourceData << "\n";
      ResourceData.close();

      PRINT(" ***** export LimitingResourceData ****",data);  
      ofstream LimitingResourceData;//for 2D contour plots with gnuplot
      LimitingResourceData.open("mpDPFT_LimitingResourceData.dat");
      LimitingResourceData << std::setprecision(data.OutPrecision);
      VecAtIndex(0,data,pos); posxinit = pos[0];
      for(int i=0;i<data.EdgeLength;i++){
	for(int j=0;j<data.EdgeLength;j++){
	  if(data.DIM==2) index = i*data.EdgeLength+j;
	  else if(data.DIM==3) index = i*data.EdgeLength*data.EdgeLength+j*data.EdgeLength+data.steps/2;//x-y-plane    
	  pos = data.VecAt[index];
	  if(pos[0]>posxinit){//to provide correct data file structure for /*gnuplot*/ pm3d
	    LimitingResourceData << "\n";
	    posxinit = pos[0];
	  }
	  LimitingResourceData << pos[0] << " "  << pos[1] << " " << Norm(pos);//columns 1 2 3
	  for(int s=0;s<data.S;s++) LimitingResourceData  << " " << data.LimitingResource[s][index]; //columns 4->4+K-1
	  LimitingResourceData << "\n";
	}	
      }
      LimitingResourceData << "\n";
      LimitingResourceData.close();    
      
      }
    }

    PRINT("FreeRAM9 = " + to_string_with_precision((double)getFreeRAM(data),5),data);

    if(data.DIM==2 && data.FLAGS.RefData){
      PRINT(" ***** export ReferenceData ****",data);  
      ofstream ReferenceData;//for 2D contour plots with gnuplot
      ReferenceData.open("mpDPFT_ReferenceData.dat");
      ReferenceData << std::setprecision(data.OutPrecision);
      VecAtIndex(0,data,pos); posxinit = pos[0];
      for(int i=0;i<data.GridSize;i++){
	pos = data.VecAt[i];
	if(pos[0]>posxinit){//to provide correct data file structure for gnuplot pm3d
	  ReferenceData << "\n";
	  posxinit = pos[0];
	}
	ReferenceData << pos[0] << " "  << pos[1] << " " << Norm(pos) << " " << data.ReferenceData[i] << "\n";//columns 1 2 3 4
      }	
      ReferenceData << "\n";
      ReferenceData.close();
    }
    
    if(data.FLAGS.ForestGeo){
      PRINT(" ***** export ReferenceDensities ****",data);  
      ofstream ReferenceDensities, FOMReferenceDensities;//for 2D contour plots with gnuplot
      ReferenceDensities.open("mpDPFT_ReferenceData_ForestGeo.dat");
      FOMReferenceDensities.open("mpDPFT_FOMReferenceData_ForestGeo.dat");
      ReferenceDensities << std::setprecision(data.OutPrecision);
      FOMReferenceDensities << std::setprecision(data.OutPrecision);
      for(int i=0;i<data.EdgeLength;i++){
	if(i>0){
	  ReferenceDensities << "" << "\n" << "";
	  FOMReferenceDensities << "" << "\n" <<"";
	}
	for(int j=0;j<data.EdgeLength;j++){
	  int ij = i*data.EdgeLength+j;
	  ReferenceDensities << TASK.refData[ij][0];
	  FOMReferenceDensities << TASK.FOMrefData[ij][0];
	  for(int k=1;k<TASK.refData[0].size();k++) ReferenceDensities << " " << TASK.refData[ij][k];
	  for(int k=1;k<TASK.refData[0].size();k++) FOMReferenceDensities << " " << TASK.FOMrefData[ij][k];
	  ReferenceDensities << "\n";
	  FOMReferenceDensities << "\n";
	}
      }
      ReferenceDensities.close();
      FOMReferenceDensities.close();
    }    

    PRINT(" ***** export CutData ****",data);  
    ofstream CutData;//for cuts defined by data.OutCut
    CutData.open("mpDPFT_CutData.dat");
    CutData << std::setprecision(data.OutPrecision);
    if(Norm(data.CutPositions[(data.CutPositions.size()-1)/2])>MP){ data.warningCount++; PRINT("ExportData: Warning!!! Cut not through origin... output may misrepresent data.",data); }
    GetCutDataStream(FI,data);
    CutData << data.CutDataStream.str();
    CutData.close();
 
    if(data.MovieQ>0 && data.SCcount>0){
      PRINT(" ***** export MovieData ****",data); 
      ofstream MovieData;//for cuts defined by data.OutCut, maximum 6 densities, SCcount<10000
      MovieData.open("mpDPFT_MovieData.tmp");
      MovieData << std::setprecision(4);
      int J = data.MovieStorage[0].size();
      for(int i=0;i<data.EdgeLength;i++){
	for(int j=0;j<J;j++){
	  MovieData << " " << data.MovieStorage[i][j];
	}
	if(i<data.EdgeLength-1) MovieData << "\n";
      }
      MovieData.close();
  
      ofstream movieshell; movieshell.open("mpDPFT_Movie.sh");
  
      ostringstream moviebody;
  
      double deltaDen = data.GlobalDenMax-data.GlobalDenMin; vector<int> column(data.S); for(int s=0;s<data.S;s++) column[s] = 2+s*3;
      moviebody << "#!/bin/bash" << "\n" << "GNUPLOT=gnuplot" << "\n" << "$GNUPLOT << EOF" << "\n" << "set term png size 800,600" << "\n" << "set style line 1 dt 1 lw 4 lc rgb \"#0000ff\"" << "\n" << "set style line 2 dt 1 lw 4 lc rgb \"#ffa500\"" << "\n" << "set style line 3 dt 1 lw 4 lc rgb \"#00ff00\"" << "\n" << "set style line 4 dt 1 lw 4 lc rgb \"#a9a9a9\"" << "\n" << "set style line 5 dt 1 lw 4 lc rgb \"#000000\"" << "\n" << "set style line 6 dt 1 lw 4 lc rgb \"#ff0000\"" << "\n" << "set style line 10 dt 1 lw 1 lc rgb \"#000000\"" << "\n";
      
      if(data.MovieQ==1) moviebody << "set yrange [" << data.GlobalDenMin-0.1*deltaDen << ":" << data.GlobalDenMax+0.1*deltaDen << "]" << "\n";
      else if(data.MovieQ==2) moviebody << "set logscale y" << "\n" << "set yrange [" << max(MachinePrecision,data.GlobalDenMin-0.1*deltaDen) << ":" << data.GlobalDenMax+0.1*deltaDen << "]" << "\n";
      moviebody << "set mxtics 5" << "\n" << "set mytics 5" << "\n" << "set grid xtics ytics mxtics mytics" << "\n" << "do for [frame=0:" << data.FrameCount-1 << "]{ " << "\n" << "set output \"image.\".frame.\".png\"" << "\n" << "Null(x)=0" << "\n" << "plot ";
      for(int s=0;s<data.S;s++) moviebody << "'mpDPFT_MovieData.dat' using 1:" << column[s] << "+frame*" << data.S*3 << " with lines ls " << s+1 << " title 'Den" << s << " frame='.frame, ";
      moviebody << "Null(x) with lines ls 10 notitle" << "\n" << "}" << "\n" << "EOF" << "\n" << "ffmpeg -analyzeduration 100M -probesize 50M -f image2 -framerate " << ((double)data.FrameCount+1.)/20. << " -i image.%d.png -crf 0 -vf fps=25 -pix_fmt yuv420p mpDPFT_Movie.mp4";
      movieshell << moviebody.str();
  
      movieshell.close();
    }
    
    if(data.TaskType==100){
      PRINT(" ***** export 1pExDFT Data ****",data);
      int L = TASK.ex.settings[0];
      ofstream OptOccNum;
      OptOccNum.open("mpDPFT_1pExDFT_OptOccNum.dat");
      OptOccNum << std::setprecision(data.OutPrecision);
      for(int l=0;l<L;l++){
	if(l<9) OptOccNum << "0";
	OptOccNum << l+1 << " " << TASK.ex.LevelNames[l] << " " << TASK.ex.OccNum[l] << " " << TASK.ex.Phases[l] << endl;
      }
      OptOccNum.close();
/*      ofstream rho1p;
      rho1p.open("mpDPFT_1pExDFT_rho1p.dat");
      rho1p << std::setprecision(data.OutPrecision);
      for(int k=0;k<L;k++){ for(int l=0;l<L;l++){ rho1p << TASK.ex.rho1p[k][l]; if(l<L-1) rho1p << " "; } if(k<L-1) rho1p << endl; }
      rho1p.close(); */     
    }
    
    if(data.FLAGS.KD){
		PRINT(" ***** export Triangulation Data ****",data);
		ExportTriangulation(data);
	}

	PRINT("FreeRAM10 = " + to_string_with_precision((double)getFreeRAM(data),5),data);

    double TotalAbundances = 0.; for(int s=0;s<data.S;s++) TotalAbundances += data.TmpAbundances[s];
    data.FigureOfMerit = GetFigureOfMerit(data);    
    
    data.txtout << "**** Final Output Variables ****\\\\";
    data.txtout << "FinalAbundances              = " << vec_to_str(data.TmpAbundances) << "\\\\";
    data.txtout << "TargetAbundances             = " << vec_to_str(data.Abundances) << "\\\\";
    if(TotalAbundances>0.) data.txtout << "RelativeAbundances           = "; for(int s=0;s<data.S;s++) data.txtout << to_string_with_precision(data.TmpAbundances[s]/TotalAbundances,4) << " "; data.txtout << "\\\\";
    if(data.S==2){
		data.txtout << "NetMagnetization             = " << vec_to_str(GetNetMagnetization(data)) << "\\\\";
		if(data.DIM==3) data.txtout << "NetColumnMagnetization       = " << GetColumnDensityMagnetization(data) << "\\\\";
	}
    data.txtout << "muVec                        = " << vec_to_str(data.muVec) << "\\\\";
    data.txtout << "DeltamuModifier              = " << data.DeltamuModifier << "\\\\";
    data.txtout << "Linear/Pulay-Mixings         = " << data.LinMix << "/" << data.PulMix << "\\\\";
    if(data.TaskType==100){
      data.txtout << "**** 1p-exact DFT ***" << "\\\\";
      data.txtout << "settings (L,DensityExpression,MinEint-type,InitOccNum-type,SearchSpace,NearestProperOccNumQ,Penalty,Hint-Type,CombinedQ): " << "\\\\" << vec_to_str(TASK.ex.settings) << " --- Optimizers: " << vec_to_str(TASK.ex.Optimizers) << "\\\\";
      if(TASK.ex.Optimizers[0]==100){
	data.txtout << "ConjGradDesc Parameters (epsf,epsg,epsx,diffstep) = " << vec_to_str(TASK.ConjGradParam) << "\\\\";
	data.txtout << "CGD gradient evaluations = " << TASK.ex.CGDgr << "\\\\";
	data.txtout << "CGD function evaluations (4 per dimension and gradient) = " << TASK.ex.FuncEvals << "\\\\";
      }
      else if(TASK.ex.Optimizers[0]==101){
	//data.txtout << "PSO Parameters (...) = " << ... << "\\\\";
	data.txtout << "PSO function evaluations = " << TASK.ex.FuncEvals << "\\\\";
      }
      else if(TASK.ex.Optimizers[0]==102){
	//data.txtout << "PSO Parameters (...) = " << ... << "\\\\";
	data.txtout << "LCO function evaluations = " << TASK.ex.FuncEvals << "\\\\";
      }
      else if(TASK.ex.Optimizers[0]==104){
	//data.txtout << "PSO Parameters (...) = " << ... << "\\\\";
	data.txtout << "GAO function evaluations = " << TASK.ex.FuncEvals << "\\\\";
      }
      else if(TASK.ex.Optimizers[0]==105){
	//data.txtout << "PSO Parameters (...) = " << ... << "\\\\";
	data.txtout << "cSA function evaluations = " << TASK.ex.FuncEvals << "\\\\";
      }       
      data.txtout << std::setprecision(-log10(data.RelAcc)) << "OccNum(" << TASK.ex.settings[0] << ") ="; for(int l=0;l<TASK.ex.settings[0];l++) data.txtout << " " << to_string_with_precision(TASK.ex.OccNum[l],16); data.txtout << "\\\\" << "Phases:" << "\\\\"; for(int l=0;l<TASK.ex.settings[0];l++) data.txtout << " " << to_string_with_precision(TASK.ex.Phases[l],16);
	  if(TASK.ex.settings[20]==1){ data.txtout << "\\\\" << "Unitaries:" << "\\\\"; for(int l=0;l<(TASK.ex.settings[0]*TASK.ex.settings[0]-TASK.ex.settings[0])/2;l++) data.txtout << " " << to_string_with_precision(TASK.ex.Unitaries[l],16); data.txtout << std::setprecision(data.OutPrecision) << "\\\\"; }
	  else data.txtout << "\\\\";
      data.txtout << "E1p = " << to_string_with_precision(TASK.ex.Etot-TASK.ex.Eint,16) << "\\\\";
      data.txtout << "Eint = " << to_string_with_precision(TASK.ex.Eint,16) << "\\\\";
      data.Etot = TASK.ex.Etot;
      data.txtout << "********************" << "\\\\";
    }
    else if(data.FLAGS.SCO){
		data.txtout << "SCO: search space dimension = " << to_string(data.EdgeLength) << "\\\\";
	}
    else{
      for(int s=0;s<data.S;s++) data.txtout << "AuxilliaryEkin[s=" << s << "]          = " << vec_to_str(data.AuxEkin[s]) << "\\\\";
      for(int s=0;s<data.S;s++) data.txtout << "AuxilliaryEint[s=" << s << "]          = " << vec_to_str(data.AuxEint[s]) << "\\\\";
      if(data.System==2 || data.FLAGS.ForestGeo) for(int s=0;s<data.S;s++) data.txtout << "Average tauVecMatrix = " << Integrate(data.ompThreads,data.method, data.DIM, data.tauVecMatrix[s], data.frame)/data.area << "\\\\";
      data.txtout << "DispersalEnergies            = " << vec_to_str(data.DispersalEnergies) << "\\\\";
      data.txtout << "AlternativeDispersalEnergies = " << vec_to_str(data.AlternativeDispersalEnergies) << "\\\\";
      data.txtout << "EnvironmentalEnergies        = " << vec_to_str(data.EnvironmentalEnergies) << "\\\\";
      data.txtout << "InteractionEnergies          = " << vec_to_str(data.InteractionEnergies) << "\\\\";
      if(data.S==1 && data.Environments[0]==10) data.txtout << "NucleiEnergy          = " << data.NucleiEnergy << "\\\\";
    }
    data.txtout << "TotalEnergy                  = " << to_string_with_precision(data.Etot,16) << "\\\\";
    data.txtout << "FigureOfMerit                = " << to_string_with_precision(data.FigureOfMerit,16) << "\\\\";
    if(data.FLAGS.ForestGeo){
      vector<double> rD(data.GridSize);
      for(int s=0;s<data.S;s++){
	int sys = data.System-31, column = 30+4*s+sys;
	if(data.System==35 || data.System==36){ sys = data.System-35; column = 15+2*s+sys; }
	for(int i=0;i<data.GridSize;i++) rD[i] = TASK.refData[i][column]/data.Deltax2;
	data.txtout << "xiOverlap with refData[" << s << "] = " << xiOverlap(data.Den[s],rD) << "\\\\";
	if(TASK.CoarseGrain && TASK.refDataType==2){
	  for(int i=0;i<data.GridSize;i++) rD[i] = TASK.FOMrefData[i][column]/data.Deltax2;//fraction of landcover
	  data.txtout << "xiOverlap with FOMrefData[" << s << "] = " << xiOverlap(data.Den[s],rD) << "\\\\";
	}
      }
      if(TASK.Corr.Measure==2) for(int s=0;s<data.S;s++) data.txtout << "PearsonOverlap with refData[" << s << "] = " << pow(Overlap(data.Den[s],rD),1./TASK.Corr.POW) << "\\\\";
      if(TASK.Corr.Measure==3) for(int s=0;s<data.S;s++) data.txtout << "LargeDataDeviants with refData[" << s << "] = " << pow(Overlap(data.Den[s],rD),1./TASK.Corr.POW) << "\\\\";
      for(int s=0;s<data.S;s++) data.txtout << "fomOverlap[" << s << "] = " << fomOverlap(s,data) << "\\\\";
    }
    if(data.K>0){ data.txtout << "ConsumedResources:" << "\\\\";
    for(int k=0;k<data.K;k++) data.txtout << to_string(data.ConsumedResources[k]) << "/" << to_string(data.AvailableResources[k]) << "\\\\"; }
    data.txtout << "SCiterations                 = " << data.SCcount-1 << "/" << data.maxSCcount << "\\\\";
    data.txtout << "ConvergenceCheck             = " << data.CC << "\\\\";
    data.txtout << "ComputationalTime            = " << data.Timing2 << "\\\\";
    data.txtout << "\\newpage \n";
    
    PRINT(data.txtout.str(),data);

    PRINT(" ***** produce plot and movie shells ****",data);
  
    ofstream plotshell; plotshell.open("mpDPFT_Plots.sh");
    ofstream combinedplots; combinedplots.open("mpDPFT_CombinedPlots.tex");
  
  
    ostringstream gnuplotbody, IncludeGraphics;
  
    string tmpfilename, pixelsQ = " with pm3d notitle"; if(data.FLAGS.ForestGeo) pixelsQ = " with image pixels notitle";;
  
    vector<string> plottype = {{"Den"},{"Env"},{"V"}}, plottype2 = {{"Den"},{"Env+V"}};
    double zmin, zmax;
    int linestyle;
  
    //if(ABS(data.Wall)>MP) GetFieldStats(0,1,data);//just for plotting
  
    for(int s=0;s<data.S;s++){//generate plots for all species
    
      for(int p=0;p<3;p++){//generate various plot types for each species
      
	if(p==0){ zmin = data.DenMin[s]-0.1*ABS(data.DenMin[s]); zmax = data.DenMax[s]+0.1*ABS(data.DenMax[s]); }
	else if(p==1){ zmin = data.EnvMin[s]-0.1*ABS(data.EnvMin[s]); zmax = data.EnvMax[s]+0.1*ABS(data.EnvMax[s]); }
	else if(p==2){ zmin = data.VMin[s]-0.1*ABS(data.VMin[s]); zmax = data.VMax[s]+0.1*ABS(data.VMax[s]); }
	if(ABS(zmax-zmin)<MP){ zmin=0.; zmax=1.; }
      
	if(data.DIM>1){
	  tmpfilename = "-2Dcontour" + plottype[p] + to_string(s);//<< "set dgrid3d " << 100*data.EdgeLength << "," << 100*data.EdgeLength << "\n"
	  gnuplotbody = ostringstream(); gnuplotbody << "set title '" << plottype[p] + to_string(s) << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_ContourData.dat' using 1:2:" << (s+1)*3+(p+1) << pixelsQ << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	  IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}"; 
	}
	else IncludeGraphics << data.IncludeGraphicsSTART.str();
      
	if(p<2){
	  if(p==0) linestyle = 1;
	  if(p==1){
	    linestyle = 3;
	    zmin = min((1.-0.1*Sign(min(data.minmaxfi[s][1][0],data.muVec[s])))*min(data.minmaxfi[s][1][0],data.muVec[s]),(1.-0.1*Sign(min(data.minmaxfi[s][2][0],data.muVec[s])))*min(data.minmaxfi[s][2][0],data.muVec[s]));
	    zmax = 1.1*max(max(data.minmaxfi[s][1][1],data.minmaxfi[s][2][1]),ABS(data.muVec[s]-min(data.minmaxfi[s][1][1],data.minmaxfi[s][2][1])));
	  }
	  if(ABS(zmax-zmin)<MP){ zmin=-1.23456789; zmax=1.23456789; }
	  tmpfilename = "-Cut" + plottype2[p] + to_string(s);
	  string CutString = " along (" + to_string_with_precision(data.OutCut[0],3) + "," + to_string_with_precision(data.OutCut[1],3) + ")\\$\\to\\$(" + to_string_with_precision(data.OutCut[2],3) + "," + to_string_with_precision(data.OutCut[3],3) + ")";
	  gnuplotbody = ostringstream(); gnuplotbody << "set title '" << plottype2[p] << to_string(s) << CutString << "'" << "\n" << "set mxtics 5" << "\n" << "set mytics 5" << "\n" << "set grid xtics ytics mxtics mytics" << "\n" << "set yrange [" << to_string(zmin) << ":" << zmax << "]" << "\n" << "mu(x)=" << to_string(data.muVec[s]) << "\n" << "plot 'mpDPFT_CutData.dat' using 1:" << to_string((s+1)*3-1+p) << " with lines ls " << to_string(linestyle) << " title '" << plottype[p] + to_string(s) << "'";
	  if(p==1) gnuplotbody  << ", mu(x) ls 2 title '\\$\\mu\\$', 'mpDPFT_CutData.dat' using 1:" << to_string((s+1)*3+1) << " with lines ls 7 title '" << plottype[p+1] << to_string(s) << "' \n";
	  else gnuplotbody << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	  IncludeGraphics << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
	}
	else{
	  int ptmp=0; linestyle = 1; zmin = data.DenMin[s]-0.1*ABS(data.DenMin[s]); zmax = data.DenMax[s]+0.1*ABS(data.DenMax[s]);
	  tmpfilename = "-LogPlot-Cut" + plottype2[ptmp] + to_string(s);
	  string CutString = " along (" + to_string_with_precision(data.OutCut[0],3) + "," + to_string_with_precision(data.OutCut[1],3) + ")\\$\\to\\$(" + to_string_with_precision(data.OutCut[2],3) + "," + to_string_with_precision(data.OutCut[3],3) + ")";
	  gnuplotbody = ostringstream(); gnuplotbody << "set title 'LogPlot" << plottype2[ptmp] << to_string(s) << CutString << "'" << "\n" << "set logscale y" << "\n" << "set format y \"%e\"" << "\n" << "set samples 1000" << "\n" << "set yrange [" << max(zmin,1.0e-16) << ":" << max(zmax,1.0e-15) << "]" << "\n" << "mu(x)=" << to_string(data.muVec[s]) << "\n" << "plot 'mpDPFT_CutData.dat' using 1:" << to_string((s+1)*3-1+ptmp) << " with lines ls " << to_string(linestyle) << " title '" << plottype[ptmp] + to_string(s) << "'";
	  gnuplotbody << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	  IncludeGraphics << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
	}
      }
      if(data.DIM==2 && data.K>0){
	//limiting resource plots
	zmin = 0.; zmax = (double)data.K;
	tmpfilename = "-2Dcontour-LimitingResourceForS" + to_string(s);
	gnuplotbody = ostringstream(); gnuplotbody << "set title 'LimitingResourceForS" << to_string(s) << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_LimitingResourceData.dat' using 1:2:" << 4+s << pixelsQ << "\n";
	plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
      }
      
      IncludeGraphics << "\n \\FloatBarrier ------------------------------------------------------\\newpage \n";
	  
	  
		if(data.TaskType==61 && TASK.DynDFTe.mode==1){
			zmin = data.DenMin[0]-0.1*ABS(data.DenMin[0]); zmax = data.DenMax[0]+0.1*ABS(data.DenMax[0]);
			for(int sid=0;sid<TASK.DynDFTe.SnapshotID.size();sid++){
				double time = min(1.,(double)TASK.DynDFTe.SnapshotID[sid]/(0.5*(double)data.maxSCcount));
				string SID = to_string(TASK.DynDFTe.SnapshotID[sid]);
				tmpfilename = "-Snapshot_" + SID + "_g";
				plotshell << data.epslatexSTART.str() << tmpfilename << "/'`" << "\n" << "$GNUPLOT << EOF" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "]" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "]" << "\n" << "t=" << time << "\n" << "Env(x,y)=-25.*(0.5*exp(-25.*(x + 0.25 - 0.5*t)**2 - 60.*(y - 0.2*t)**2) + 0.4*exp(-50.*((x + 0.25 - 0.5*t)**2 + (y + 0.2*t)**2)))"  << "\n" << "set isosample 250,250" << "\n" << "set contour base" << "\n" << "set cntrparam level incremental -12, 1.5, 0" << "\n" << "unset surface" << "\n" << "set table 'mpDPFT_DynDFTe_CurrentEnvContours" << tmpfilename << ".dat'" << "\n" << "splot Env(x,y)" << "\n" << "t=1" << "\n" << "set table 'mpDPFT_DynDFTe_FinalEnvContours" << tmpfilename << ".dat'" << "\n" << "splot Env(x,y)" << "\n" << "unset table" << "\n" << "set terminal epslatex dashed size 10,10" << data.epslatexLineStyles.str();
				gnuplotbody = ostringstream(); gnuplotbody << "set title 'Snapshot-" << SID << "-g'" << "\n" << "set multiplot" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "unset colorbox" << "\n" << "set palette defined ( 0 \"white\", 0.15 \"#ccffff\", 0.25 \"#99ccff\", 0.35 \"green\", 0.45 \"yellow\", 0.55 \"orange\", 0.65 \"red\", 0.75 \"#ee00ee\", 1 \"black\" )" << "\n" << "plot 'mpDPFT_DynDFTe_Snapshot_Den_" << SID << ".dat' using 1:2:3 with image notitle" << "\n" << "plot 'mpDPFT_DynDFTe_FinalEnvContours" << tmpfilename << ".dat' w l lw 1.5 dt (5,5) lc \"#aaaaaa\" notitle" << "\n" << "plot 'mpDPFT_DynDFTe_CurrentEnvContours" << tmpfilename << ".dat' w l lw 2 lc \"#aaaaaa\" notitle" << "\n" << "scale = " << TASK.DynDFTe.VectorScale[sid][0] << "\n" << "plot 'mpDPFT_DynDFTe_Snapshot_Vec_" << SID << ".dat' every 2 using 1:2:(scale*\\$4):(scale*\\$5) with vectors head filled dt 1 lw 1 lc rgb \"#bbbbbb\" notitle" << "\n" << "unset multiplot" << "\n";
				plotshell << gnuplotbody.str() << data.epslatexEND.str();
				IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}";
				tmpfilename = "-Snapshot_" + SID + "_v";
				plotshell << data.epslatexSTART.str() << tmpfilename << "/'`" << "\n" << "$GNUPLOT << EOF" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "]" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "]" << "\n" << "t=" << time << "\n" << "Env(x,y)=-25.*(0.5*exp(-25.*(x + 0.25 - 0.5*t)**2 - 60.*(y - 0.2*t)**2) + 0.4*exp(-50.*((x + 0.25 - 0.5*t)**2 + (y + 0.2*t)**2)))"  << "\n" << "set isosample 250,250" << "\n" << "set contour base" << "\n" << "set cntrparam level incremental -12, 1.5, 0" << "\n" << "unset surface" << "\n" << "set table 'mpDPFT_DynDFTe_CurrentEnvContours" << tmpfilename << ".dat'" << "\n" << "splot Env(x,y)" << "\n" << "t=1" << "\n" << "set table 'mpDPFT_DynDFTe_FinalEnvContours" << tmpfilename << ".dat'" << "\n" << "splot Env(x,y)" << "\n" << "unset table" << "\n" << "set terminal epslatex dashed size 10,10" << data.epslatexLineStyles.str();
				gnuplotbody = ostringstream(); gnuplotbody << "set title 'Snapshot-" << SID << "-v'" << "\n" << "set multiplot" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "unset colorbox" << "\n" << "set palette defined ( 0 \"white\", 0.15 \"#ccffff\", 0.25 \"#99ccff\", 0.35 \"green\", 0.45 \"yellow\", 0.55 \"orange\", 0.65 \"red\", 0.75 \"#ee00ee\", 1 \"black\" )" << "\n" << "plot 'mpDPFT_DynDFTe_Snapshot_Den_" << SID << ".dat' using 1:2:3 with image notitle" << "\n" << "plot 'mpDPFT_DynDFTe_FinalEnvContours" << tmpfilename << ".dat' w l lw 1.5 dt (5,5) lc \"#aaaaaa\" notitle" << "\n" << "plot 'mpDPFT_DynDFTe_CurrentEnvContours" << tmpfilename << ".dat' w l lw 2 lc \"#aaaaaa\" notitle" << "\n" << "scale = " << TASK.DynDFTe.VectorScale[sid][1] << "\n" << "plot 'mpDPFT_DynDFTe_Snapshot_Vec_" << SID << ".dat' every 2 using 1:2:(scale*\\$6):(scale*\\$7) with vectors head filled dt 1 lw 1 lc rgb \"#bbbbbb\" notitle" << "\n" << "unset multiplot" << "\n";				
				plotshell << gnuplotbody.str() << data.epslatexEND.str();
				IncludeGraphics << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
			}
			
			ofstream DynDFTeEnergies; DynDFTeEnergies.open("mpDPFT_DynDFTe_Energies.dat");
			DynDFTeEnergies << data.IntermediateEnergies.str();
			DynDFTeEnergies.close();
			int cols = 1+2*data.S+data.Interactions.size()+1;
			
        tmpfilename = "-DynDFTe-Energies";
        gnuplotbody = ostringstream();
        gnuplotbody << "set title 'DynDFTe Energies'" << "\n" << "plot";
        for(int e=2;e<=cols;e++){
            linestyle = e-1;
            if(e==2) gnuplotbody << " "; else gnuplotbody << ", ";
            gnuplotbody << "'mpDPFT_DynDFTe_Energies.dat' using 1:" << to_string(e) << " with lines ls " << to_string(linestyle) << " title '" << data.EnergyNames[e-2] << "'";
        }
        gnuplotbody << "\n";
        plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
        IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();				
			
			IncludeGraphics << "\n \\FloatBarrier ------------------------------------------------------\\newpage \n";
		}
	
    }
  
    if(data.DIM==2){
      //resource plots
      for(int k=0;k<data.K;k++){
	zmin = data.ResMin[k]-0.1*ABS(data.ResMax[k]-data.ResMin[k]); zmax = data.ResMax[k]+0.1*ABS(data.ResMax[k]);
	tmpfilename = "-2Dcontour-Resource" + to_string(k);
	gnuplotbody = ostringstream(); gnuplotbody << "set title 'Resource" << to_string(k) << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_ResourceData.dat' using 1:2:" << 4+k << pixelsQ << "\n";
	plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	if(k==0 || k%3==0) IncludeGraphics << data.IncludeGraphicsSTART.str();
	IncludeGraphics << data.include3Graphics.str() << tmpfilename << "}";
	if((k+1)%3==0 || k==data.K-1) IncludeGraphics << data.IncludeGraphicsEND.str();     
      }
      IncludeGraphics << "\n \\FloatBarrier ------------------------------------------------------\\newpage \n";
      //total density plot
      double TotalDenMin = 0., TotalDenMax = 0.; vector<double> TotalDen(data.Den[0]); for(int s=1;s<data.S;s++) AddField(data,TotalDen, data.Den[s], data.DIM, data.EdgeLength); 
      for(int i=0;i<data.GridSize;i++){	if(TotalDen[i]<TotalDenMin) TotalDenMin = TotalDen[i]; else if(TotalDen[i]>TotalDenMax) TotalDenMax = TotalDen[i]; }
      zmin = TotalDenMin-0.1*ABS(TotalDenMin); zmax = TotalDenMax+0.1*ABS(TotalDenMax);
      ostringstream colString;
      colString << "("; for(int s=0;s<data.S;s++){ colString << "("; colString << "\\" << "$" << (s+1)*3+1; if(s<data.S-1) colString << ")+"; else colString << ")"; } colString << ")";
      tmpfilename = "-2Dcontour-TotalDensity";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'TotalDensity'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_ContourData.dat' using 1:2:" << colString.str() << pixelsQ << "\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();      
      //density difference plot
      if(data.S==2){
	double AbsDiffDenMax = 0.;
	for(int i=0;i<data.GridSize;i++){ zmax = ABS(data.Den[0][i]-data.Den[1][i]); if(zmax>AbsDiffDenMax) AbsDiffDenMax = zmax; }	
	zmin = min(-AbsDiffDenMax,-MachinePrecision); 
	ostringstream colString;
	colString << "("; for(int s=0;s<data.S;s++){ colString << "("; colString << "\\" << "$" << (s+1)*3+1; if(s<data.S-1) colString << ")-"; else colString << ")"; } colString << ")";
	tmpfilename = "-2Dcontour-DensityDifference";
	gnuplotbody = ostringstream(); gnuplotbody << "set title 'DensityDifference'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << AbsDiffDenMax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( -0.05 \"black\", 0.05 \"#0000dd\", 0.1 \"#0000ff\", 0.2 \"#00ddff\", 0.45 \"#ff88ff\", 0.5 \"white\", 0.55 \"#88ff88\", 0.8 \"yellow\", 0.9 \"orange\", 0.95 \"#ff6633\", 1.05 \"#cc0000\")" << "\n" << "splot 'mpDPFT_ContourData.dat' using 1:2:" << colString.str() << pixelsQ << "\n";
	plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
      }
      IncludeGraphics << "\n \\FloatBarrier ------------------------------------------------------\\newpage \n";
    }
  
    if(data.DIM==2 && data.FLAGS.RefData){
      //reference plots
      zmin = data.RefMin-0.1*ABS(data.RefMax-data.RefMin); zmax = data.RefMax+0.1*ABS(data.RefMax);
      tmpfilename = "-2Dcontour-ReferenceDensity";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'ReferenceDensity'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_ReferenceData.dat' using 1:2:4 with pm3d notitle" << "\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();  
    }
    
    if(data.FLAGS.ForestGeo){
      //reference density plots, ForestGeo, L or R for all species
      int sys = data.System-31; if(data.System==35 || data.System==36) sys = data.System-35;
      for(int s=0;s<data.S;s++){
	  int column = 30+4*s+sys; if(data.System==35 || data.System==36) column = 15+2*s+sys;
	  int gnuplotcolumn = column+1;
	  double test, test1, test2, Min = min(TASK.refData[0][column]/data.Deltax2,data.DenMin[s]);
	  if(TASK.CoarseGrain && TASK.refDataType==2) Min = min(Min,TASK.FOMrefData[0][column]/data.Deltax2);
	  double Max = Min;
	  for(int i=0;i<data.EdgeLength;i++){
	    for(int j=0;j<data.EdgeLength;j++){
	      int index = i*data.EdgeLength+j;
	      if(data.System<35 || j<(data.EdgeLength+1)/2){
		test = data.Den[s][index];
		test1 = TASK.refData[index][column]/data.Deltax2;
		if(TASK.CoarseGrain && TASK.refDataType==2) test2 = TASK.FOMrefData[index][column]/data.Deltax2;
		if(test<Min) Min = test; else if(test>Max) Max = test;
		if(test1<Min) Min = test1; else if(test1>Max) Max = test1;
		if(TASK.CoarseGrain && TASK.refDataType==2) if(test2<Min) Min = test2; else if(test2>Max) Max = test2;
	      }
	    }
	  }
	  zmin = Min; zmax = Max;	  
	  
	  tmpfilename = "-2DcontourDen-Prediction" + to_string(s);
	  gnuplotbody = ostringstream(); gnuplotbody << "set title 'Den" << to_string(s) << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set palette defined ( 0 \"white\", 0.15 \"#ccffff\", 0.25 \"#99ccff\", 0.35 \"green\", 0.45 \"yellow\", 0.55 \"orange\", 0.65 \"red\", 0.75 \"#ee00ee\", 1 \"black\" )" << "\n" << "splot 'mpDPFT_ContourData.dat' using 1:2:" << (s+1)*3+(0+1) << pixelsQ << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
	  IncludeGraphics << data.IncludeGraphicsSTART.str() << data.include3Graphics.str() << tmpfilename << "}";

	  tmpfilename = "-2Dcontour-ForestGeo-FOM-density-data-"; if(sys%2==0) tmpfilename += "L-species"; else tmpfilename += "R-species"; tmpfilename += to_string(s);
	  gnuplotbody = ostringstream(); gnuplotbody << "set title '" << tmpfilename << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set palette defined ( 0 \"white\", 0.15 \"#ccffff\", 0.25 \"#99ccff\", 0.35 \"green\", 0.45 \"yellow\", 0.55 \"orange\", 0.65 \"red\", 0.75 \"#ee00ee\", 1 \"black\" )" << "\n" << "splot 'mpDPFT_FOMReferenceData_ForestGeo.dat' using 1:2:(\\$" << gnuplotcolumn << "/" << data.Deltax2 << ")" << pixelsQ << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();	  
	  IncludeGraphics << data.include3Graphics.str() << tmpfilename << "}";
	  
	  tmpfilename = "-2Dcontour-ForestGeo-density-data-"; if(sys%2==0) tmpfilename += "L-species"; else tmpfilename += "R-species"; tmpfilename += to_string(s);
	  gnuplotbody = ostringstream(); gnuplotbody << "set title '" << tmpfilename << "'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set yrange [" << -data.edge/2. << ":" << data.edge/2. << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin << ":" << zmax << "] noreverse nowriteback" << "\n" << "set palette defined ( 0 \"white\", 0.15 \"#ccffff\", 0.25 \"#99ccff\", 0.35 \"green\", 0.45 \"yellow\", 0.55 \"orange\", 0.65 \"red\", 0.75 \"#ee00ee\", 1 \"black\" )" << "\n" << "splot 'mpDPFT_ReferenceData_ForestGeo.dat' using 1:2:(\\$" << gnuplotcolumn << "/" << data.Deltax2 << ")" << pixelsQ << "\n";
	  plotshell <<  data.epslatexSTART.str() << tmpfilename << data.square.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();	  
	  IncludeGraphics << data.include3Graphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
      }
    }     
    
    if(data.KD.UseTriangulation){
      tmpfilename = "-GoodTriangles";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'GoodTriangles(KD)-[A,B]'\n" << "set style fill solid 0.5 noborder\n" << "plot for [i=0:*] 'mpDPFT_AuxData_GnuplotKD.dat' index i using 1:2 with filledcurves closed lc rgb \"blue\" notitle\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
	  
	  double maxA = data.KD.GoodTriangles[data.KD.GoodTriangles.size()-1][2][0] + 1.0e-12;
      tmpfilename = "-GoodTriangles-shifted-mirrored-log-log";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'GoodTriangles(KD)-[log(maxA-A),log(B)]'\n" << "set style fill solid 0.5 noborder\n" << "set logscale xy\n" << "maxA = " << maxA << "\n" << "plot for [i=0:*] 'mpDPFT_AuxData_GnuplotKD.dat' index i using (maxA-\\" << "$" << "1):2 with filledcurves closed lc rgb \"blue\" notitle\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();	
	}
    
    if(data.DIM==3){
      cout << "data.Global3DcolumnDenMax = " << data.Global3DcolumnDenMax << endl;
      zmin = data.Global3DcolumnDenMin-0.1*ABS(data.Global3DcolumnDenMin); zmax = data.Global3DcolumnDenMax+0.1*ABS(data.Global3DcolumnDenMax);
      tmpfilename = "-Overview-Cutalongx-3DcolumnDensity";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'Overview 3DcolumnDensity'" << "\n" << "set mxtics 5" << "\n" << "set mytics 5" << "\n" << "set grid xtics ytics mxtics mytics" << "\n" << "set yrange [" << to_string(zmin) << ":" << to_string(zmax) << "]" << "\n" << "plot";
      for(int s=0;s<data.S;s++){
        linestyle = s+1;
        if(s==0) gnuplotbody << " "; else gnuplotbody << ", ";
        gnuplotbody << "'mpDPFT_CutData.dat' using " << to_string((data.S)*3+2) << ":" << to_string((data.S)*3+3+s) << " with lines ls " << to_string(linestyle) << " title '" << plottype[0] + to_string(s) << "'";
      }
      gnuplotbody << "\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();          
    }

    if(data.Symmetry==1 && data.S==1){
      zmin = data.ScaledDenMin-0.1*ABS(data.ScaledDenMin); zmax = data.ScaledDenMax+0.1*ABS(data.ScaledDenMax);
      if(ABS(zmax-zmin)<MP){ zmin=-1.23456789; zmax=1.23456789; }
      tmpfilename = "-Cut-ScaledDen";
      string xCol = "(\\$1>=0 ? \\$1 : 1/0)", yCol;
      if(data.DIM==1) yCol = "2";
      if(data.DIM==2) yCol = "(\\$1*\\$2)";
      if(data.DIM==3) yCol = "(\\$1>=0 ? \\$1*\\$1*\\$2 : 1/0)";
      gnuplotbody = ostringstream(); gnuplotbody << "set title 'ScaledDen'" << "\n" << "set mxtics 5" << "\n" << "set mytics 5" << "\n" << "set grid xtics ytics mxtics mytics" << "\n" << "set yrange [" << to_string(zmin) << ":" << zmax << "]" << "\n" << "plot 'mpDPFT_CutData.dat' using " << xCol << ":" << yCol << " with lines ls 1 notitle" << "\n";
      plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
      IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
    }
  
    if(data.S>1){//overview cut plot of all densities
        zmin = data.GlobalDenMin-0.1*ABS(data.GlobalDenMin); zmax = data.GlobalDenMax+0.1*ABS(data.GlobalDenMax);
        int p = 0;
        tmpfilename = "-Overview-Cut" + plottype2[p];
        string CutString = " along (" + to_string_with_precision(data.OutCut[0],3) + "," + to_string_with_precision(data.OutCut[1],3) + ")\\$\\to\\$(" + to_string_with_precision(data.OutCut[2],3) + "," + to_string_with_precision(data.OutCut[3],3) + ")";
        gnuplotbody = ostringstream();
        gnuplotbody << "set title 'Overview " << plottype2[p] << CutString << "'" << "\n" << "set mxtics 5" << "\n" << "set mytics 5" << "\n" << "set grid xtics ytics mxtics mytics" << "\n" << "set yrange [" << to_string(zmin) << ":" << zmax << "]" << "\n" << "plot";
        for(int s=0;s<data.S;s++){
            linestyle = s+1;
            if(s==0) gnuplotbody << " "; else gnuplotbody << ", ";
            gnuplotbody << "'mpDPFT_CutData.dat' using 1:" << to_string((s+1)*3-1+p) << " with lines ls " << to_string(linestyle) << " title '" << plottype[p] + to_string(s) << "'";
        }
        gnuplotbody << "\n";
        plotshell <<  data.epslatexSTART.str() << tmpfilename << data.rectangle.str() << data.epslatexLineStyles.str() << gnuplotbody.str() << data.epslatexEND.str();
        IncludeGraphics << data.IncludeGraphicsSTART.str() << data.includeGraphics.str() << tmpfilename << "}" << data.IncludeGraphicsEND.str();
    }

    combinedplots << data.TeX1.str() << data.txtout.str() << IncludeGraphics.str() << data.TeX2.str() << "\n";
  
    plotshell.close();
    combinedplots.close();
  
  }

  freeContainerMemory(data.Den);

  PRINT("FreeRAM11 = " + to_string_with_precision((double)getFreeRAM(data),5),data);
  
  EndTimer("ExportData",data);
}

void CleanUp(datastruct &data){
  if(data.FLAGS.LIBXC){
    delete [] data.LIBXC_rho; data.LIBXC_rho = NULL; 
    delete [] data.LIBXC_sigma; data.LIBXC_sigma = NULL;
    delete [] data.LIBXC_exc; data.LIBXC_exc = NULL;
    delete [] data.LIBXC_vxc; data.LIBXC_vxc = NULL;
    delete [] data.LIBXC_vsigma; data.LIBXC_vsigma = NULL;
  }
}

void StoreTASKdata(taskstruct &task, datastruct &data){
  if(task.Type==1){
    task.dataset[task.count[0]].S = data.S;
    task.dataset[task.count[0]].muVec = data.muVec;
    task.dataset[task.count[0]].TmpAbundances = data.TmpAbundances;
    task.dataset[task.count[0]].Etot = data.Etot;
    task.dataset[task.count[0]].OutPrecision = data.OutPrecision;
    task.dataset[task.count[0]].epslatexSTART << data.epslatexSTART.str();
    task.dataset[task.count[0]].square << data.square.str();
    task.dataset[task.count[0]].rectangle << data.rectangle.str();
    task.dataset[task.count[0]].epslatexLineStyles << data.epslatexLineStyles.str();
    task.dataset[task.count[0]].epslatexEND << data.epslatexEND.str();
    task.dataset[task.count[0]].TeX1 << data.TeX1.str();
    task.dataset[task.count[0]].TeX2 << data.TeX2.str();
    task.dataset[task.count[0]].IncludeGraphicsSTART << data.IncludeGraphicsSTART.str();
    task.dataset[task.count[0]].includeGraphics << data.includeGraphics.str();
    task.dataset[task.count[0]].IncludeGraphicsEND << data.IncludeGraphicsEND.str();
  }
  
  if(task.Type>1){
    if(!data.FLAGS.ForestGeo) TASKPRINT(data.controlfile.str(),task,0); 
    if(task.CGinProgress){
      GLOBALCOUNT++; int VECsize = task.VEC.size(); if(task.Type==44) VECsize = data.VEC.size();
      int requiredEvals =1+4*VECsize, IterationCount = (int)((double)GLOBALCOUNT/(double)requiredEvals)+1, GradEvalCount = GLOBALCOUNT%requiredEvals;
      if(GradEvalCount==0){ IterationCount = (int)((double)GLOBALCOUNT/(double)requiredEvals); GradEvalCount = requiredEvals; }
      //TASKPRINT("\n",task,1);
      string FOM = " Etot = ";
      double OUT = data.Etot;
      if(task.Type==4 || task.Type==44){
	FOM = " FigureOfMerit = ";
	data.FigureOfMerit = GetFigureOfMerit(data);
	OUT = data.FigureOfMerit;
      }
      if(task.Type==2){
	//if(GLOBALCOUNT%(1+4*data.S)==0 || data.SCcount>=data.maxSCcount){ task.Aux = MP; task.AuxVec = data.thetaVecOriginal; }
	if(GLOBALCOUNT%requiredEvals==0 || data.SCcount>=data.maxSCcount){ task.Aux = MP; task.AuxVec = data.thetaVecOriginal; }
	else{
	  task.Aux = 2.+MP; task.V = data.V; task.AuxVec = data.thetaVec;
	  double newtheta = 0.2; SetField(data,task.AuxVec, 1, data.S, newtheta); 
	}
	//cout << "data.thetaVec -> " << vec_to_str(data.thetaVec) << endl;
      }
      string finalinternal = ""; if(GradEvalCount==requiredEvals) finalinternal = "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
      TASKPRINT("ConjugateGradientDescent(" + to_string(task.count[0]) + ") gradient evaluation #" + to_string(IterationCount) + ", internal function evaluation #" + to_string(GLOBALCOUNT) + " completed" + "\n" + FOM + to_string_with_precision(OUT,20) + " [(" + to_string(GradEvalCount) + "/" + to_string(requiredEvals) + ") @ {" + vec_to_str(task.VEC) + "}]" + "\n" + "...SC = " + to_string(data.SCcount) + " ...CC = " + to_string_with_precision(data.CC,16) + "\n" + "...TmpAbundances = " + vec_to_str(data.TmpAbundances) + "\n" + "...resume with InterpolVQ = " + to_string((int)task.Aux) + finalinternal + "\n",task,1);
    }
    if(task.Type==5 || task.Type==6 || task.Type==7){
      data.FigureOfMerit = GetFigureOfMerit(data);
      TASKPRINT("ParameterExploration; FigureOfMerit = " + to_string_with_precision(data.FigureOfMerit,20) + " @ {" + vec_to_str(task.VEC)\
      + "}]\n" +"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",task,1);
    }
    else if(task.Type==8 || task.Type==88){
      if(data.DensityExpression==5){
	task.VEC.clear(); task.VEC.resize(data.S + data.MppVec[0].size()*data.S+1); 
	for(int s=0;s<data.S;s++) task.VEC[s] = data.muVec[s];
	for(int s=0;s<data.S;s++){
	  for(int a=0;a<data.MppVec[s].size();a++){
	    //cout << data.MppVec[s][a] << " " << data.S+s*data.MppVec[s].size()+a << endl;
	    task.VEC[data.S+s*data.MppVec[s].size()+a] = data.MppVec[s][a];
	  }	
	}
	task.VEC[data.S + data.MppVec[0].size()*data.S] = data.DeltamuModifier;
	//cout << "task.VEC -> " << vec_to_str(task.VEC) << endl;
      }
      string EintComponents = "";
      int A=data.Interactions.size();
      for(int a=0;a<A;a++) EintComponents += " --- " + to_string(7+a) + ": Eint(" + to_string(data.Interactions[a]) + ")";
      TASKPRINT("0: species --- 1: stretchfactor --- 2: steps --- 3: Etot --- 4: Edis --- 5: Eenv  --- 6: EInt" + EintComponents + " --- " + to_string(7+A) + ": AuxEkin(1)" + " --- " + to_string(7+A+1) + ": AuxEkin(2)" + " --- " + to_string(7+A+2) + ": AuxEkin(3)" + " --- " + to_string(7+A+3) + ": AuxEkin(4)" + " --- " + to_string(7+A+4) + ": AuxEint(1)" + " --- " + to_string(7+A+5) + ": AuxEint(2)" + " --- " + to_string(7+A+6) + ": AuxEint(3)" + " --- " + to_string(7+A+7) + ": AuxEint(4)" + " --- " + to_string(7+A+8) + ": NucleiEnergy(4)",task,1);
      for(int s=0;s<data.S;s++){
	string energies = to_string(s) + " " + to_string(data.stretchfactor) + " " + to_string(data.steps) + " " + to_string(data.Etot) + " " + to_string(data.DispersalEnergies[s]) + " " + to_string(data.EnvironmentalEnergies[s]) + " " + to_string(data.EInt);
	for(int a=0;a<A;a++) energies += " " + to_string(data.InteractionEnergies[a]);
	for(int a=0;a<4;a++) energies += " " + to_string(data.AuxEkin[s][a]);
	for(int a=0;a<4;a++) energies += " " + to_string(data.AuxEint[s][a]);
	energies += " " + to_string(data.NucleiEnergy);
	task.auxout += energies + "\n";
	TASKPRINT(task.auxout,task,1);
      }
    }
    else if(task.Type==9) data.GradDesc.f = data.Etot;
	else if(task.Type==61){//DynDFTe
		if(TASK.DynDFTe.mode==0){
			TASK.DynDFTe.n0 = data.Den;//initial equilibrium density n0 to be loaded when DynDFTe.mode==1
			VecToFile(TASK.DynDFTe.n0[0],"mpDPFT_DynDFTe_n0.dat");
			#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
			for(int i=0;i<data.GridSize;i++) for(int d=0;d<data.DIM;d++){
				TASK.DynDFTe.v[0][i][d] = data.RN(data.MTGEN);
				TASK.DynDFTe.g[0][i][d] = data.RN(data.MTGEN);
			}
		}
		if(TASK.DynDFTe.mode==1){//dynamics toward final equilibrium density nf
			TASK.DynDFTe.nf = data.Den;
			VecToFile(TASK.DynDFTe.nf[0],"mpDPFT_DynDFTe_nf.dat");
		}
	}
  }
  
  
  
}

void ProcessTASK(taskstruct &task){

  if(task.Type==1){
    ofstream TASKdata;
    TASKdata.open("mpDPFT_TASKenergies.dat");
    TASKdata << std::setprecision(task.dataset[0].OutPrecision);  
  
    vector<double> Nmin(task.dataset[0].TmpAbundances), Nmax(Nmin); 
    double zmin = task.dataset[0].Etot, zmax = zmin;
    for(int count=0;count<task.ParameterCount;count++){
      double Etot = task.dataset[count].Etot; cout << "Etot = " << Etot << endl;
      for(int s=0;s<task.dataset[count].S;s++){
	double Ns = task.dataset[count].TmpAbundances[s];
	if(Ns>Nmax[s]) Nmax[s] = Ns; else if(Ns<Nmin[s]) Nmin[s] = Ns;
	TASKdata << Ns << " ";
      }
      if(Etot>zmax) zmax = Etot; else if(Etot<zmin) zmin = Etot;
      TASKdata << Etot << endl;
      if( (count+1)%task.AxesPoints == 0) TASKdata << endl;
    }
  
    TASKdata.close();

    ostringstream gnuplotbody, IncludeGraphics;
  
    ofstream plotshell; plotshell.open("mpDPFT_TASKplots.sh");
  
    string tmpfilename = "-EnergyLandscape";
    gnuplotbody = ostringstream(); 
  
    if(task.dataset[0].S==2) gnuplotbody << "set title 'Energy(N1,N2)'" << "\n" << "set view map" << "\n" << "unset surface" << "\n" << "set style data pm3d" << "\n" << "set xrange [" << Nmin[0]-0.01*(Nmax[0]-Nmin[0]) << ":" << Nmax[0]+0.01*(Nmax[0]-Nmin[0]) << "] noreverse nowriteback" << "\n" << "set yrange [" << Nmin[1]-0.01*(Nmax[1]-Nmin[1]) << ":" << Nmax[1]+0.01*(Nmax[1]-Nmin[1]) << "] noreverse nowriteback" << "\n" << "set cbrange [" << zmin-0.01*(zmax-zmin) << ":" << zmax+0.01*(zmax-zmin) << "] noreverse nowriteback" << "\n" << "set pm3d implicit" << "\n" << "set palette defined ( 0 \"black\", 0.05 \"blue\", 0.3 \"cyan\", 0.45 \"green\", 0.6 \"yellow\", 0.8 \"orange\", 1 \"red\" )" << "\n" << "splot 'mpDPFT_TASKenergies.dat' using 1:2:3 with pm3d notitle" << "\n";
  
    plotshell <<  task.dataset[0].epslatexSTART.str() << tmpfilename << task.dataset[0].square.str() << task.dataset[0].epslatexLineStyles.str() << gnuplotbody.str() << task.dataset[0].epslatexEND.str(); 
    plotshell.close();
  }
  else if(task.Type==5 || task.Type==6 || task.Type==7){
    ofstream TASKdata;
    TASKdata.open("mpDPFT_TASKParameterExploration.dat");
    TASKdata << std::setprecision(16);
    for(int i=0;i<task.ParameterExplorationContainer.size();i++){
      for(int j=0;j<task.ParameterExplorationContainer[i].size();j++){
	TASKdata << task.ParameterExplorationContainer[i][j] << " ";
      }
      TASKdata << endl;
    }
    TASKdata.close();
  }

}

void LoadReferenceDensity(datastruct &data){//ToDo: 1D, 3D
	StartTimer("LoadReferenceDensity",data);
	
  if(data.DIM==2 && data.FLAGS.RefData){
  data.ReferenceData.resize(data.GridSize);
  
  vector<double> tmpfield; double val;
  ifstream infile;
  infile.open("TabFunc_ReferenceDensity.dat");
  string line;
  int lineCount = 0;
  while(getline(infile, line)){
    istringstream iss(line);
    iss >> val; tmpfield.push_back(val);
    lineCount++;
  }
    
  int n = (int)(pow((double)lineCount,1./data.DDIM)+0.5);
  vector<double> lattice(n);
  for(int i=0;i<n;i++) lattice[i] = -0.5*data.edge+data.edge*(double)i/((double)(n-1));
  real_1d_array x,f; x.setcontent(n, &(lattice[0])); f.setcontent(lineCount, &(tmpfield[0]));
  spline2dinterpolant spline; spline2dbuildbilinearv(x, n, x, n, f, 1, spline);
  
  data.RefMin = 0.; data.RefMax = 0.;
  
  #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
  for(int i=0;i<data.EdgeLength;i++){//the x-index of f has to run fastest for alglib::spline2dbuildbilinearv !!!
    double RD;
    for(int j=0;j<data.EdgeLength;j++){
      RD = spline2dcalc(spline, data.Lattice[i], data.Lattice[data.EdgeLength+j]);
      data.ReferenceData[j*data.EdgeLength+i] = RD;
      if(RD < data.RefMin) data.RefMin = RD; else if(RD > data.RefMax) data.RefMax = RD;
    }
  }  
  }
  EndTimer("LoadReferenceDensity",data);
}

void LoadFOMrefData(int averageQ, datastruct &data, taskstruct &task){
  //first 15 entries: {"x", "y", "elevation", "pH", "Al", "B", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "P", "Zn", "N"}, the remaining entries are the species densities in consecutive pairs for each species s (Table[{"dbh", "bas"}, {s, 1, S}])   
  int EntriesPerline = 671;//i.e. 328 species
  int xSteps = 50;
  int ySteps = 50;  
  int n = data.steps+1;
  int NumQuadrants = n*n;
  task.FOMrefData.clear(); task.FOMrefData.resize(NumQuadrants);
  for(int i=0;i<NumQuadrants;i++) task.FOMrefData[i].resize(EntriesPerline);
  
  if(task.CoarseGrain && task.refDataType==2){
    if(task.maskratio<0.) task.FOMrefData = task.refData;
    else{
      ifstream infile;
      infile.open("TabFunc_ReferenceData_ForestGeoData.dat");
      int NumberOfRows = (xSteps+1)*(ySteps+1);
      vector<vector<double>> tmprefData(NumberOfRows);
      for(int i=0;i<NumberOfRows;i++) tmprefData[i].resize(EntriesPerline);   
      string line;
      int l = 0;
      double val;
      while(getline(infile,line)){
	istringstream iss(line);
	if(line != ""){
	  iss.str(line);
	  for(int j=0;j<EntriesPerline;j++){ iss >> val; tmprefData[l][j] = val; }
	  l++;
	}
      }
      infile.close();
 
      vector<int> Counter(NumQuadrants); fill(Counter.begin(),Counter.end(),0);
      double delta = data.edge/((double)n);
      for(int i=0;i<NumQuadrants;i++) fill(task.FOMrefData[i].begin(),task.FOMrefData[i].end(),0.);
      for(int q=0;q<NumQuadrants;q++) for(int col=0;col<15;col++) task.FOMrefData[q][col] = task.refData[q][col];//coordinates, environments, resources
      
      int j = 0, jMax = task.mask.size()-1, t = task.mask[j];
      for(int i=0;i<NumberOfRows;i++){
	double xp = tmprefData[i][0]+500., yp = tmprefData[i][1]+500.;
	//cout << " xp=" << xp << " yp=" << yp << " i=" << i << " task.mask[j]=" << task.mask[j] << endl;
	if(i==t && yp<=500./*xp<1000.-MP && yp<500.-MP*/){
	  int qx = (int)(xp/delta); if(qx>data.steps) qx = data.steps;
	  int qy = (int)(yp/delta); if(qy>data.steps) qy = data.steps;
	  int q = qx*n+qy;	  
	  Counter[q]++;
	  for(int col=15;col<EntriesPerline;col++) task.FOMrefData[q][col] += tmprefData[i][col];//collect species abundances {either dbh[m] or bas[m^2]} at fine-grained (20m)^2 task.mask positions
	  //cout << "q=" << q << " i=" << i << " t=" << t << " j=" << j << " xp=" << xp << " yp=" << yp << " S0=" << task.FOMrefData[q][15+2*0+1] << endl;
	  j++;
	  if(j<=jMax) t = task.mask[j];
	}
      }      
      double OrigDataDeltax2 = 20.*20., QuadrantDeltax2 = delta*delta, ratio = QuadrantDeltax2/OrigDataDeltax2;
      for(int qx=0;qx<n;qx++){
	for(int qy=0;qy<n;qy++){
	  int q = qx*n+qy;
	  if(Counter[q]>0){
	    double count = ((double)Counter[q]);
	    for(int col=15;col<EntriesPerline;col++){
	      task.FOMrefData[q][col] *= ratio/count;//divide by count: produce average (fine-grained) abundance within coarse-grained Quadrant, ratio: normalize to coarse-grained Quadrant
	    }
	  }
	}
      }
      task.Abundances.clear(); task.Abundances.resize(data.S); fill(task.Abundances.begin(),task.Abundances.begin(),0.);
      for(int qx=0;qx<n;qx++){
	for(int qy=0;qy<n;qy++){
	  int q = qx*n+qy;
	  for(int s=0;s<data.S;s++) task.Abundances[s] += task.FOMrefData[q][15+2*s+(data.System-35)];
	}
      }
      PRINT("task.Abundances = " + vec_to_str(task.Abundances),data);
      
      RiemannBareSumCompensationFactor = (1000.-delta)*(1000.-delta)/1000000.;
      vector<double> integratedAbundances(328);
      for(int s=0;s<328;s++){
	vector<double> RefDen(n*n);
	for(int qx=0;qx<n;qx++){
	  for(int qy=0;qy<n;qy++){
	    int q = qx*n+qy;
	    RefDen[q] = task.FOMrefData[q][15+2*s+(data.System-35)]/(delta*delta);
	  }
	}
	
	integratedAbundances[s] = Integrate(data.ompThreads,data.method, data.DIM, RefDen, data.frame);
      }
      PRINT("all Abundances( integrated, to be used in LoadrefData() ) = " + vec_to_CommaSeparatedString_with_precision(integratedAbundances,8),data);
      //RiemannBareSumCompensationFactor = 1.;
      
      //don't average, it is bad for discontinuous data !!!
//       vector<vector<double>> QuadrantrefData = task.FOMrefData;
//       while(averageQ>0){
// 	for(int col=2;col<EntriesPerline;col++){
// 	  double testFOMrefData = 0., testQuadrantrefData = 0.;
// 	  if(col==15+2*1+(data.System-35)) for(int q=0;q<task.FOMrefData.size();q++) testQuadrantrefData += QuadrantrefData[q][col];
// 	  for(int q=0;q<task.FOMrefData.size();q++){
// 	    if(task.refDataType==2) task.FOMrefData[q][col] = AverageOverNeigbours(q,col,data.steps+1,(data.steps+2)/2,QuadrantrefData);
// 	    else task.FOMrefData[q][col] = AverageOverNeigbours(q,col,data.steps+1,data.steps+1,QuadrantrefData);
// 	  }
// 	  if(col==15+2*1+(data.System-35)){
// 	    for(int q=0;q<task.FOMrefData.size();q++) testFOMrefData += task.FOMrefData[q][col];
// 	    int qy = q%(data.steps+1), qx = (q-qy)/(data.steps+1);
// 	    cout << << q << " " << qx << " " << qy << " " << testQuadrantrefData << " " << testFOMrefData << endl;
// 	  }
// 	}
// 	QuadrantrefData = task.FOMrefData;
// 	averageQ--;
//       }
    
      //task.AbsRes.clear(); task.AbsRes.resize(11); fill(task.AbsRes.begin(),task.AbsRes.end(),0.); 
      //for(int r=0;r<11;r++) for(int q=0;q<task.FOMrefData.size();q++) task.AbsRes[r] += task.FOMrefData[q][4+r];    
    
      //absolute amount of resources (total area), NonUniformResourceDensities will be normalized to one later
      task.AbsRes.clear(); task.AbsRes.resize(11); fill(task.AbsRes.begin(),task.AbsRes.end(),1.); 
    
      cout << "LoadFOMrefData AbsRes = " << vec_to_str(task.AbsRes) << endl;
 
    }
  }
}

void LoadrefData(int averageQ, datastruct &data, taskstruct &task){
  int xSteps = 24;
  int ySteps = 24;
  int EntriesPerline = 1182;
    
  ifstream infile;
  if(task.refDataType==0 || task.refDataType==1){
    //first 30 entries: {"x(L&R)", "y(L&R)", "elevation(L)", "elevation(R)", "pH(L)", "pH(R)", "nitrogen(L)", "nitrogen(R)"}, the remaining entries are the species densities in consecutive four-tuples for each species s (Table[{"dbh(L)", "dbh(R)", "bas(L)", "bas(R)"}, {s, 1, NumSpecies}])    
    if(task.refDataType==0) infile.open("TabFunc_ReferenceData_ForestGeoDataClean_dbhOrdering.dat");
    else infile.open("TabFunc_ReferenceData_ForestGeoDataClean_basOrdering.dat");
  }
  else if(task.refDataType==2){
    //first 15 entries: {"x", "y", "elevation", "pH", "Al", "B", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "P", "Zn", "N"}, the remaining entries are the species densities in consecutive pairs for each species s (Table[{"dbh", "bas"}, {s, 1, S}])    
    infile.open("TabFunc_ReferenceData_ForestGeoData.dat");
    xSteps = 50;
    ySteps = 50;
    EntriesPerline = 671;
  }
  
  int NumberOfRows = (xSteps+1)*(ySteps+1);
  vector<vector<double>> tmprefData(NumberOfRows);
  for(int i=0;i<NumberOfRows;i++) tmprefData[i].resize(EntriesPerline); 
  
  string line;
  int l = 0;
  double val;
  
  while(getline(infile,line)){
    istringstream iss(line);
    if(line != ""){
      iss.str(line);
      for(int j=0;j<EntriesPerline;j++){ iss >> val; tmprefData[l][j] = val; }
      l++;
    }
  }
  infile.close();
  
  if(task.CoarseGrain && task.refDataType==2){
    int n = data.steps+1;
    int NumQuadrants = n*n;
    task.refData.resize(NumQuadrants);
    vector<int> Counter(NumQuadrants); fill(Counter.begin(),Counter.end(),0);
    double delta = data.edge/((double)n);
    for(int i=0;i<NumQuadrants;i++){ task.refData[i].resize(EntriesPerline); fill(task.refData[i].begin(),task.refData[i].end(),0.); }
    for(int row=0;row<NumberOfRows;row++){
      double xp = tmprefData[row][0]+500., yp = tmprefData[row][1]+500.;
      int qx = (int)(xp/delta); if(qx>data.steps) qx = data.steps;
      int qy = (int)(yp/delta); if(qy>data.steps) qy = data.steps;
      int q = qx*n+qy;
      task.refData[q][0] = qx*delta-(500.-0.5*delta);//spatial coordinates of quadrant centres
      task.refData[q][1] = qy*delta-(500.-0.5*delta);
      if(yp<=500./*xp<1000.-MP && yp<500.-MP*/){
	Counter[q]++;
	for(int col=2;col<EntriesPerline;col++) task.refData[q][col] += tmprefData[row][col];
      }
      //cout << "xp = " << xp << " yp = " << yp << " qx = " << qx << " qy = " << qy << " q = " << q << " S0_ref = " << task.refData[q][2] << " " << task.refData[q][15+2*0+(data.System-35)] << endl;   
    }
    double OrigDataDeltax2 = 20.*20., QuadrantDeltax2 = delta*delta, ratio = QuadrantDeltax2/OrigDataDeltax2;
    for(int qx=0;qx<n;qx++){
      for(int qy=0;qy<n;qy++){
	int q = qx*n+qy;
	if(Counter[q]>0){
	  double count = ((double)Counter[q]);
	  for(int col=2;col<EntriesPerline;col++){
	    task.refData[q][col] *= ratio/count;//count: produce average (fine-grained) abundance in coarse-grained Quadrant, ratio: normalize to coarse-grained Quadrant
	  }
	}
      }
    }
  }
  else{
    task.refData.resize(NumberOfRows);
    for(int i=0;i<NumberOfRows;i++) task.refData[i].resize(EntriesPerline);
    task.refData = tmprefData;
  }
  
  vector<vector<double>> QuadrantrefData = task.refData;
  while(averageQ>0){
    for(int col=0;col<EntriesPerline;col++){
      for(int q=0;q<task.refData.size();q++){
	if(col==0 || col==1) task.refData[q][col] = QuadrantrefData[q][col];
	else{//average
	  if(task.refDataType==2) task.refData[q][col] = AverageOverNeigbours(q,col,data.steps+1,(data.steps+2)/2,QuadrantrefData);
	  else task.refData[q][col] = AverageOverNeigbours(q,col,data.steps+1,data.steps+1,QuadrantrefData);
	}
      }
    }   
    QuadrantrefData = task.refData;
    averageQ--;
  }
   
  task.AbsResL.resize(13); task.AbsResR.resize(13);
  task.AbsResL = {{220.558,133.782,159.982,179.843,183.159,151.163,162.677,178.256,193.316,109.228,163.678,116.426,157.261/*aggregated*/}};//absolute amount of resources (Left)
  task.AbsResR = {{179.442,266.218,240.018,220.157,216.841,248.837,237.323,221.744,206.684,290.772,236.322,283.574,242.739/*aggregated*/}};//absolute amount of resources (Right)
  task.AbsRes.clear(); task.AbsRes.resize(11);//absolute amount of resources (total area)
//   if(task.CoarseGrain && task.refDataType==2){
//     fill(task.AbsRes.begin(),task.AbsRes.end(),0.);
//     for(int r=0;r<11;r++) for(int q=0;q<task.refData.size();q++) task.AbsRes[r] += task.refData[q][4+r];
//   }
//   else fill(task.AbsRes.begin(),task.AbsRes.end(),1.);
  fill(task.AbsRes.begin(),task.AbsRes.end(),1.);
  cout << "LoadrefData AbsRes = " << vec_to_str(task.AbsRes) << endl;
}

void LoadFitParameters(vector<double> &LoadedParams){
  ifstream infile;
  infile.open("TabFunc_FitParameters.dat");
  string line;
  int i = 0, EntriesPerline = 2+LoadedParams.size();
  double val;
  fill(LoadedParams.begin(),LoadedParams.end(),0.);
  
  while(getline(infile,line)){
    istringstream iss(line);
    if(line != ""){
      iss.str(line);
      if(i>0) for(int j=0;j<EntriesPerline;j++){ iss >> val; if(j>1) LoadedParams[j-2] += val; }
      i++;
    }
  }
  infile.close();  
  
  for(int j=0;j<LoadedParams.size();j++) LoadedParams[j] /= (double)(i-1);
}

void LoadFitResults(vector<double> &LoadedParams){//take parameters from first line of mpDPFT_FitResults.dat
  ifstream infile;
  infile.open("mpDPFT_FitResults.dat");
  string line;
  int EntriesPerline = 1+LoadedParams.size();
  double val;
  fill(LoadedParams.begin(),LoadedParams.end(),0.);
  
  while(getline(infile,line)){
    istringstream iss(line);
    if(line != ""){
      iss.str(line);
      for(int j=0;j<EntriesPerline;j++){ iss >> val; if(j>0) LoadedParams[j-1] = val; }
    }
    break;
  }
  infile.close();
}

vector<double> LoadIntermediateOPTresults(void){//take parameters from first line of mpDPFT_FitResults.dat
  ifstream infile;
  infile.open("mpDPFT_IntermediateOPTresults.dat");
  string line,tmp;
  double val;
  vector<double> LoadedOPTresults(0);
  
  int count = 0;
  while(getline(infile,line)){
    istringstream iss(line);
    while(iss>>tmp){
      if(count>0){
	stringstream s; 
	s<<tmp;
	s>>val;
	LoadedOPTresults.push_back(val);
      }
      count++;
    }
    break;
  }
  infile.close();
  cout << "LoadIntermediateOPTresults(" << LoadedOPTresults.size() << "): " << vec_to_str(LoadedOPTresults) << endl;
  return LoadedOPTresults;
}

double AverageOverNeigbours(int row, int col, int n, int ny, vector<vector<double>> &tmprefData){
  double av = 0.;
  int j = row%n, i = (row-j)/n; 
  if(i>0 && i<n-1 && j>0 && j<ny-1) for(int k=-1;k<2;k++) for(int l=-1;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/9.;
  else if(i==0 && j>0 && j<ny-1) for(int k=0;k<2;k++) for(int l=-1;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/6.;
  else if(i==n-1 && j>0 && j<ny-1) for(int k=-1;k<1;k++) for(int l=-1;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/6.;
  else if(j==0 && i>0 && i<n-1) for(int k=-1;k<2;k++) for(int l=0;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/6.;
  else if(j==ny-1 && i>0 && i<n-1) for(int k=-1;k<2;k++) for(int l=-1;l<1;l++) av += tmprefData[(i+k)*n+(j+l)][col]/6.;
  else if(i==0 && j==0) for(int k=0;k<2;k++) for(int l=0;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/4.;
  else if(i==0 && j==ny-1) for(int k=0;k<2;k++) for(int l=-1;l<1;l++) av += tmprefData[(i+k)*n+(j+l)][col]/4.;
  else if(i==n-1 && j==0) for(int k=-1;k<1;k++) for(int l=0;l<2;l++) av += tmprefData[(i+k)*n+(j+l)][col]/4.;
  else if(i==n-1 && j==ny-1) for(int k=-1;k<1;k++) for(int l=-1;l<1;l++) av += tmprefData[(i+k)*n+(j+l)][col]/4.;
  else av = tmprefData[row][col];
  return av;
}

void LoadAbundances(datastruct &data){
  //ForestGeoData, see Ecology_DFT_ForestGeo.nb
  
  data.Abundances.clear(); data.Abundances.resize(data.S);
  if( (TASK.CoarseGrain && TASK.refDataType==2 && TASK.maskratio>0.) && (TASK.optInProgress || TASK.fitQ==0 || TASK.fitQ==2 || TASK.fitQ==4 || TASK.fitQ==5) ){
    data.Abundances = TASK.Abundances;
    //cout << "LoadAbundances: " << vec_to_str(data.Abundances) << endl;
  }
  else{  
  int NumberOfSpecies;
  if(TASK.refDataType==2) NumberOfSpecies = 328;
  else NumberOfSpecies = 288;
  vector<double> abundances(NumberOfSpecies);
  
  if(TASK.refDataType==0){
    if(data.System==31) abundances = {{574796.,400909.,299850.,189873.,105263.,143785.,108716.,95666.7,82480.9,103904.,106080.,128338.,83801.6,49283.3,71216.9,29799.3,92924.1,81526.4,48269.9,34908.6,19825.6,48936.6,68843.6,50448.1,62012.6,40368.3,44495.4,59075.9,48987.4,51586.,64664.3,42339.1,48092.6,41636.,27948.,35419.7,29877.1,12704.7,18295.,34471.7,25883.,47872.9,31951.1,52795.9,33376.,12398.,30861.9,25794.7,30701.4,29780.9,20563.7,22652.1,23486.9,20537.,16701.4,23010.7,20467.3,25885.4,24525.3,17152.1,16046.9,20981.4,13325.1,10551.4,14702.7,19196.3,17901.3,16284.1,10758.7,12674.,18933.,7916.14,16192.1,12854.3,14480.7,17764.1,15846.1,14545.3,20342.9,10498.4,14574.1,11019.7,13144.7,9080.29,12351.9,11780.1,3845.71,11041.9,5096.57,10103.1,11977.6,934.714,14772.,10803.,10492.3,2555.86,10105.4,12984.6,8010.57,4998.86,10196.4,10612.9,6438.,9909.43,9094.43,6027.29,8667.14,5545.29,8296.71,3471.43,8091.,5062.43,8750.86,5404.71,6812.14,6827.57,8828.43,6306.14,5536.57,7829.57,5322.43,10893.,11488.1,5123.71,10390.6,7078.43,5584.86,5325.71,7372.43,7741.14,5591.14,6226.71,5939.57,8930.57,10191.7,4828.57,1940.86,6533.43,9097.14,5488.86,6199.29,491.857,6557.,4381.,4933.71,3747.29,7513.29,2083.86,11.,3560.29,2928.43,2179.14,7481.29,4416.29,3021.71,2667.43,1261.14,3653.,4463.,3467.57,3525.,3047.,3575.29,3001.29,3445.71,2909.71,2679.29,2384.86,4742.86,2384.,239.429,1897.43,2716.14,3178.14,3485.14,4096.57,2347.,2663.57,2750.86,2134.14,610.286,2155.71,1389.43,3783.,2560.86,1560.29,2732.14,2072.29,2096.57,1237.43,2281.29,1206.29,1893.,2119.57,1306.71,638.857,1075.14,1209.29,1906.,2391.43,1428.14,2021.57,809.,604.857,1558.,354.714,759.286,1940.29,1482.29,485.714,979.429,1751.86,891.857,925.,980.857,583.714,664.143,731.714,1468.71,714.286,832.714,533.,66.,579.143,897.857,639.429,312.,820.429,797.857,424.,121.143,242.714,134.286,648.,354.571,437.714,498.571,553.714,509.286,180.571,362.429,226.571,123.857,173.714,310.857,155.429,287.571,130.,33.1429,151.,41.5714,143.,123.143,171.429,178.429,17.7143,8.57143,111.143,117.714,139.143,44.5714,13.4286,15.8571,5.71429,82.4286,35.1429,48.2857,26.8571,27.5714,68.7143,32.2857,52.7143,1.71429,37.1429,23.2857,25.2857,17.8571,7.85714,9.85714,10.1429,2.14286,9.28571,7.,3.57143}}; //Abundances(L,dbh)
    else if(data.System==32) abundances = {{517562.,364568.,399899.,228042.,211003.,152748.,130954.,105787.,114567.,86160.4,61364.4,34616.4,75030.3,99930.4,71652.,100217.,37045.4,47094.4,74925.3,85084.4,93853.9,51932.9,28410.3,46439.6,32457.9,52179.3,46583.7,30073.7,40127.4,36522.9,21868.6,43379.3,37425.,40440.9,49896.4,41514.,45515.4,61145.,53726.6,35428.1,38385.,15839.4,31543.4,10151.4,22705.6,43061.4,23367.,25458.,17725.1,18439.,27117.,24518.9,22051.4,21707.,24634.3,17692.1,19110.6,13375.9,13809.,19457.3,19649.,14323.4,21437.9,23790.4,18932.9,14175.,14997.3,16500.6,21956.1,19180.,12738.7,22526.3,12121.3,15368.9,12625.7,9109.,10720.6,11992.3,5555.14,15350.,10796.1,14297.4,12059.4,15480.3,12008.7,11800.,19153.4,10873.4,16718.4,11633.6,9405.29,20396.4,5752.86,9482.14,8936.14,16221.9,8472.57,4905.14,8399.14,10784.4,5581.57,5035.43,9086.14,5534.,5847.57,8900.14,6250.43,8940.14,6139.,10899.6,6216.43,9069.71,5208.,8396.43,6931.57,6913.71,4899.86,7311.71,7898.43,5456.43,7759.14,2169.,1557.43,7820.86,1815.71,4881.71,6317.86,6542.29,4347.57,3886.71,5484.29,4591.29,4860.57,1830.86,367.714,5287.14,8013.71,2983.43,186.857,3745.29,3026.43,8529.86,2428.86,3971.43,3334.14,4493.71,715.714,5752.57,7812.29,4241.,4764.71,5485.,33.7143,2655.,4000.14,4265.57,5548.14,3125.29,2282.14,3199.57,3135.29,3512.14,2423.86,2812.43,2285.86,2685.14,2878.86,2843.86,399.857,2597.71,4730.57,3072.14,2163.71,1679.57,1274.29,569.429,2171.86,1666.29,1459.43,2007.,3422.,1866.57,2426.43,26.2857,885.286,1830.57,641.857,1247.57,1121.57,1938.86,755.857,1737.14,1005.86,775.714,1516.43,2143.57,1550.14,1385.86,530.143,13.7143,943.286,307.143,1340.71,1506.71,540.143,1696.71,1233.29,28.5714,429.857,1390.57,893.571,57.2857,881.,838.857,780.714,1132.71,979.857,838.714,63.5714,737.857,588.571,669.857,1133.14,531.143,28.5714,268.714,533.429,2.,14.5714,305.857,605.143,437.571,544.714,12.,287.714,203.714,80.7143,10.1429,28.2857,278.286,73.5714,196.857,291.429,188.571,35.2857,180.429,32.4286,180.714,276.714,139.429,228.571,89.8571,107.286,48.2857,22.,175.857,172.714,64.8571,51.1429,10.2857,95.8571,119.714,112.571,114.143,6.14286,51.5714,34.5714,55.1429,49.1429,7.71429,40.,5.42857,44.8571,4.,10.2857,2.42857,6.85714,9.42857,5.28571,3.57143,9.42857,1.42857,3.14286,3.71429}}; //Abundances(R,dbh)  
    else if(data.System==33) abundances = {{2.42237e+7,6.32947e+6,4.13576e+7,2.62813e+7,2.56179e+7,3.00199e+6,8.32557e+6,5.49731e+6,8.14805e+6,1.51786e+7,7.33052e+6,5.04888e+7,9.30787e+6,5.26342e+6,1.37e+7,4.9458e+6,1.80167e+7,1.05437e+6,1.56367e+6,3.73114e+6,1.67726e+6,1.4199e+6,2.43086e+7,5.58894e+6,7.52471e+6,2.2006e+6,1.37829e+7,1.71468e+6,3.4795e+6,8.02601e+6,9.26024e+6,1.20086e+7,8.38308e+6,2.15035e+6,603332.,4.00491e+6,251291.,9.07311e+6,4.04549e+6,1.77977e+6,8.44569e+6,1.703e+6,2.09976e+6,3.4855e+6,2.49228e+6,704526.,1.35041e+6,448154.,3.25993e+6,6.25297e+6,3.7297e+6,2.40431e+7,2.03366e+6,8.15276e+6,701753.,3.5966e+6,1.1039e+6,1.62901e+6,645981.,5.08434e+6,1.40625e+6,854551.,2.07973e+6,326884.,834320.,3.23327e+6,2.16145e+6,371638.,1.15381e+6,587150.,580943.,1.95277e+6,1.22163e+6,1.39347e+7,1.40551e+6,454336.,7.41592e+6,742866.,3.6991e+6,2.33427e+6,4.60018e+6,6.85395e+6,2.95879e+6,98055.3,234045.,5.46526e+6,104469.,1.16424e+6,388150.,804500.,1.43209e+6,615587.,85123.4,739387.,672755.,785507.,1.80366e+6,515034.,3.58087e+6,155660.,2.55518e+6,36777.8,221218.,358702.,1.43017e+6,246738.,159867.,380713.,192134.,154932.,2.69134e+6,1.71478e+6,306842.,427992.,224785.,474944.,366104.,714760.,575782.,1.92532e+6,1.73517e+6,3.27093e+6,230021.,191281.,613766.,447118.,2.03919e+6,550919.,133576.,383188.,432848.,319014.,728983.,2.85415e+6,500708.,104061.,149714.,508371.,239291.,121197.,591097.,36475.2,36586.9,334430.,776347.,137483.,141572.,1.85105e+6,71.7597,2.31157e+6,832586.,450370.,420344.,393501.,2.1728e+6,339821.,43632.5,138444.,704726.,238163.,260245.,386913.,259330.,301645.,190411.,300567.,383511.,115400.,663374.,290897.,36226.9,205005.,230431.,112212.,190144.,16699.4,195829.,189818.,1.44569e+6,59188.1,29122.,370612.,183968.,715757.,84019.7,20508.7,474808.,1.98232e+6,326879.,6308.16,122605.,138754.,713027.,78386.7,13444.4,20506.7,112143.,438602.,45472.6,134208.,25051.8,32620.4,71079.4,51113.3,484595.,50729.6,6495.96,377202.,38948.4,23786.1,5590.62,115406.,21890.4,342995.,46989.7,29497.8,65684.1,5309.13,111951.,10117.1,59391.,7257.13,1822.19,61890.4,50201.,4840.91,16962.1,359356.,29517.,3715.99,3177.72,30710.7,506.854,65642.7,66189.4,49835.6,1696.91,23863.5,168174.,1802.95,19959.3,9566.73,313.47,1435.45,57251.4,3311.27,7169.16,12157.6,109.892,9151.48,1357.31,332.223,6922.95,1886.37,1824.61,206.608,32.0571,8913.69,471.014,313.005,899.425,26.1586,42.5558,25.6457,482.571,288.61,349.454,288.674,113.113,1158.24,378.37,2023.14,2.30811,814.762,65.9895,249.42,35.6314,48.4863,76.3119,14.7623,3.60642,11.4283,16.6857,5.01693}}; //Abundances(L,basal area)
    else if(data.System==34) abundances = {{2.40733e+7,5.3967e+6,3.88839e+7,3.7884e+7,6.96348e+7,3.18421e+6,1.14179e+7,5.66841e+6,1.45757e+7,1.2407e+7,4.31161e+6,9.00587e+6,8.50192e+6,1.03352e+7,1.39102e+7,1.48389e+7,6.44994e+6,663033.,2.75065e+6,1.06754e+7,8.75623e+6,1.57674e+6,8.87914e+6,8.60843e+6,4.24025e+6,3.23027e+6,1.32612e+7,972411.,3.79221e+6,5.85296e+6,3.28328e+6,1.72604e+7,6.57448e+6,2.23905e+6,1.06434e+6,4.63632e+6,402732.,5.29047e+7,1.37955e+7,1.75336e+6,1.25427e+7,535037.,2.36426e+6,604258.,1.81765e+6,2.83108e+6,1.63502e+6,399588.,2.24821e+6,4.24261e+6,6.39346e+6,3.20292e+7,1.60697e+6,9.67969e+6,1.12942e+6,2.27557e+6,998373.,970326.,425029.,5.94044e+6,1.72362e+6,456697.,4.64057e+6,852927.,1.49793e+6,2.91465e+6,1.62886e+6,389931.,2.87335e+6,865647.,364499.,6.35014e+6,819857.,1.66478e+7,1.19542e+6,257666.,5.17317e+6,640326.,1.65961e+6,3.59132e+6,3.56222e+6,1.14393e+7,3.35683e+6,192884.,191915.,5.57281e+6,475072.,978751.,1.86461e+6,874232.,1.28422e+6,2.29466e+7,30442.5,561233.,671429.,5.35891e+6,1.71232e+6,149728.,4.20568e+6,241964.,1.04664e+6,19156.4,384947.,117855.,820207.,462711.,83293.3,497344.,136051.,545222.,1.43591e+6,2.87842e+6,182038.,507604.,220816.,682796.,193859.,1.05578e+6,865645.,1.37397e+6,4.6679e+6,886432.,33815.6,273174.,176423.,248496.,2.65923e+6,989839.,68482.6,198657.,456962.,226979.,559354.,465164.,35191.3,107107.,1.10448e+6,193317.,5245.69,84056.3,389796.,1.62747e+6,12549.4,294528.,625716.,147095.,8874.97,7.17028e+6,366920.,2.79872e+6,1.4353e+6,1.21307e+6,216.962,157902.,2.12438e+6,677988.,256092.,131067.,388046.,211194.,194949.,362035.,224920.,314979.,170878.,403710.,1.29107e+6,140892.,55369.1,295983.,4.45606e+6,211109.,214443.,58342.2,69935.4,2189.24,329307.,133621.,817047.,51238.5,813765.,458923.,363043.,461.397,24795.7,18875.5,66415.,1.22242e+6,176774.,9984.11,25695.9,235368.,284909.,18626.2,16433.2,70017.7,154247.,209653.,18885.6,75.0135,14935.,5948.77,121578.,84449.6,107990.,188218.,13491.8,641.141,8407.24,116243.,5226.86,1718.85,13493.9,279203.,39692.3,132490.,352783.,7046.03,3174.05,16283.8,30719.2,9756.68,66646.1,75659.1,641.141,2148.06,23330.7,3.14159,137.268,2816.07,25813.1,127568.,2622.28,39.5264,17123.3,13095.1,207.329,80.7998,628.383,3047.02,1652.08,3724.23,864.868,2159.27,977.885,3513.73,397.331,16411.5,1492.34,10436.4,10643.4,199.155,2659.02,721.508,128.773,20755.2,1456.75,3303.74,278.608,30.0375,3037.04,1450.2,243.441,10232.6,14.8905,938.166,196.862,2116.38,219.976,27.9858,262.067,23.1452,158.234,12.5664,15.1309,4.63225,9.36066,28.6911,21.9431,5.20927,35.1987,1.60285,7.75781,10.8353}}; //Abundances(R,basal area)    
  }
  else if(TASK.refDataType==1){
    if(data.System==31) abundances = {{105127.,287248.,192467.,12587.,129697.,22633.5,579003.,69091.6,12864.4,41729.5,43931.4,69114.5,105552.,77951.3,94784.1,937.,24792.,110005.,18032.5,81198.6,29259.3,20415.5,54719.1,11024.3,49431.4,48528.9,48514.6,35167.4,62198.,392262.,65619.8,30068.5,15499.4,106474.,17180.3,11693.5,20783.1,20308.4,47959.5,36073.6,1827.88,8275.75,14137.4,3684.63,8094.75,143653.,17911.1,13211.6,13061.5,43377.5,5508.,24337.1,39159.,30246.9,2630.63,10221.9,19357.6,40950.4,16465.4,31890.8,226.375,50601.6,50460.4,5211.88,31695.9,3070.5,53819.,11520.8,5623.5,10064.8,8188.,10600.,30701.6,10372.9,34598.3,13869.5,23501.9,18139.6,7875.75,12098.8,8895.63,47758.9,2054.13,60203.4,26977.6,14484.6,13808.4,20591.,2785.88,47684.6,8573.,4925.38,6007.38,2691.,16322.6,6023.13,11048.5,82672.,15797.9,14283.5,5115.,2120.75,9920.75,10660.3,2797.25,18120.,27373.6,492.5,12854.5,9140.63,10382.3,1906.25,6151.75,21573.8,4813.13,4250.88,6169.38,10461.5,24394.6,6496.75,18867.3,5247.13,2574.88,5641.25,6918.25,10153.5,3531.38,25807.,17518.6,28061.4,6270.5,5554.5,2915.38,5384.,1656.38,2194.63,7256.75,3868.,842.125,3043.88,3237.38,632.5,2113.38,2753.5,12907.3,4289.75,8542.38,2675.75,20.375,9873.38,1216.75,6467.38,9890.38,5807.5,1561.,10591.,2353.25,2078.13,1130.,1938.5,2412.38,3902.13,1419.5,9168.25,1966.63,3241.88,5201.5,4250.13,3530.25,8575.88,657.25,11825.3,12437.4,1616.63,6933.88,15510.4,1154.88,3201.38,4943.63,1046.63,3182.5,3482.88,1150.75,8367.25,2818.75,3680.88,3622.,1223.25,10019.6,8562.25,2323.38,5651.13,7324.5,4737.63,349.375,299.,2415.63,7006.,562.625,758.5,2592.25,212.375,1500.88,576.75,469.75,592.75,2158.5,489.25,810.5,2471.38,9364.63,1422.5,1952.75,6871.63,975.375,387.625,1862.25,2069.63,635.75,1827.25,325.125,68.625,408.5,1365.38,4309.63,1238.38,764.375,879.875,1437.38,134.375,15.5,683.625,1172.13,308.125,729.375,231.25,136.875,477.75,499.375,401.625,990.5,175.75,107.75,669.75,128.625,257.75,43.375,629.,5.,162.875,152.,411.625,151.25,191.,43.625,51.625,129.125,520.,173.,39.,124.625,7.5,11.75,127.125,36.5,181.75,60.125,46.125,33.75,28.25,134.75,51.75,153.625,16.,40.,22.125,28.375,32.,1.5,5.5,16.5,9.25,28.125,20.375,15.5,4.75,9.625,6.125}}; //Abundances(L,dbh)
    else if(data.System==32) abundances = {{208959.,393999.,230265.,60931.4,35931.8,24495.1,523331.,28863.4,15155.,42394.5,46627.,71553.8,88037.4,108520.,38091.9,20802.,38027.1,132639.,52867.4,73007.5,99396.6,21597.5,38040.5,14663.9,101544.,44457.6,37479.3,84813.1,32851.,355050.,21942.5,18225.4,10119.5,61337.5,19367.5,11821.1,26378.,95090.4,40136.,42096.1,5449.63,23088.1,10573.9,4174.25,8206.38,151636.,13129.6,20893.1,12066.3,55774.6,7739.13,18762.9,49413.6,17179.,16356.1,15127.3,5336.5,40670.4,20612.4,21370.5,4790.88,10266.6,77518.3,9487.88,31151.9,3926.5,51970.6,2312.75,6444.13,5523.38,6214.38,21364.5,22088.8,8694.75,35371.1,46423.3,21931.,15202.9,5242.,9070.13,1842.38,50768.1,1313.25,30334.4,13692.1,18905.5,12143.5,19112.9,4548.25,16883.3,5421.75,15877.3,8206.5,1411.63,12233.6,6962.13,10747.6,47626.5,23244.1,12011.5,6227.5,5533.38,11203.3,9255.75,2993.88,9244.13,48719.6,8414.63,19056.4,5014.38,8748.63,7852.,5099.,14707.9,3323.38,2429.,3278.,23566.1,13809.8,6470.38,12671.,7853.38,4267.25,5637.13,10499.8,1758.63,3456.63,25368.5,17271.9,41995.1,2865.75,8809.75,2639.63,454.5,908.25,1903.63,5066.25,31.25,780.125,3564.,10378.5,3577.,1264.38,2581.38,4926.88,4010.63,55.25,680.75,8014.25,16211.1,1751.5,8997.13,5363.,4210.25,548.125,590.5,2070.,1109.5,1337.,3423.5,2566.38,19155.,2445.25,5131.13,25.,3008.75,7889.88,2585.75,2454.25,5044.5,981.625,1592.25,12079.8,46.,6817.75,5833.75,1590.63,1262.5,11113.5,2.,2157.63,1723.13,1666.13,6240.88,1749.25,4437.88,3125.75,5509.5,207.75,6461.75,2883.63,3865.13,4245.38,5163.,1626.88,682.25,762.,648.,1428.25,1308.,14.625,440.375,40.75,582.625,24.75,36.75,793.,1431.63,544.375,914.625,4446.38,55.625,1762.25,2635.,837.5,295.125,1973.63,302.5,2231.,545.125,30.875,1101.,193.75,376.125,578.875,1443.13,1215.38,826.625,984.625,732.75,153.875,699.125,1786.63,503.75,12.75,194.5,199.25,597.625,8.875,69.125,887.,122.,104.,780.75,79.875,28.375,217.375,266.5,114.875,249.75,165.,308.75,178.5,26.125,278.25,55.875,484.875,96.375,51.125,111.875,335.75,151.125,132.,45.625,11.75,106.,6.75,4.75,48.5,35.,51.,30.25,12.,109.75,3.5,2.125,43.,9.375,46.25,13.375,4.625,10.,7.375,9.,3.125,6.25,1.25,2.75}}; //Abundances(R,dbh)  
    else if(data.System==33) abundances = {{29.271,49.7894,28.7879,9.86755,54.4365,26.8518,29.0649,27.455,14.7698,13.6176,15.4051,15.4669,16.8943,10.8819,21.037,0.613151,9.91032,9.56607,4.998,11.2286,5.27605,8.61219,10.9408,7.10369,6.03901,7.32325,9.79485,4.1856,10.35,8.46782,11.4909,8.94753,8.22366,8.13512,5.33044,6.07664,4.34829,1.841,5.43242,4.49469,2.02653,2.21082,4.87087,3.18853,3.98762,3.97473,4.15652,2.45706,3.41795,3.20009,2.15115,4.1859,2.80072,3.92323,0.839164,2.43322,4.41324,2.75792,2.45456,3.03183,0.22117,4.43182,1.92463,1.85405,2.37596,2.35433,2.70503,4.00084,2.08305,3.19871,2.82011,1.26037,1.84896,2.20337,2.19698,0.872258,2.38973,2.379,2.36735,2.03834,3.24697,1.74422,2.10534,2.21999,2.03821,1.08535,1.53154,1.48638,0.957271,2.02138,1.72702,0.479629,1.01161,1.44897,1.49598,1.02946,1.2757,1.45041,0.880404,1.13647,0.808786,0.551366,0.936997,1.00774,0.465376,1.19444,0.664731,0.0908103,0.728029,1.1634,0.89521,0.256666,0.849811,0.989454,0.859459,0.945535,0.903188,0.397849,0.857752,0.579393,0.756094,0.509789,0.367468,0.564946,0.436161,0.879589,0.599985,0.55666,0.515899,0.401488,0.738985,0.419937,0.550119,0.91278,0.681838,0.474257,0.586668,0.917915,0.418295,0.480429,0.20111,0.033393,0.512971,0.425755,0.641913,0.413362,0.787892,0.635327,0.00114462,0.275195,0.265843,0.263738,0.493861,0.389896,0.568527,0.608022,0.237942,0.416668,0.411772,0.293961,0.312945,0.116001,0.219126,0.413675,0.623592,0.327143,0.24592,0.407599,0.316279,0.340456,0.175612,0.449561,0.276223,0.490541,0.250897,0.360499,0.182848,0.348619,0.172507,0.437047,0.237618,0.295855,0.160589,0.229165,0.23372,0.177638,0.180043,0.0520548,0.325572,0.205988,0.130482,0.160387,0.178075,0.124483,0.0538684,0.0582341,0.192837,0.219674,0.0861102,0.0773511,0.193406,0.054504,0.170937,0.0714689,0.150069,0.145609,0.113873,0.0267136,0.0898474,0.10224,0.0926415,0.124656,0.0668248,0.0853079,0.0612971,0.0820685,0.0558343,0.0915223,0.0229988,0.0733652,0.0967547,0.00207649,0.0592175,0.0582804,0.057041,0.0271501,0.0229659,0.0356316,0.0321896,0.00629349,0.000514632,0.0177462,0.0164778,0.0179586,0.0411145,0.025452,0.0152451,0.0154494,0.03399,0.0323883,0.0164127,0.0166768,0.0210817,0.0113926,0.0144852,0.0176383,0.00161547,0.00751233,0.0000539961,0.00413797,0.00475225,0.0054594,0.00434855,0.00790876,0.000975563,0.00341737,0.00153084,0.00553392,0.00438547,0.00144631,0.00115719,0.00019635,0.000160221,0.00332371,0.00334501,0.00286513,0.00375352,0.00249825,0.000917541,0.00102161,0.00139349,0.00135403,0.00161841,0.00021677,0.00152446,0.0015277,0.000310919,0.000707055,0.0000141372,0.000104654,0.000327315,0.000198117,0.000273024,0.000229434,0.000145691,0.0000473202,0.0000841358,0.0000633227}}; //Abundances(L,basal area)
    else if(data.System==34) abundances = {{78.993,45.8743,40.6672,55.8606,10.3087,32.9492,28.3821,11.083,17.994,18.9863,15.2814,15.148,13.3571,18.1711,7.35469,24.1029,14.6734,12.9424,17.1832,9.77887,15.4234,10.4509,8.04835,11.7329,11.8287,10.0532,7.38703,11.8174,5.63391,7.26991,3.92762,5.90699,6.2217,4.77132,6.64486,5.79848,7.39919,9.81319,5.70857,5.14421,7.20125,6.9007,3.99634,5.24726,4.43026,4.18142,3.61345,5.09397,3.78883,3.9018,4.86902,2.63921,3.98461,2.85986,5.83503,4.10538,1.96192,3.17931,3.27886,2.26639,5.06245,0.82612,3.28779,3.28949,2.66292,2.65151,2.29328,0.98123,2.75834,1.39497,1.71248,3.26924,2.60496,2.16193,2.15818,3.43777,1.88958,1.79729,1.54856,1.81316,0.493695,1.88631,1.46588,1.23761,1.16607,1.90181,1.28348,1.27975,1.73192,0.658048,0.91322,2.15245,1.56231,1.07824,0.959417,1.41975,1.13177,0.871422,1.3561,1.08551,1.36557,1.42974,1.00859,0.889051,1.40908,0.669156,1.19596,1.74464,1.09084,0.629983,0.873186,1.38547,0.720821,0.57009,0.638891,0.54808,0.576935,1.05217,0.541056,0.814087,0.490098,0.676721,0.814272,0.583256,0.704583,0.224774,0.471837,0.497131,0.532516,0.627871,0.287407,0.602406,0.449572,0.075196,0.301999,0.47678,0.345899,0.00121423,0.48615,0.420499,0.687104,0.85461,0.360039,0.441451,0.206315,0.398659,0.00323525,0.147782,0.752741,0.459141,0.455333,0.44298,0.196357,0.296931,0.11279,0.0696214,0.43241,0.240833,0.24239,0.350807,0.328587,0.523582,0.415333,0.217098,0.00392699,0.29069,0.358463,0.173589,0.250808,0.199381,0.363037,0.0583139,0.22848,0.0021353,0.239984,0.128431,0.288657,0.12139,0.291582,0.0000251327,0.199186,0.140453,0.273344,0.165269,0.156516,0.198745,0.169354,0.287655,0.00640119,0.120826,0.160538,0.107156,0.0876152,0.133337,0.19657,0.18709,0.0482843,0.018428,0.130852,0.123106,0.000175635,0.130461,0.00262519,0.0976767,0.00384884,0.00457161,0.0341126,0.120898,0.0504916,0.037229,0.0459221,0.00498973,0.0592438,0.0306392,0.0545897,0.0291511,0.0513861,0.0137498,0.07949,0.028105,0.00300071,0.0888297,0.0280006,0.0158208,0.00779144,0.0335704,0.0374139,0.0222129,0.021627,0.0440153,0.0437421,0.0254723,0.02665,0.0241371,0.000186728,0.0135945,0.0226426,0.0194332,0.000494899,0.00204901,0.0145996,0.0132216,0.00531557,0.0138205,0.00683542,0.000788834,0.0133195,0.00337839,0.0103817,0.00623037,0.00553234,0.00405069,0.00515653,0.000990682,0.00719621,0.00386818,0.00515722,0.000946699,0.00149393,0.00432018,0.00368823,0.0043516,0.00437212,0.00100462,0.000867472,0.00126645,0.000171413,0.0000708822,0.00150777,0.0011241,0.000721977,0.000639314,0.000161399,0.00148283,0.0000384845,0.0000283725,0.000560382,0.000094346,0.000496764,0.000380624,0.0000731402,0.000194975,0.0000692132,0.000073042,0.0000319068,0.0000630282,9.81748e-6,0.0000237583}}; //Abundances(R,basal area)
  }
  else if(TASK.refDataType==2){
    if(data.System==35) abundances = {{314.085,681.248,422.732,73.5184,165.629,47.1286,1102.33,97.955,28.0194,84.124,193.59,140.668,90.5584,21.739,132.876,186.472,62.8191,242.644,128.656,25.6881,42.013,70.8999,154.206,85.9939,150.976,119.981,92.9865,92.7596,87.5623,25.6189,167.811,747.312,36.5478,95.049,23.5146,115.399,48.2939,47.1611,78.1698,31.3639,7.2775,24.7113,16.3011,88.0955,99.1521,13.2471,34.1048,18.9868,25.1278,295.289,25.3491,43.1,31.0408,7.85888,47.4259,24.6941,88.5726,105.79,12.0676,5.01725,14.6998,127.979,62.8478,13.8335,81.6208,6.997,14.4024,53.2613,31.9645,60.8683,60.2928,33.3425,15.5881,45.4329,69.9694,3.36738,19.0676,10.738,13.1178,37.0778,98.527,52.7904,90.5378,40.6698,21.1689,25.9519,8.34,33.3901,64.5679,4.10263,20.8026,21.7961,39.7039,28.5563,7.33413,13.9948,5.79113,130.299,12.9853,7.65413,39.042,76.0933,8.90713,21.124,31.9109,14.2139,11.2508,8.1365,11.3425,36.2816,19.1309,26.295,19.916,9.75825,34.0276,9.44738,6.67987,12.9671,38.2044,6.84213,11.2784,31.5383,4.2095,14.3643,13.1005,4.09825,51.1755,11.9121,5.8385,17.418,2.56463,34.7905,3.01887,6.60788,3.89925,12.323,9.13625,14.155,17.8341,27.3641,13.6159,5.555,8.30038,11.1815,2.10913,2.467,15.4645,23.0571,14.2994,70.0565,4.97875,5.33488,3.86475,6.83588,3.18763,10.0178,3.3565,1.62225,13.6204,4.42325,15.2534,5.9845,1.66263,13.7516,1.04863,8.59763,13.0914,24.5171,1.63888,5.362,6.988,16.0571,6.25063,8.03463,2.81688,4.568,1.99163,14.6081,5.34013,6.73275,1.5275,26.0845,3.37775,10.2274,6.74775,13.4175,5.207,8.11875,15.3201,2.96825,15.024,2.7455,2.8835,4.46388,1.97625,0.8945,9.51625,9.90063,11.5699,2.0665,0.65275,5.206,1.92088,1.15938,3.17763,2.60688,7.654,0.4945,1.99088,1.47813,21.3441,3.386,2.9515,3.715,0.68275,2.86675,1.54163,1.35488,0.98125,1.81288,0.356,1.16963,2.37238,1.0745,0.266,0.6295,0.60225,9.50663,0.243,13.811,1.7415,2.422,3.83587,0.811875,2.37213,0.336125,0.867125,1.7065,2.6815,1.38275,0.23575,0.742125,0.29775,0.50825,0.47075,4.8885,0.37725,0.2085,1.97975,0.169375,1.07538,2.95875,0.122375,0.42575,0.26075,1.4505,0.119875,1.8775,0.20775,0.21175,0.1905,0.1315,0.8955,0.32975,0.286125,0.720375,0.081375,0.147375,0.150875,0.257375,0.412625,0.3435,0.061125,0.317,0.1075,0.6915,0.224125,0.614,0.053625,0.049875,0.616375,0.217125,0.14375,0.050875,0.321875,0.460375,0.158625,0.037375,0.0435,0.08225,0.066875,0.18575,0.17275,0.048625,0.28775,0.04825,0.082,0.06325,0.165625,0.023875,0.057375,0.071375,0.12575,0.033375,0.02425,0.04775,0.021125,0.012375,0.015125,0.041375,0.011125,0.01275,0.01925,0.029375,0.0355,0.018875,0.005625,0.018625,0.008875,0.004875,0.011,0.010875,0.00375,0.00375,0.003125,0.003125,0.0025,0.00175}}; //Abundances(dbh)
    else if(data.System==36) abundances ={{113.7236,102.84413,75.665171,69.809214,61.44817,59.660238,61.571186,42.670445,34.272661,36.846062,33.589233,34.102403,29.549584,30.627651,31.708265,22.638826,27.055882,24.05385,25.908761,23.228896,26.964597,20.005035,23.283596,19.057735,19.599168,18.689202,17.37929,17.247765,17.101693,17.572731,18.434128,16.024854,19.064891,13.276464,12.061435,13.222863,12.968927,11.42871,12.15119,9.6884511,9.4647665,9.1064806,8.6059766,8.0150048,9.3032242,8.6032808,9.2602768,7.52439,7.8485076,7.1777733,6.4168118,7.1493189,7.0205209,8.4302193,6.8539548,7.2764151,7.617704,6.5859833,7.1132518,6.5917996,5.7749435,5.5030083,5.3709813,7.2570158,5.2621217,7.119555,5.4891535,6.5421425,5.5090781,5.029382,4.1365827,4.826004,5.7791752,4.5717128,4.5655485,4.2704806,5.0159772,4.5182577,4.1673214,4.8109387,4.9899013,3.9108365,3.3162675,3.6402888,3.3567728,2.9603405,3.1360744,3.6330728,2.9619508,3.5434159,2.6804345,2.5845441,3.2197002,2.8229416,3.9227803,3.2339481,2.8825662,2.4386237,2.4740873,2.6224623,2.4371772,2.1100253,1.9375818,2.5517201,1.9207835,1.9312759,2.1893741,2.0734143,2.6705296,2.0212612,1.9356097,2.0279059,1.6674546,1.8938712,1.6006709,1.5728239,1.6962599,1.5173065,1.5852019,1.4771985,1.6388584,1.3235576,1.2596402,1.3473111,1.2636786,1.0973699,1.1396644,1.3568452,1.2415808,1.1843449,1.1886279,1.1682448,1.0308567,0.94629732,0.97362442,0.96983087,1.0094436,1.1315896,0.89778989,0.94688953,1.0846605,1.0355738,1.2909709,0.97935025,1.1059276,1.268319,0.84506804,0.874217,0.84185498,0.78118954,0.88896035,0.82190837,0.9081611,0.76018102,0.73088916,0.78894971,0.69107468,0.81873148,0.61213601,0.70576834,0.58912976,0.83324576,0.58990279,0.69461433,0.62657843,0.69656595,0.9247471,0.67336621,0.76693389,0.69891184,0.63354397,0.59746433,0.55550906,0.58832099,0.52338745,0.42831201,0.55842667,0.55173474,0.58561558,0.47117804,0.48079481,0.52543733,0.44953886,0.51949669,0.5072619,0.40340552,0.438213,0.36920131,0.46227309,0.42149786,0.31168293,0.34547674,0.36171872,0.48731123,0.29263617,0.24922785,0.27055506,0.28295457,0.24776471,0.27848867,0.44258282,0.2322456,0.25095909,0.24751841,0.20138331,0.22202895,0.23324835,0.18898731,0.17336232,0.15612901,0.15736662,0.13964068,0.1398628,0.14970703,0.19054713,0.14669592,0.15662305,0.32489578,0.11559775,0.14103289,0.12149023,0.22888709,0.12808992,0.12805029,0.12576393,0.12893751,0.086038689,0.078056366,0.080810761,0.093142455,0.055808788,0.075589567,0.066945733,0.06963049,0.06574758,0.058856334,0.056464936,0.062686762,0.051210788,0.049070518,0.054301401,0.049009377,0.045349573,0.041484069,0.038113001,0.057566187,0.042115966,0.029976804,0.034508967,0.056351022,0.046359685,0.041424525,0.027007943,0.030517597,0.028328556,0.0304078,0.02372331,0.018760185,0.014542037,0.013008931,0.010063671,0.0138847,0.01074741,0.010479709,0.011257896,0.0100565,0.0086521398,0.01052785,0.015537807,0.010523352,0.0097163059,0.0073004355,0.0071441841,0.008719172,0.007593566,0.005976507,0.0067115687,0.0063279363,0.0064208589,0.010801272,0.0054742198,0.0059999598,0.0053277825,0.0043732551,0.0039072616,0.0046815259,0.0040447601,0.0049768212,0.0047157726,0.0031778022,0.0022147737,0.0047862726,0.0023205405,0.0028075426,0.0023945756,0.0021823349,0.0017343031,0.0018197818,0.0016197827,0.0015970571,0.0015934715,0.0011523415,0.00081652388,0.0011243974,0.00091228131,0.00054371518,0.00050964355,0.00086077996,0.00041983518,0.00034332552,0.0004274811,0.00034889751,0.00051933646,0.00017515578,0.00015234597,0.0001926158,0.00012802111,0.00011625428,0.00016471881,0.00010485779,0.0001276171,7.0802998e-05,5.0501425e-05,3.7876069e-05,2.7354939e-05,3.2825926e-05,1.6833808e-05,4.9491397e-05}};
    //{{108.264,95.6636,69.4551,65.7282,64.7453,59.801,57.447,38.538,32.7637,32.6038,30.6865,30.6149,30.2514,29.0531,28.3917,24.716,24.5837,22.5084,22.1812,21.0075,20.6994,19.063,18.9891,18.8366,17.8677,17.3764,17.1819,16.003,15.9839,15.7377,15.4185,14.8545,14.4454,12.9064,11.9753,11.8751,11.7475,11.6542,11.141,9.6389,9.22779,9.11152,8.86721,8.4358,8.41787,8.15615,7.76998,7.55104,7.20679,7.10189,7.02016,6.82511,6.78533,6.78309,6.67419,6.53859,6.37516,5.93724,5.73343,5.29822,5.28362,5.25794,5.21242,5.14354,5.03888,5.00584,4.9983,4.98207,4.8414,4.59367,4.53259,4.52961,4.45392,4.3653,4.35516,4.31003,4.27931,4.17629,3.91591,3.8515,3.74067,3.63053,3.57123,3.4576,3.20428,2.98716,2.81502,2.76613,2.72138,2.68919,2.67942,2.64024,2.63208,2.57392,2.52721,2.45539,2.44921,2.40747,2.32183,2.2365,2.22198,2.17436,1.9811,1.94558,1.8968,1.87446,1.8636,1.8607,1.83545,1.81887,1.79338,1.7684,1.64213,1.57063,1.55954,1.49835,1.49362,1.48012,1.45002,1.39881,1.39348,1.24619,1.18651,1.18174,1.1482,1.14074,1.10436,1.07182,1.05379,1.04842,1.02936,1.02639,1.02234,0.999691,0.987976,0.983836,0.951036,0.932567,0.919129,0.904446,0.900929,0.888214,0.888003,0.873009,0.867206,0.865598,0.848228,0.812021,0.791127,0.78311,0.753885,0.734336,0.721177,0.706719,0.690219,0.686827,0.681317,0.677643,0.670352,0.657501,0.654161,0.644768,0.641532,0.639583,0.634459,0.630774,0.627519,0.617834,0.604383,0.581188,0.567086,0.539837,0.538649,0.507875,0.504703,0.492676,0.490881,0.48893,0.471505,0.470008,0.464089,0.456211,0.437072,0.436805,0.436308,0.433933,0.394435,0.390235,0.376383,0.349397,0.33971,0.331973,0.326814,0.310889,0.29102,0.279549,0.267543,0.26569,0.258538,0.25782,0.250438,0.245324,0.241121,0.238102,0.216962,0.200457,0.193582,0.184965,0.173563,0.169146,0.153918,0.150181,0.147986,0.147612,0.140339,0.139469,0.138564,0.129646,0.126069,0.115947,0.115887,0.11122,0.10722,0.105272,0.102489,0.10147,0.0997554,0.0909062,0.0872181,0.0741011,0.0650954,0.0648325,0.0616257,0.0607205,0.0603798,0.0578446,0.0548913,0.0538167,0.0503088,0.0476965,0.0442567,0.0432185,0.0431278,0.0420957,0.0413012,0.0390465,0.0378877,0.0349649,0.0348826,0.0344849,0.0344374,0.0310122,0.0298983,0.0263972,0.0262775,0.025213,0.0213206,0.0184271,0.0168768,0.0149349,0.0117383,0.0108907,0.0104357,0.0103683,0.0102846,0.0102019,0.0100815,0.00951009,0.00950508,0.009327,0.00889944,0.008509,0.00833298,0.00817177,0.00728555,0.00696579,0.00668806,0.00648061,0.00587939,0.00576649,0.00576453,0.00558065,0.00484542,0.00454795,0.00453234,0.00432833,0.00421248,0.00413159,0.00392493,0.00370659,0.00256914,0.00242531,0.00225557,0.00214571,0.00211547,0.00199334,0.00178236,0.00177981,0.0016996,0.00156294,0.00155607,0.00112008,0.000952393,0.000871301,0.000801401,0.000627435,0.000594448,0.000510902,0.000485278,0.000400455,0.000393092,0.000342237,0.000302476,0.000204302,0.000177696,0.000177598,0.000149324,0.000112999,0.000110348,0.0000939533,0.000087081,0.0000688205,0.0000490874,0.0000441786,0.0000319068,0.0000319068,0.000019635,0.0000192423}}; //Abundances(bas) //{{108.264,95.6636,69.4551,65.7282,64.7453,59.801,57.447,38.538,32.7637,32.6038,30.2514,30.6149,30.6865,24.716,28.3917,29.0531,24.5837,22.5084,20.6994,18.8366,19.063,22.1812,21.0075,17.1819,17.8677,16.003,17.3764,18.9891,15.4185,14.4454,12.9064,15.7377,11.9753,15.9839,11.8751,11.6542,14.8545,11.7475,9.6389,9.11152,9.22779,8.86721,8.41787,11.141,7.10189,7.02016,7.55104,6.67419,7.20679,8.15615,6.53859,6.82511,7.76998,8.4358,6.78309,6.37516,6.78533,4.9983,4.8414,5.28362,5.14354,5.21242,5.03888,4.98207,5.93724,5.00584,4.53259,5.29822,4.52961,5.25794,4.31003,4.17629,4.59367,4.27931,4.35516,3.57123,4.3653,3.74067,3.91591,5.73343,3.63053,4.45392,3.4576,3.20428,3.8515,2.81502,2.72138,2.98716,2.67942,2.52721,2.63208,2.40747,2.76613,2.45539,2.68919,2.64024,1.87446,2.32183,2.44921,1.9811,2.2365,1.8607,1.83545,1.94558,1.81887,2.57392,1.57063,1.49835,2.17436,1.55954,1.7684,2.22198,1.8968,1.64213,1.45002,1.48012,1.49362,1.39348,1.39881,1.18174,1.1482,1.24619,0.888003,1.02234,1.18651,0.951036,1.05379,1.10436,0.987976,1.14074,0.983836,1.04842,0.865598,0.900929,0.919129,0.932567,1.02639,1.79338,0.848228,1.8636,0.888214,0.999691,0.812021,0.677643,0.681317,0.654161,0.706719,0.639583,0.630774,1.02936,0.641532,0.867206,0.634459,0.581188,0.657501,0.686827,0.78311,0.904446,0.539837,0.670352,0.690219,0.567086,0.492676,0.490881,0.437072,0.791127,0.604383,0.504703,0.538649,0.644768,1.07182,0.464089,0.617834,0.753885,0.433933,0.390235,0.627519,0.394435,0.436805,0.33971,0.456211,0.734336,0.873009,0.331973,0.349397,0.507875,0.29102,0.376383,0.310889,0.721177,0.326814,0.471505,0.279549,0.470008,0.250438,0.258538,0.267543,0.25782,0.26569,0.200457,0.184965,0.436308,0.147612,0.169146,0.241121,0.193582,0.238102,0.153918,0.216962,0.129646,0.48893,0.139469,0.147986,0.126069,0.11122,0.102489,0.173563,0.140339,0.245324,0.115887,0.0997554,0.0909062,0.10147,0.0616257,0.0650954,0.150181,0.0872181,0.115947,0.0548913,0.138564,0.0741011,0.0538167,0.10722,0.0420957,0.105272,0.0378877,0.0503088,0.0578446,0.0607205,0.0432185,0.0349649,0.0413012,0.0298983,0.0344849,0.0344374,0.0648325,0.0262775,0.0213206,0.0603798,0.0442567,0.0348826,0.0431278,0.0476965,0.0390465,0.0149349,0.025213,0.0104357,0.0310122,0.0168768,0.0263972,0.008509,0.0102019,0.0108907,0.00950508,0.0184271,0.00951009,0.00833298,0.00576453,0.00576649,0.0100815,0.0103683,0.00558065,0.0117383,0.0102846,0.00728555,0.009327,0.00587939,0.00668806,0.00370659,0.00696579,0.00648061,0.00889944,0.00453234,0.00256914,0.00817177,0.00484542,0.00454795,0.00225557,0.00156294,0.00242531,0.00392493,0.00211547,0.00432833,0.00112008,0.00413159,0.00421248,0.00199334,0.00214571,0.00177981,0.000594448,0.000627435,0.000871301,0.0016996,0.00178236,0.00155607,0.000510902,0.000400455,0.000204302,0.000952393,0.000801401,0.000112999,0.000177696,0.000393092,0.000302476,0.000342237,0.000485278,0.0000688205,0.000177598,0.000087081,0.000149324,0.000110348,0.0000939533,0.0000490874,0.0000441786,0.0000319068,0.0000319068,0.000019635,0.0000192423}}; //Abundances(bas)
  }  
  
  
  for(int s=0;s<data.S;s++) data.Abundances[s] = abundances[s];
  }

}

vector<size_t> sort_indices(vector<double> &v){//ascending order
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  // sort indices based on comparing values in v using std::stable_sort instead of std::sort to avoid unnecessary index re-orderings when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2){ return v[i1] < v[i2]; });
  return idx;
}

vector<size_t> sort_indices_by_copy(vector<double> v){//ascending order
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  // sort indices based on comparing values in v using std::stable_sort instead of std::sort to avoid unnecessary index re-orderings when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2){ return v[i1] < v[i2]; });
  return idx;
}


vector<size_t> sort_indices_reverse(vector<double> &v){//descending order
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  // sort indices based on comparing values in v using std::stable_sort instead of std::sort to avoid unnecessary index re-orderings when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2){ return v[i1] > v[i2]; });
  return idx;
}

double Overlap(vector<double> &p,vector<double> &d){
  if(Norm(p)<MP) return 0.;
  else{
    //discard outliers
    int newSize = d.size();
    if(TASK.Corr.Measure!=3) newSize = (int)((1.-TASK.Corr.PercentageOfOutliers)*(double)p.size()+0.1);
    vector<double> pp(newSize), dd(newSize);
    if(TASK.Corr.Measure!=3 && TASK.Corr.PercentageOfOutliers>MP){
      int count = 0;
      vector<double> diff(VecAbsDiff(p,d));
      for(auto i: sort_indices(diff)){
	if(count<newSize){ pp[count] = p[i]; dd[count] = d[i]; }
	else break;
	count++;
      }
    }
    else{ pp = p; dd = d; }
    
    if(TASK.Corr.Measure==1){//least-squares overlap
      return pow(xiOverlap(pp,dd),TASK.Corr.POW);
    }
    else if(TASK.Corr.Measure==2){//Pearson correlation (default: POW=1)
      double ppMean = VecAv(pp), ddMean = VecAv(dd), res = 0., ppVar = 0., ddVar = 0.;
      for(int i=0;i<newSize;i++){
	res += (pp[i]-ppMean)*(dd[i]-ddMean);
	ppVar += (pp[i]-ppMean)*(pp[i]-ppMean);
	ddVar += (dd[i]-ddMean)*(dd[i]-ddMean);
      }
      if(ppVar<MP || ddVar<MP) return 0.;
      else return pow(res/sqrt(ppVar*ddVar),TASK.Corr.POW);
    }
    else if(TASK.Corr.Measure==3){//LargeDataDeviants
      int newSize = (int)(TASK.Corr.PercentageOfOutliers*(double)d.size());//USERINPUT
      vector<double> pp(newSize), dd(newSize), sigma(newSize);
      double LDD = 0.;
      int count = 0;
      for(auto i: sort_indices_reverse(d)){
	if(count<newSize){ pp[count] = p[i]; dd[count] = d[i]; }
	else break;
	count++;
      }
      for(int i=0;i<newSize;i++){
	if(ABS(pp[i])<MP && ABS(dd[i])<MP) sigma[i] = 1.;
	else sigma[i] = ABS(min(pp[i],dd[i])/max(pp[i],dd[i]));
      }
      LDD += accumulate(sigma.begin(),sigma.end(),0.)/((double)newSize);
      count = 0;
      for(auto i: sort_indices_reverse(p)){
	if(count<newSize){ pp[count] = p[i]; dd[count] = d[i]; }
	else break;
	count++;
      }
      for(int i=0;i<newSize;i++){
	if(ABS(pp[i])<MP && ABS(dd[i])<MP) sigma[i] = 1.;
	else sigma[i] = ABS(min(pp[i],dd[i])/max(pp[i],dd[i]));
      }
      LDD += accumulate(sigma.begin(),sigma.end(),0.)/((double)newSize);      
      return pow(0.5*LDD,TASK.Corr.POW);
    }
    else return 0.;
    
  }
}

double fomOverlap(int s, datastruct &data){
  if(TASK.CoarseGrain && TASK.refDataType==2){
    int M = data.GridSize, col = 15+2*s+(data.System-35);
    vector<double> p(M), d(M);
    for(int m=0;m<M;m++){
      p[m] = data.Den[s][m];
      d[m] = TASK.FOMrefData[m][col]/data.Deltax2;//fraction of landcover
    }
    return Overlap(p,d);    
  }
  else{
    int sys=data.System-31;
    int M = TASK.mask.size(), col = 30+4*s+sys; if(TASK.refDataType==2) col = 15+2*s+(data.System-35);
    vector<double> p(M), d(M);
    for(int m=0;m<M;m++){
      p[m] = data.Den[s][TASK.mask[m]];
      d[m] = TASK.refData[TASK.mask[m]][col]/data.Deltax2;
    }
    return Overlap(p,d);
  }
}

double GetFigureOfMerit(datastruct &data){
  
  if(data.TaskType==44 || data.TaskType==444){//ForestGeo:
      
    vector<double> tmp(data.S); fill(tmp.begin(),tmp.end(),0.);
    #pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
    for(int s=0;s<data.S;s++){
      tmp[s] = 1.-fomOverlap(s,data);
      //optional:
      //tmp[s] *= sqrt(data.Abundances[data.S-1]/data.Abundances[s]);//put lowest weight on most abundant species
      //tmp[s] *= data.Abundances[data.S-1]/data.Abundances[s];//put lowest weight on most abundant species
      
      //variant 2:
/*      for(int i=0;i<M;i++) tmp[s] += ABS(p[i]-d[i]);
      tmp[s] /= (double)M*data.Abundances[s]/data.area;//normalize with average density
      tmp[s] *= sqrt(data.Abundances[data.S-1]/data.Abundances[s]);//put lowest weight on most abundant species    */   
    }
    double fom = 0.;
    //option 1: (weighted) average
    //sort(tmp.begin(),tmp.end()); for(int s=0;s<data.S;s++){ fom += tmp[s]/((double)data.S); /*fom += pow((double)(s+1),2.)*tmp[s]/((double)data.S);*/ }
    //option 2: largest deviant
    fom = *max_element(tmp.begin(),tmp.end());

    if(std::isfinite(fom)){
      if(ABS(fom-1.)<MP/* || data.SCcount>=data.maxSCcount*/) return accumulate(tmp.begin(),tmp.end(),0.0);
      else return fom;
    }
    else return 1.0e+300;     

  }
  else if(data.TaskType>=4 && data.TaskType<=7){
  
  //check ASYM of Den[0] with reference density
/*  double MeanRelDev = 0.;
  int sumcount = 0, s = 0;
  for(int i=0;i<data.TmpField1.size();i++){
    if(max(ABS(data.Den[s][i]),ABS(data.TmpField1[i]))>0.001*(data.DenMax[s]-data.DenMin[s])){
      MeanRelDev += ABS(ASYM(data.Den[s][i],data.TmpField1[i]));
      sumcount++; 
    }
  }  
  return MeanRelDev/((double)sumcount);*/ 

  //normalized scalar product of Den[0] and reference density
  int s = 0;
  return 1.-2.*VecMult(data.Den[s],data.ReferenceData, data)/(Norm2(data.Den[s])+Norm2(data.ReferenceData));
  
  //Kenkel dispersal-interaction-nutrient fit to uniform data, with harmonic resources
  
  }
  else return 0.;
}

vector<double> GetNetMagnetization(datastruct &data){//for two species
  vector<double> netmag;
  double res = 0.;
  
  double a2 = VecMult(data.Den[0],data.Den[0],data), b2 = VecMult(data.Den[1],data.Den[1],data);
  if(a2+b2>MP) res = 1.-2.*VecMult(data.Den[0],data.Den[1],data)/(a2+b2);
  netmag.push_back(res);
  netmag.push_back(sqrt(max(res,0.)));
  res = 0.;
  
  double localmagnetization = 0., maxlocalmagnetization = 0., den0, den1, threshold = 1.0e-3*min(data.DenMax[0]-data.DenMin[0],data.DenMax[1]-data.DenMin[1]);
  vector<double> VecAtmaxlocalmagnetization(data.DIM);
  int count = 0;
  for(int i=0;i<data.GridSize;i++){
    den0 = data.Den[0][i]; den1 = data.Den[1][i];
    if(den0>threshold && den1>threshold){
      localmagnetization = ABS((den0-den1)/(den0+den1));
      res += localmagnetization;
      if(localmagnetization>maxlocalmagnetization){ maxlocalmagnetization = localmagnetization; VecAtmaxlocalmagnetization = data.VecAt[i]; }
      count++;
    }
  }
  res /= (double)count;
  netmag.push_back(res);
  
  netmag.push_back(maxlocalmagnetization); 
  if(data.PrintSC) PRINT("maxlocalmagnetization = " + to_string(maxlocalmagnetization) + " at rVec = {" + vec_to_str(VecAtmaxlocalmagnetization) + "}",data);

  den0 = data.Den[0][data.CentreIndex]; den1 = data.Den[1][data.CentreIndex]; if(den0+den1>MP) netmag.push_back(ABS((den0-den1)/(den0+den1))); else netmag.push_back(0.);
  res = 0.;  
  
  for(int i=0;i<data.GridSize;i++) res += ABS(data.Den[0][i]-data.Den[1][i]);
  res *= pow(data.Deltax,data.DDIM)/(data.Abundances[0]+data.Abundances[1]);
  netmag.push_back(res); res = 0.;
  
  return netmag;
}

double GetColumnDensityMagnetization(datastruct &data){//for two species in 3D
	if(data.DIM!=3) return 0.;
	else{
		int n = data.EdgeLength, n2 = n*n;
		vector<vector<vector<double>>> ColDen2D(data.S);
		for(int s=0;s<data.S;s++) ColDen2D[s].resize(n);
		vector<double> Mag(n);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int i=0;i<n;i++){
			Mag[i] = 0.;
			for(int s=0;s<data.S;s++) ColDen2D[s][i].resize(n,0.);
			for(int j=0;j<n;j++){
				for(int k=0;k<n;k++){
					for(int s=0;s<data.S;s++) ColDen2D[s][i][j] += data.Deltax*data.Den[s][i*n2+j*n+k];//integrate over third direction -> proper 2D density
				}
				Mag[i] += ABS(ColDen2D[0][i][j] - ColDen2D[1][i][j]);
			}
		}
		return data.Deltax*data.Deltax*accumulate(Mag.begin(),Mag.end(),0.)/(data.Abundances[0]+data.Abundances[1]);
	}
}



std::string getExecutableDirectory(void){
    // Buffer to hold the path of the executable
    char buffer[PATH_MAX];
    
    // Read the symbolic link to get the path of the executable
    ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    
    if (length == -1) {
        // If readlink fails, return an empty string
		cout << "getExecutableDirectory failed" << endl;
        return "";
    }
    
    // Null-terminate the string
    buffer[length] = '\0';
    
    // Convert the path to a string
    std::string execPath(buffer);
    
    // Find the last '/' in the path to get the directory
    size_t lastSlash = execPath.rfind('/');
    
    if (lastSlash != std::string::npos) {
        // Return the directory part of the path
        return execPath.substr(0, lastSlash);
    }
    
    // If no '/' was found, return an empty string (should not happen)
    cout << "getExecutableDirectory failed" << endl;
    return "";
}

std::string findMatchingFile(const std::string& baseName, const std::string& extension){
    // Get the directory of the executable
    std::string directoryPath = getExecutableDirectory(), res = "";
    
    // Open the directory
    DIR* dir = opendir(directoryPath.c_str());
    if (!dir) {// Failed to open the directory, return empty string
		std::cout << "Failed to open the directory." << "\n";
        return "";
    }
    
    // Iterate through the files in the directory
    bool success = false;
    struct dirent* entry;
    while((entry = readdir(dir)) != nullptr){
        // Get the file name
        std::string fileName(entry->d_name);
		//cout << fileName << endl;
        // Check if the file name starts with baseName and ends with extension
        if (fileName.find(baseName) == 0 && fileName.rfind(extension) == fileName.length() - extension.length()) {
			if(success==true) cout << "findMatchingFile: Warning !!! multiple file name matches" << endl;
            // Return the full file name
            //res = directoryPath + "/" + fileName;
            res = fileName;
			success = true;
        }
    }
    
    // Close the directory
    closedir(dir);
	
	//return the complete file name
    if(res=="") std::cout << "No matching file found." << "\n";
    return res;
}

vector<vector<double>> readFile(const string &filename){
	vector<vector<double>> data;
    ifstream file(filename);

    if(file.is_open()){
		string line;
		while(getline(file,line)){
			istringstream iss(line);
			vector<double> row;
			double value;
			while(iss >> value) row.push_back(value);
			data.push_back(row);
		}
		file.close();
    }
    else cout << "Unable to open file " << filename << endl;

    return data;
}

void ExportTriangulation(datastruct &data){
	
	int precision = 8;//16;
	
	ofstream GnuplotFile;
	GnuplotFile.open("mpDPFT_AuxData_GnuplotKD.dat");
	GnuplotFile << setprecision(precision);
	
	if(data.KD.UseTriangulation && data.KD.UpdateTriangulation){//update the GoodTriangles file
		CleanUpTriangulation(data);
	
		string timestamp = YYYYMMDD() + "_" + hhmmss();
		data.KD.GoodTrianglesFilename = "TabFunc_K" + to_string(data.DIM) + "_" + to_string(data.KD.quality) + "_GoodTriangles_" + timestamp + ".dat";
	
		//Export triangles
		ofstream GoodTrianglesFile;
		GoodTrianglesFile.open(data.KD.GoodTrianglesFilename);
		GoodTrianglesFile << setprecision(precision);
		for(auto T: data.KD.GoodTriangles){
			for(auto point: T){
				for(auto coordinate: point){
					GoodTrianglesFile << coordinate << " ";
				}
			}
			GoodTrianglesFile << endl;
		}
		GoodTrianglesFile.close();
	
		for(auto T: data.KD.GoodTriangles){
			for(int p=0;p<3;p++) GnuplotFile << T[p][0] << " " << T[p][1] << "\n";
			GnuplotFile << T[0][0] << " " << T[0][1] << "\n" << "e\n\n";
		}
	}
	else{//replot the existing GoodTriangles file
		for(auto T: data.KD.GoodTriangles){
			for(int p=0;p<3;p++) GnuplotFile << T[p][0] << " " << T[p][1] << "\n";
			GnuplotFile << T[0][0] << " " << T[0][1] << "\n" << "e\n\n";
		}		
	}
	
	GnuplotFile.close();
}

void CleanUpTriangulation(datastruct &data){
	if(!data.KD.GoodTriangles.empty()){
		StartTimer("CleanUpTriangulation",data);
		
		vector<vector<vector<double>>> TmpCleanedUpTriangles1(0), TmpCleanedUpTriangles2;
		for(int t=0;t<data.KD.GoodTriangles.size();++t) TmpCleanedUpTriangles1.push_back(data.KD.GoodTriangles[t]);//copy of existing GoodTriangles
		
        double MergerRatio = 1.;
      	int m = 0;
		while(MergerRatio>data.KD.MergerRatioThreshold){//try to merge triangles into bigger triangles
			int T = TmpCleanedUpTriangles1.size();
			PRINT("CleanUpTriangulation: Merging#" + to_string(++m) + " ...",data);
			vector<vector<vector<double>>> MergedTriangles = vector<vector<vector<double>>>(T,vector<vector<double>>(3,vector<double>(3)));
            int tt = 1;
			vector<int> partner(T,-1);

            //determine mergers
            #pragma omp parallel for schedule(dynamic)
			for(int t1=0;t1<T;++t1){//go through all triangles t1
              	if(omp_get_thread_num()==0 && t1>tt){//report
                		PRINT("CleanUpTriangulation: Check triangle " + to_string(t1) + "/" + to_string(T),data);
                		if((double)tt<0.2*(double)T) tt *= 2;
                		else tt += (int)(0.2*(double)T);
             	}
				for(int t2=t1+1;t2<T;++t2){//search for a good partner
					if(GoodMergerQ(TmpCleanedUpTriangles1[t1], TmpCleanedUpTriangles1[t2], MergedTriangles[t1], data.DIM, data.InternalAcc, 1., data.KD.quality)){
						partner[t1] = t2;
						break;
					}
				}
            }

			for(int t1=0;t1<T;++t1) if(partner[t1]>=0) partner[partner[t1]] = t1;

			vector<int> toKeep(T,0);
          	#pragma omp parallel for schedule(dynamic)
            for(int t1=0;t1<T;++t1){
				if(partner[t1]<0){
					if(data.KD.ReevaluateTriangulation==0) toKeep[t1] = 1;
                    else if(data.KD.ReevaluateTriangulation>0 && GoodTriangleQ(data.DIM, TmpCleanedUpTriangles1[t1], data.InternalAcc, 1., data.KD.quality)) toKeep[t1] = 1;
				}
			}
			set<int> ToKeep;
			for(int t1=0;t1<T;++t1) if(toKeep[t1]) ToKeep.insert(t1);
			
			//assemble new list of triangles (TmpCleanedUpTriangles2)
			TmpCleanedUpTriangles2.clear();
			for(auto t : ToKeep) TmpCleanedUpTriangles2.push_back(TmpCleanedUpTriangles1[t]);//keep triangles without partner
			for(int t1=0;t1<T;++t1) if(partner[t1]>=0 && partner[t1]>t1) TmpCleanedUpTriangles2.push_back(MergedTriangles[t1]);//add mergers
			int C = TmpCleanedUpTriangles2.size(), mergers = T-C;
			PRINT("CleanUpTriangulation: " + to_string(2*mergers) + "/" + to_string(T) + " triangles removed (and " + to_string(mergers) + " pseudo-mergers added) -> Temporary number of clean triangles = " + to_string(C),data);
			MergerRatio = (double)mergers/(double)TmpCleanedUpTriangles1.size();

			if(MergerRatio>data.KD.MergerRatioThreshold){
				//remove a RemovalFraction of the smallest triangles
				double percentage = min(data.KD.RemovalFraction,1.-(1.+data.KD.RetainSurplusFraction)*(double)data.KD.OrigNumGoodTriangles/(double)TmpCleanedUpTriangles2.size());
				RemoveSmallTriangles(percentage, TmpCleanedUpTriangles2, data.KD.GoodTriangles, data);
			
				//prepare next iteration
				T = data.KD.GoodTriangles.size();
				percentage = 1.-(double)T / (double)TmpCleanedUpTriangles2.size();
				PRINT("CleanUpTriangulation: " + to_string((int)(100. * percentage)) + "% smallest triangles removed",data);
				TmpCleanedUpTriangles1.clear();
				for(int t=0;t<T;++t) TmpCleanedUpTriangles1.push_back(data.KD.GoodTriangles[t]);
			}
		}
		
		//remove smallest triangles such that GoodTriangles contains at most (1+data.KD.RetainSurplusFraction)*data.KD.OrigNumGoodTriangles triangles
		double percentage = 0.;
		if(data.KD.OrigNumGoodTriangles>0) percentage = max(0.,1. - (1.+data.KD.RetainSurplusFraction) * (double)data.KD.OrigNumGoodTriangles / (double)TmpCleanedUpTriangles2.size());
		RemoveSmallTriangles(percentage, TmpCleanedUpTriangles2, data.KD.GoodTriangles, data);
		PRINT("CleanUpTriangulation: " + to_string((int)(100. * percentage)) + "% smallest triangles removed",data);
		
		PRINT("-> Final (original) number of exported triangles = " + to_string(data.KD.GoodTriangles.size()) + " (" + to_string(data.KD.OrigNumGoodTriangles) + ")",data);
		sort(data.KD.GoodTriangles.begin(),data.KD.GoodTriangles.end());
		
		EndTimer("CleanUpTriangulation",data);
	}
}

void RemoveSmallTriangles(const double percentage, const vector<vector<vector<double>>> &InTriangles, vector<vector<vector<double>>> &OutTriangles, datastruct &data){//remove percentage of smallest triangles
	OutTriangles.clear(); OutTriangles.resize(0);
	int C = InTriangles.size();
	int c = (int)((1.-percentage)*(double)C);
	if(percentage>0. && c<C){
		vector<double> Areas(C);
		#pragma omp parallel for schedule(dynamic) if(data.ompThreads>1)
		for(int t=0;t<C;++t) Areas[t] = TriangleArea(InTriangles[t]);
		int count = 0;
		
		for(auto a: sort_indices_reverse(Areas)){
			if(count<c) OutTriangles.push_back(InTriangles[a]);
			else break;
			count++;
		}
	}
	else for(int t=0;t<InTriangles.size();++t) OutTriangles.push_back(InTriangles[t]);
}

bool ExtractIntegerFromPattern(string filename, int &out){
  	std::regex pattern("_(\\d+)_GoodTriangles");
  	std::smatch match;
  	if(std::regex_search(filename, match, pattern)){// Extract the first capture group, which corresponds to the number in parentheses
      	if (match.size() > 1) {
    		out = stoi(match[1].str());
			return true;
        }
        else{
          cout << "ExtractIntegerFromPattern: match failed" << endl;
          return false;
        }
  	}
  	else{
      cout << "ExtractIntegerFromPattern: regex_search failed" << endl;
      return false;
    }
}

vector<int> primeFactors(const int num, int printQ){//prime factors of n
  	int n = num;
    vector<int> tmpfacs(0);
  	if(n>1){
  		for(int i=2;i*i<=n;++i){
    		while(n%i==0){//i is a divisor of n
      			if(printQ==1) cout << " n = " << n << endl;
      			int test = n/i;
      			if(test>=1){
        			tmpfacs.push_back(i);
        			n = test;
      			}
            	else break;
    		}
  		}
  		if(n>1) tmpfacs.push_back(n);
    }
    else tmpfacs.push_back(1);
  	vector<int> primefactors(tmpfacs.size());
  	for(int i=0;i<tmpfacs.size();i++) primefactors[i] = tmpfacs[tmpfacs.size()-1-i];
  	//cout << IntVec_to_CommaSeparatedString(primefactors) << endl;
  	return primefactors;
}

vector<int> GetNestedThreadNumbers(const vector<int> primefactors, int printQ){
  if(primefactors.size()==1) return {primefactors[0],1};
  int a = primefactors[0];
  int b = primefactors[1];
  if(printQ==1) cout << "GetNestedThreadNumbers.. a=" << a << endl;
  if(printQ==1) cout << "GetNestedThreadNumbers.. b=" << b << endl;
  if(primefactors.size()>2){
    if(printQ==1) cout << "GetNestedThreadNumbers. primefactors.size()=" << primefactors.size() << endl;
    int i = 2;
    while(i<primefactors.size()){
      if(printQ==1) cout << "GetNestedThreadNumbers i=" << i << endl;
      while(b<a && i<primefactors.size()){
        b *= primefactors[i];
        if(printQ==1) cout << "GetNestedThreadNumbers b=" << b << endl;
        i++;
      }
      while(a<=b && i<primefactors.size()){
        a *= primefactors[i];
        if(printQ==1) cout << "GetNestedThreadNumbers a=" << a << endl;
        i++;
      }
    }
  }
  vector<int> res = {max(a,b),min(a,b)};
  if(printQ==1) cout << "GetNestedThreadNumbers " << IntVec_to_CommaSeparatedString(res) << endl;
  return res;
}
