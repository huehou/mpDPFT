#!/bin/bash
#sysbench cpu --cpu-max-prime=20000 --threads=$(nproc) run (check "total number of events" --- CQT_laptop->158189)
#execute in command line, see README.md
threads=${1:-$(nproc)}
job=$2
cp *.sh /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/
rm *.pdf
rm *.eps
rm -r epslatex2epspdf_tmp
rm tmp_split.*
rm *.backup
rm *.new
rm *.*.new
rm *Movie*
rm mpDPFT_OPLenergies.dat
rm mpDPFT_Den_Cube.dat
chmod +rwx *.*
cp /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpDPFT.input /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpDPFT.tmpinput
cp /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpDPFT.input /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpDPFT.originput
FILE=/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
if [ -d "$FILE" ];
    then echo "$FILE exists"
    else mkdir run$job
fi
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/
rm TabFunc_X2C_*.dat
rm TabFunc_QuadraticProgram*.dat
rm TabFunc_K*GoodTriangles*.dat
rm TabFunc_NYFunction*.dat
rm TabFunc_Nuclei*.dat
rm mpDPFT_Den_Cube.dat
rm mpDPFT_AuxMat*.dat
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/
#To recompile ALGLIB and kernel files:
#cp epslatex2epspdf Makefile *.* /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
#To use precompiled ALGLIB:
cp *.*input /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
cp -up mpDPFTmain.cpp mpDPFT.cpp mpDPFT.h MPDPFT_HEADER_*.h *.hpp Plugin*.* Makefile README.md *.sh *.tex epslatex2epspdf *.info *.dat /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
cp -up -r /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/Eigen_Headers /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
cp -up -r /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/CEC2014_input_data /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
cp -up -r /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/
rm *.pdf
rm *.eps
rm -r epslatex2epspdf_tmp
rm tmp_split.*
rm *.backup
rm *.new
rm *.*.new
rm *Movie*
rm mpDPFT_DynDFTe_*.dat
rm mpDPFT_OPLenergies.dat
rm mpDPFT_RBF_*.dat
rm mpDPFT_1pExDFT_MonitorMatrix_*.dat
rm mpDPFT_ObjFunc*.*
rm mpDPFT_TabFunc_NYFunction*.*
rm mpDPFT_testK*.*
rm /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/TabFunc_Hint.dat
mv /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/TabFunc_Hint*.dat /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/TabFunc_Hint.dat
VInterpolIdentifier="mpDPFT_V_*.dat" && VInterpolIdentifier=$(echo $VInterpolIdentifier| cut -c 10-24) && if [[ ${#VInterpolIdentifier} -lt 15 ]]; then VInterpolIdentifier="?"; fi && echo "$VInterpolIdentifier" > mpDPFT_Aux.dat && mv mpDPFT_V_*.dat mpDPFT_V.dat
make -j$(nproc)
export OMP_NUM_THREADS=$threads
export OMP_THREAD_LIMIT=$threads
# export OMP_MAX_ACTIVE_LEVELS=$threads
# export OMP_NESTED=true
# export SUNW_MP_MAX_POOL_THREADS=$threads-1
# export SUNW_MP_MAX_NESTED_LEVELS=2
# export SUNW_MP_MAX_ACTIVE_LEVELS=2
#export OMP_SCHEDULE=OMP_SCHED_STATIC
#export OMP_SCHEDULE=omp_sched_dynamic
#export OMP_SCHEDULE=OMP_SCHED_GUIDED
#export OMP_SCHEDULE=OMP_SCHED_AUTO
nice -19 ./mpDPFT #option 1; default
#gdb mpDPFT #option 2; for debugging. Select in MakeFile: CC= g++ -g ...; type 'run' when in (gdb) terminal, type 'bt' or 'thread apply all bt' for back-tracing; when stuck -> ctrl-c -> bt -> cont
#valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./mpDPFT
#valgrind --tool=massif --massif-out-file=massif.out ./mpDPFT
#make clean
chmod +rwx *.sh
chmod +rwx *.* *
FILE=/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/mpDPFT_MovieData.tmp
if test -f "$FILE"; then
    mkdir Movie
    cp mpDPFT_MovieData.tmp mpDPFT_MovieData.dat
    cp mpDPFT_Movie.sh mpDPFT_MovieData.dat /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/Movie/
    cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/Movie/
    ./mpDPFT_Movie.sh
    cp mpDPFT_Movie.mp4 /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/
    cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/
    rm -R /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/Movie/
fi
FILE2=/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/mpDPFT_OPLenergies.dat
if test -f "$FILE2"; then
    ./mpDPFT_OPLplots.sh
fi
./mpDPFT_Plots.sh
rm *.eps
chmod +rwx mpDPFT_CombinedPlots.tex
pdflatex mpDPFT_CombinedPlots.tex
rm mpDPFT_CombinedPlots.log
rm mpDPFT_CombinedPlots.aux
rm texput.log
read -r VInterpolIdentifier < "mpDPFT_Aux.dat"
echo "VInterpolIdentifier=$VInterpolIdentifier" && rm mpDPFT_Aux.dat && TimeStamp="$(date +%Y%m%d_%H%M%S)" && DirectoryName="mpDPFT_$TimeStamp-$VInterpolIdentifier"
mkdir /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/#zips/$DirectoryName/
cp -r *.cpp *.h *.hpp *.*input Makefile README.md *.sh *.tex epslatex2epspdf *.info *.dat *.pdf *.mp4 *.sty Eigen_Headers CEC2014_input_data /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/#zips/$DirectoryName/
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/#zips/$DirectoryName/
mv mpDPFT.originput mpDPFT.input && mv mpDPFT_V.dat mpDPFT_V_$TimeStamp.dat
zip -r mpDPFT_SOURCE_$TimeStamp-$VInterpolIdentifier.zip *.cpp *.h *.hpp *.input *.sty TabFunc*.dat epslatex2epspdf Makefile README.md mpDPFT.sh mpDPFTmanualOPTloopBreakQ.dat mpDPFTmanualSCloopBreakQ.dat Eigen_Headers CEC2014_input_data && chmod +rwx *.zip
cp mpDPFT_SOURCE_*.zip /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#Source_Backups
rm *.zip
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run$job/
if test -f "$FILE"; then
    rm mpDPFT_MovieData.tmp
    #okular mpDPFT_CombinedPlots.pdf & xdg-open mpDPFT_Movie.mp4
    #okular mpDPFT_CombinedPlots.pdf & vlc mpDPFT_Movie.mp4
fi
okular mpDPFT_CombinedPlots.pdf
