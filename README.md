# mpDPFT (multi-purpose Density-Potential Functional Theory)
**Tested on Ubuntu LTS 16, 18, 20, 22, & 24**

This semi-automatic installation (with occasional confirmations in the terminal) should take less than 30 minutes. Make sure that the items mentioned below are all in place (eventually) by simply executing this README.md as explained in the INSTALLATION section below.


---


## OVERVIEW

mpDPFT is a C++ codebase for DPFT simulations of fermionic quantum systems—from quantum gases to electronic structure. The main code executes the selfconsistent DPFT loop for a broad variety of interaction functionals (similar to traditional Kohn--Sham implementations, though with various orbital-free kinetic energy potential functionals). Notable numerical features are multi-species Pulay mixing, FFT derivatives stabilized with convolutions, self-consistent loop implementation with annealing; linked libraries include fftw, alglib, boost, gsl, libxc. The code relies on openMP for parallelization. Some features use MPI and others interact with Python scripts.

**Integrated side projects:**
--- Black-box function optimization via evolutionary algorithms (CMA-ES, particle swarm optimition, genetic algorithm, simulated annealing)
--- One-particle-exact density functional theory (1pEx-DFT)
--- Ecosystem modelling (DFTe)

**Notable Applications:**
--- Trapped ultracold Fermi gases (1D, 2D, 3D, and 2D-to-3D crossover)
--- Electron-hole puddles in 2D materials
--- Electronic structure from atoms to nanoparticles

**Lead Developer:** Martin-Isbjörn Trappe

**Contributors:** Jun Hao Hue, Thanh Tri Chau, Jonathan Wei Zhong Lau, Mikołaj Paraniak, Michael Tsesmelis

**Related Publications:** The following publications contain material produced with mpDPFT (or its predecessors) and document the development of mpDPFT
--- [Unsupervised state learning from pairs of states](https://arxiv.org/abs/2007.05308)
--- [Phase separation of a repulsive two-component Fermi gas](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.023325)
--- [Atoms, dimers, and nanoparticles from orbital-free DPFT](https://link.aps.org/doi/10.1103/PhysRevA.108.062802)
--- [Single-particle-exact density functional theory](https://doi.org/10.1016/j.aop.2023.169497)
--- [A density functional theory for ecology across scales](https://doi.org/10.1038/s41467-023-36628-4)
--- [Density-potential functional theory for fermions in one dimension](https://doi.org/10.1142/9789811272158_fmatter)
--- [Phase Transitions of Repulsive Two-Component Fermi Gases in Two Dimensions](https://doi.org/10.1088/1367-2630/ac2b51)
--- [First-principles quantum corrections for carrier correlations in double-layer two-dimensional heterostructures](https://doi.org/10.1103/PhysRevB.99.235415)
--- [Systematic corrections to the Thomas--Fermi approximation without a gradient expansion](https://doi.org/10.1088/1367-2630/aacde1)


---


## PREREQUISITES

Before installation, note these additional instructions (for post-installation):

- **Required Tools for Post-Processing:**
  - `texlive-full`
  - `gnuplot`
  - `ffmpeg`
  - `ps2eps`
  - `okular`

- **Post-Installation Steps:**
  - Restart your computer.
  - Install the following tools for post-processing: `texlive-full`, `gnuplot`, `ffmpeg`, `ps2eps`, `okular`
  - Copy all mpDPFT files and folders into your chosen installation directory (`YOURDIRECTORY`).
  - In `YOURDIRECTORY/mpDPFT.sh`, replace all instances of `/home/martintrappe/Desktop/PostDoc/Code/mpDPFT` with the full path: `$CurrentWorkingDirectory/$YOURDIRECTORY`.
  - Uncomment the line after `#To recompile ALGLIB` in `mpDPFT.sh` (usually needed only for the first compilation).
  - Optionally, copy `mpDPFT.sh` to your default terminal folder for easy access.
  - Copy ALGLIB source files (https://www.alglib.net/download.php) into YOURDIRECTORY/
  - Copy placeins.sty (http://mirrors.ctan.org/macros/latex/contrib/placeins/placeins.sty) into YOURDIRECTORY/
  - Copy epslatex2epspdf into the folder that contains the ps2eps executable (likely, /usr/bin/): `sudo cp YOURDIRECTORY/epslatex2epspdf /usr/bin/ && sudo chown YourUbuntuUserName:YourUbuntuUserName /usr/bin/epslatex2epspdf && sudo chmod 755 /usr/bin/epslatex2epspdf`

- **Notes for Execution**
  - run mpDPFT, optionally specify X (number_of_threads) for parallelization and, also optionally, Y (after specifying X) for running several independent jobs in the sub-folders YOURDIRECTORY/runY: `./mpDPFT.sh X Y`
  - All output will be stored the folder YOURDIRECTORY/#DATA/#zips/ (as created during installation)
  - The output is summarized in mpDPFT_CombinedPlots.pdf (generated from mpDPFT_CombinedPlots.tex), the main log files are mpDPFT_Control.dat and mpDPFT_ControlTask.dat, the main data files that are generated are mpDPFT_CutData.dat, mpDPFT_ContourData.dat, mpDPFT_Den.dat, and mpDPFT_V.dat.
  - Examples with all source files and all input files are provided in /#DATA/#zips/ (including mpDPFT.input, which specifies the main input parameters - other input parameters are hard-coded and marked by //BEGIN USER INPUT in mpDPFT.cpp and elsewhere)
  - For controlled termination of selfconsistent DPFT loop [or task] during runtime: Replace 0 by 1 [or by 2] in mpDPFTmanualSCloopBreakQ.dat (or in mpDPFTmanualOPTloopBreakQ.dat)


---


## INSTALLATION INSTRUCTIONS

### Step 1: Set Up Your Working Directory

  - Place this `README.md` into a folder of your choice. This folder becomes your current working directory (`$(pwd)`). Then, **Change to this directory** in the terminal:
   ```bash
   cd /path/to/your/chosen/folder
   ```

### Step 2: Install

  - Option 1: Manual installation. Follow the commands line by line in the section BASH INSTALLER below.

  - Option 2: Semi-automatic installation. Paste and execute the following command (for changing shell to bash, creating mpDPFT installer, and executing the installer) into the terminal [[from here onward, the installer (install_mpDPFT.sh) will guide you through the installation; some alternative options for installation are commented out in the BASH INSTALLER section below - simply adjust as you see fit. You will be asked for a destination directory (YOURDIRECTORY) of your choice (e.g., 'mpDPFT' or 'Desktop/mpDPFT'), which is relative to the path of this README.md, i.e., relative to $(pwd)]]:
  `sudo chsh -s /bin/bash && cp README.md install_mpDPFT.sh && tail -n +100 "install_mpDPFT.sh" > "install_mpDPFT.tmp" && mv "install_mpDPFT.tmp" "install_mpDPFT.sh" && chmod +rwx install_mpDPFT.sh && ./install_mpDPFT.sh`


---




## BASH INSTALLER

```bash
read -p "Enter folder name for installation (into the path relative to this README.md, e.g., 'mpDPFT' or 'Desktop/mpDPFT'): " YOURDIRECTORY

CurrentWorkingDirectory=$(pwd)
mkdir -p $CurrentWorkingDirectory/$YOURDIRECTORY
mkdir -p $CurrentWorkingDirectory/$YOURDIRECTORY/#DATA/
mkdir -p $CurrentWorkingDirectory/$YOURDIRECTORY/#DATA/#zips/
mkdir -p $CurrentWorkingDirectory/$YOURDIRECTORY/#Source_Backups/

#install zip/unzip utilities:
sudo apt install zip
sudo apt install unzip

#install g++ compiler:
sudo apt update && sudo apt upgrade
sudo apt install build-essential

#install fortran 90 compiler:
sudo apt-get update
sudo apt-get install gfortran

#install gsl:
wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz
chmod +rwx *
tar -zxvf gsl-2.8.tar.gz
cd gsl-2.8/
./configure
#alternatively:
#./configure --prefix=/home/users/nus/cqtmst/scratch/gsl
make
make check
sudo make install
cd
rm gsl-2.8.tar.gz
LD_LIBRARY_PATH=/usr/local/lib
#LD_LIBRARY_PATH=/home/users/nus/cqtmst/scratch/gsl
export LD_LIBRARY_PATH

#install boost  (1.77 or higher):
#[if need to build from scratch:
#wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
sudo chmod +rwx boost_1_86_0.tar.gz
tar -xvzf boost_1_86_0.tar.gz
cd boost_1_86_0/
sudo ./bootstrap.sh
#alternatively:
#./bootstrap.sh --prefix=/home/users/nus/cqtmst/scratch/boost
sudo ./b2 install
#]
#[else:
sudo apt install libboost-all-dev
#]
sudo apt install aptitude
sudo apt-get install libgmp-dev
#alternatively:
#./configure --prefix=/home/users/nus/cqtmst/scratch/gmp
#make
#make install
sudo apt-get install libmpfr-dev
#alternatively:
#./configure --prefix=/home/users/nus/cqtmst/scratch/mpfr --with-gmp=/home/users/nus/cqtmst/scratch/gmp
#make
#make install

#install FFTW:
wget ftp://ftp.fftw.org/pub/fftw/fftw-3.3.8.tar.gz
chmod +rwx *
tar -zxvf fftw-3.3.8.tar.gz
cd fftw-3.3.8/
./configure --enable-openmp
#alternatively:
#./configure --prefix=/home/users/nus/cqtmst/scratch/fftw --enable-openmp
make
sudo make install
cd
rm fftw-3.3.8.tar.gz

#install libxc into /opt/etsf/lib:
wget https://gitlab.com/libxc/libxc/-/archive/2.1.2/libxc-2.1.2.tar.bz2
chmod +rwx *.*
tar -xf libxc-2.1.2.tar.bz2
cd libxc-2.1.2/
chmod +rwx *.*
autoreconf -i
sudo ./configure
#alternatively for cray compiler:
#./configure CC=cc CXX=g++ --prefix=/home/users/nus/cqtmst/scratch/libxc --enable-fortran
sudo make
make check
sudo make install
cd
rm libxc-2.1.2.tar.bz2

rm install_mpDPFT.sh
```
