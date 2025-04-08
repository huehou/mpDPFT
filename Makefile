NUM_CORES := $(shell nproc)

CXX = g++
CXXSTD = -std=c++20
CXXFLAGS = -I/opt/etsf/include -I${CURDIR}/Eigen_Headers -fopenmp -D_GLIBCXX_PARALLEL -mtune=native -march=native -funroll-loops -fconcepts -Wno-volatile
#for NSCC aspire2
#CXXFLAGS = -I/home/users/nus/cqtmst/scratch/gsl/include -I/home/users/nus/cqtmst/scratch/libxc/include -I/home/users/nus/cqtmst/scratch/fftw/include -I/opt/etsf/include -I${CURDIR}/Eigen_Headers -fopenmp -D_GLIBCXX_PARALLEL -mtune=native -march=native -funroll-loops -fconcepts -Wno-volatile
CXXOPTFLAGS = -flto=$(shell nproc) -O3

# Check if g++ with -std=c++20 support is available
GPP_VERSION := $(shell g++ --version | awk '/g\+\+/ {print $$NF}' | cut -d. -f1)
MIN_GPP_VERSION = 10

ifeq ($(shell test $(GPP_VERSION) -ge $(MIN_GPP_VERSION) && echo "yes"),yes)
    $(info g++ with -std=c++20 support is available (version $(GPP_VERSION)))
else
    $(info g++ with -std=c++20 support is not available or version is too low (version $(GPP_VERSION)))
    $(info Switching to an older standard (-std=c++17))
    CXXSTD = -std=c++17
endif

#################### BEGIN USER INPUT ####################

#default:
CC = $(CXX) $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS)
#for gdb & valgrind:
#CC = $(CXX) -ggdb3 $(CXXSTD) $(CXXFLAGS) -O2 -fno-inline -fno-omit-frame-pointer
#with sanitation:
#CC = $(CXX) -fsanitize=address -fsanitize=undefined -ggdb3 $(CXXSTD) $(CXXFLAGS) -O0 -fno-inline -fno-omit-frame-pointer
#with production code optimization:
#CC = $(CXX) -ggdb3 $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS)
#for using cSA:
#CC = $(CXX) $(CXXSTD) $(CXXFLAGS) $(CXXOPTFLAGS) -pthread

#default:
LDFLAGS= -L${EBROOTGSL} -L/usr -L/usr/local/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops
#for NSCC aspire2
#LDFLAGS= -L${EBROOTGSL} -L/usr -L/home/users/nus/cqtmst/scratch/gsl/lib -L/home/users/nus/cqtmst/scratch/libxc/lib -L/home/users/nus/cqtmst/scratch/fftw/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops
#for using cSA:
#LDFLAGS= -L${EBROOTGSL} -L/usr -L/usr/local/lib -L/opt/etsf -L/opt/etsf/lib -L/usr/include -lgsl -lgslcblas -lmpfr -lgmp -lfftw3_omp -lfftw3 -lm -lxc -funroll-loops -pthread

#################### END USER INPUT ####################

OBJS = mpDPFTmain.o mpDPFT.o Plugin_KD.o Plugin_Triangulation.o Plugin_1pEx.o Plugin_1pEx_Rho1p.o Plugin_OPT.o Plugin_cec14_test_func.o statistics.o specialfunctions.o solvers.o optimization.o linalg.o kernels_sse2.o kernels_fma.o kernels_avx2.o interpolation.o integration.o fasttransforms.o diffequations.o dataanalysis.o ap.o alglibmisc.o alglibinternal.o
HEADERS = stdafx.h statistics.h specialfunctions.h solvers.h optimization.h linalg.h kernels_sse2.h kernels_fma.h kernels_avx2.h interpolation.h integration.h fasttransforms.h diffequations.h dataanalysis.h ap.h alglibmisc.h alglibinternal.h

.PHONY: all clean

all: mpDPFT

mpDPFT: $(OBJS)
	@echo "Linking mpDPFT..."
	@time $(CC) $(OBJS) -o $@ $(LDFLAGS)

# %.o: %.cpp $(HEADERS)
# 	$(CC) -c $< -o $@
# Auto-generate dependencies
DEPFILES = $(OBJS:.o=.d)

-include $(DEPFILES)

%.o: %.cpp
	$(CC) -MMD -MP -c $< -o $@

# clean:
# 	rm -f mpDPFT

