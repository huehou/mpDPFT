#!/bin/bash

# Compilation and execution script for mpScript_testKD

# Directory containing mpScript_testKD
SCRIPT_DIR="/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts"

# Change to the script's directory
cd "$SCRIPT_DIR" || { echo "Failed to change to directory $SCRIPT_DIR"; exit 1; }

# Set the LD_LIBRARY_PATH for this session
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/lib/x86_64-linux-gnu/openmpi/bin

make
chmod +rwx mpScript_testKD
#./mpScript_testKD
# mpirun -np 4 ./mpScript_testKD
mpirun -np 1 ./mpScript_testKD
make clean

echo "mpScript_testKD.sh completed successfully. Press Ctrl+C to exit."
