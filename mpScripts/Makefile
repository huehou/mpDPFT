# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2 -I/usr/lib/x86_64-linux-gnu/openmpi/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

# Target executable
TARGET = mpScript_testKD

# Source files
SRC = mpScript_testKD.cpp

# Default target: Compile only
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)
