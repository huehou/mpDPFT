#include "mpScript_testKD.h"
#include <iostream>
#include <cstdlib> // For system()
#include <mpi.h>   // OpenMPI header

void runMPIExample(int argc, char** argv) {
	MPI_Init(&argc, &argv);  // Now using passed arguments

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::cout << "Hello from rank " << rank << std::endl;

	MPI_Finalize();
}

int main(int argc, char** argv) {  // Pass argc and argv to main
	std::cout << "mpScript_testKD project initialized." << std::endl;

	// Call a function from mpScript_testKD.h
	sayHello();

	std::cout << "Calling Python script..." << std::endl;
	const char* script_path = "/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/mpScripts/mpScript_testKD.py";
	std::string command = "/usr/bin/python3 " + std::string(script_path) + " &";  // Adjust the Python path as needed

	int ret_code = system(command.c_str());
	//int ret_code = system("python3 mpScript_testKD.py &");
	if (ret_code != 0) {
		std::cerr << "Failed to execute Python script!" << std::endl;
	}

	// Run MPI example
	std::cout << "Running MPI example..." << std::endl;
	runMPIExample(argc, argv);  // Pass argc and argv to the function

	return 0;
}

