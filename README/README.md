# Dissertation
/ Compile using these flags: g++ -std=c++11 -O3 -march=native Babai.cpp -lfplll -lmpfr -lgmp  -o Babai
// Run with ./Babai
// Argument 1: "test" if you wish to test specific lattice, otherwise "0" for non-random, "1" for random
// Argument 2: dimension of lattice
// Argument 3: "0" for non-random vector (change values in lines 378-382), "1" for random, generated here