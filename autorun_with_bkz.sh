#!/bin/bash
read -p "Enter lattice dimension please : " input
input_name="lattice"
randomness=0

if [[ ! $input =~ ^[0-9]+$ ]] ; 
then
	if (($input == 0))
	then
    	echo "Please enter an integer number greater than zero, program will now exit"
    	exit
    fi
fi
g++ -std=c++11 -march=native -o3 BKZ_reduction.cpp -lfplll -lmpfr -lgmp -o reduction
./reduction $input $randomness
nvcc CUDA_Lindner.cu -lfplll -lmpfr -lgmp -o cuda
./cuda $input
exit
