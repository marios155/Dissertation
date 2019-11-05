#!/bin/bash
read -p "Enter lattice dimension please : " input
name="lattice"

if [[ ! $input =~ ^[0-9]+$ ]] ; 
then
	if (($input == 0))
	then
    	echo "Please enter an integer number greater than zero, program will now exit"
    	exit
    fi
fi
nvcc CUDA_Lindner.cu -lfplll -lmpfr -lgmp -o cuda
./cuda $input
exit
