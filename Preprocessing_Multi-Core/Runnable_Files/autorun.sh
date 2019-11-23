#!/bin/bash
cd ..
read -p "Enter File with experiment parameters, target vector and lattice basis please: " name
read -p "Enter number of parameters for the experiment please : " input
if [[ ! $input =~ ^[0-9]+$ ]] ; 
then
	if (($input == 0))
	then
    	echo "Please enter an integer number greater than zero, program will now exit"
    	exit
    fi
fi
g++ extract_experiment_data.cpp -o extract
./extract $input $name
exit