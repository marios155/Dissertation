#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <fplll.h>
#include <gmp.h>
#include <mpfr.h>

using namespace std;
using namespace fplll;

void parse_to_file (const char* filename, vector<string> &values)
{
    ofstream outFile(filename);
    for (const auto &value : values) outFile << values << "\n";
}

void delete_file_lines (const char *filename, int lines) 
{
	string deleteline;
	string line;
	ifstream fin;
	fin.open(filename);
	ofstream temp;
	temp.open("temp");
	for (int i = 0; i < lines; i++)
	{
		getline(fin, line);
	}
	int count = 0;
	while (fin)
	{
		getline(fin, line);
		if (line != "]]")
		{
			temp << line << endl;
		}
	}
	temp.close();
	fin.close();
	//remove(filename);
	//rename("temp",filename);

}

int delete_file_vector (const char* filename)
{
	string line;
	ifstream fin;
	fin.open(filename);
	ofstream temp;
	temp.open("lattice");
	getline(fin, line);
	while (line.find("[[") == -1)
	{
		getline(fin, line);
	}
	while (fin)
	{
		temp << line << endl;
		getline(fin, line);
	}
	fin.close();
	temp.close();
}

int read_parameters (const char *filename, int lines, vector<string> &parameters) 
{
	int status = 0;
	ifstream stream;
	string line;
	stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try  
	{
		stream.open(filename);
		for (int i = 0; i < lines; i++) 
		{
			getline(stream, line);
			parameters.push_back(line);

		}
		stream.close();
	}
	catch (const ifstream::failure&) 
	{
    	status = 1;
    	cerr << "Error by reading " << filename << "." << endl;
    	cout << stream.rdstate() << endl;
    }
    delete_file_lines(filename, lines);
    return status;

}



int read_vector (const char *filename, int lines, vector<string> &values) {
	int status = 0;
	ifstream stream;
	stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	string line;
	try  
	{
		stream.open(filename);
		cout << endl;
		getline(stream, line);
		while (line.find("[[") == -1)
		{
			cout << line << endl;
			cout << endl;
			values.push_back(line);
			getline(stream, line);

		}
		stream.close();
	}
	catch (const ifstream::failure&) 
	{
    	status = 1;
    	cerr << "Error by reading " << filename << "." << endl;
    	cout << stream.rdstate() << endl;
    }
    delete_file_vector ("temp");
    return status;
}


int main (int argc, char** argv)
{
	vector<string> parameters;
	vector<string> values;
	int status = 0;
	status |= read_parameters(argv[2], atoi(argv[1]), parameters);
	status |= read_vector ("temp", stoi(parameters[1]), values);
	if (status == 0)
	{
		cout << "File parsed correctly." << endl;
	}
	else
	{
		cout << "Parsing failure, see reading error occurence." << endl;
	}
	return 0;
}