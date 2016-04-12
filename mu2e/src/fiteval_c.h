#ifndef FITEVAL_C_H
#define FITEVAL_C_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "csv.h"

using namespace std;

class FitFunctionMaker
{
    public: 
        FitFunctionMaker(string fit_csv);
    private:
        int ns;
        int ms;
        double Reff;
        vector<vector<double> > As;
        vector<vector<double> > Bs;
        vector<double> Cs;
        vector<double> Ds;
};

vector<string> &split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);


#endif
