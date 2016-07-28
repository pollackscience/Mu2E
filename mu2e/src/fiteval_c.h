#ifndef FITEVAL_C_H
#define FITEVAL_C_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <gsl/gsl_sf_bessel.h>
#include "csv.h"

using namespace std;

class FitFunctionMaker
{
    public: 
        FitFunctionMaker(string fit_csv);
        vector<double> get_field(double a, double b, double z, bool cart);

    private:
        int ns;
        int ms;
        double Reff;
        vector<vector<double> > As;
        vector<vector<double> > Bs;
        vector<double> Ds;
        vector<vector<double> > kms;
        vector<vector<double> > iv;
        vector<vector<double> > ivp;
};

vector<string> &split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);


#endif
