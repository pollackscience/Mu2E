#include "fiteval_c.h"
vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

FitFunctionMaker::FitFunctionMaker(string fit_csv):
    ns(-1),ms(-1),Reff(-1),As(),Bs(),Cs(),Ds()
{
    // Read in csv, tokenize the parameter name, start filling
    // the private class members with the correct values.
    // This assumes the csv is ordered correctly, so I should have some assertions
    // that raise exceptions when ordering is incorrect.  Read-time is not critical,
    // as this is only read from file once, then stored in memory.
	io::CSVReader<2> in(fit_csv);
	in.set_header("param","val");
	string param; double val;
    // Grab the first 3 vals
	while(in.read_row(param, val)){
		vector<string> tparams = split(param,'_');
        if (tparams[0].compare("R")==0){
            Reff = val;
        }else if(tparams[0].compare("ns")==0){
            ns = val;
        }else if(tparams[0].compare("ms")==0){
            ms = val;
        }
        if(!(ns==-1 || ms==-1 || Reff==-1)) break;
	}
    // Ready the 2D arrays
    for(int i=0; i<ns; ++i){
        As.push_back(vector<double>());
        Bs.push_back(vector<double>());
    }
    // Fill the 2D params
	while(in.read_row(param, val)){
		vector<string> tparams = split(param,'_');
        if (tparams[0].compare("A")==0){
            As[stoi(tparams[1])].push_back(val);
        }else if (tparams[0].compare("B")==0){
            Bs[stoi(tparams[1])].push_back(val);
        }else if (tparams[0].compare("C")==0){
            Cs.push_back(val);
        }else if (tparams[0].compare("D")==0){
            Ds.push_back(val);
        }
    }
    // Calculate the zeros of the bessel functions
    for(int n=0; n<ns; ++n){
        kms.push_back(vector<double>());
        for(int m=1; m<=ms; ++m){
            kms[n].push_back(gsl_sf_bessel_zero_Jnu(n,m)/Reff);
        }
    }
    for (int n=0; n<ns; ++n){
        vector<double> tmp_v;
        tmp_v.reserve(ms);
        iv.push_back(tmp_v);
        ivp.push_back(tmp_v);
    }


   cout<<Bs[0][0]<<endl;
   cout<<Cs[0]<<endl;
   cout<<Reff<<ns<<ms<<endl;
}

vector<double> FitFunctionMaker::mag_field_function(double a, double b, double z, bool cart=true){
    double r,phi;
    vector<double> out(3,0);
    if (cart){
        if (a==0) a+=1e-8;
        if (b==0) b+=1e-8;
        r = sqrt(pow(a,2)+pow(b,2));
        phi = atan2(b,a);
    }else{
        r = a;
        phi = b;
    }
    double abs_r = abs(r);

    for (int n=0;n<ns;++n){
        for (int m=1; m<=ms; ++m){
            double tmp_rho = kms[n][m-1]*abs_r;
            iv[n].push_back(gsl_sf_bessel_In(n,tmp_rho));
            ivp[n].push_back((n/tmp_rho)*iv[n][m-1]+gsl_sf_bessel_In(n+1,tmp_rho));
        }
    }

    double br(0.0);
    double bphi(0.0);
    double bz(0.0);
    for (int n =0; n<ns; ++n){
        for (int m =0; m<ms; ++m){
            br += (Cs[n]*cos(n*phi)+Ds[n]*sin(n*phi))*ivp[n][m]*kms[n][m]*(As[n][m]*cos(kms[n][m]*z) + Bs[n][m]*sin(-kms[n][m]*z));
            bphi += n*(-Cs[n]*sin(n*phi)+Ds[n]*cos(n*phi))*(1/abs_r)*iv[n][m]*(As[n][m]*cos(kms[n][m]*z) + Bs[n][m]*sin(-kms[n][m]*z));
            bz += -(Cs[n]*cos(n*phi)+Ds[n]*sin(n*phi))*iv[n][m]*kms[n][m]*(As[n][m]*sin(kms[n][m]*z) + Bs[n][m]*cos(-kms[n][m]*z));
        }
    }

    if (cart){
        out[0] = br*cos(phi)-bphi*sin(phi);
        out[1] = br*sin(phi)+bphi*cos(phi);
        out[2] = bz;
    }else{
        out[0] = br;
        out[1] = bphi;
        out[2] = bz;
    }
    return out;
}

//////////////////////////////////////
// For producing python .so wrapper //
// Use -DBPYTHON compile argument   //
// Requires linking to boost python //
//////////////////////////////////////
#ifdef BPYTHON
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;
typedef vector<double> vec_d;
BOOST_PYTHON_MODULE(fiteval_c)
{

    class_<vec_d>("vec_d")
        .def(vector_indexing_suite<vec_d>());

    class_<FitFunctionMaker>("FitFunctionMaker",init<string>())
        .def("mag_field_function",&FitFunctionMaker::mag_field_function);

};
#endif

int main(){
    FitFunctionMaker* myfitfunc = new FitFunctionMaker("param_825.csv");
    vector<double> my_vec = myfitfunc->mag_field_function(1,1,1);
    for(const auto& i: my_vec){
        cout<<i<<' ';
    }
    cout<<endl;
    delete myfitfunc;
}
