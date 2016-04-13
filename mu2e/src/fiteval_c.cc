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


   cout<<Bs[0][0]<<endl;
   cout<<Cs[0]<<endl;
   cout<<Reff<<ns<<ms<<endl;
}

#include <boost/python.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(fiteval_c)
{

    class_<FitFunctionMaker>("FitFunctionMaker",init<string>());
};

int main(){
    FitFunctionMaker* myfitfunc = new FitFunctionMaker("param_825.csv");
    delete myfitfunc;
}
