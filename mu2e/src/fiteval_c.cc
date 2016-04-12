#include "fiteval_c.h"

FitFunctionMaker::FitFunctionMaker(string fit_csv){
	io::CSVReader<2> in(fit_csv);
	in.set_header("param","val");
	string param; double val;
	while(in.read_row(param, val)){
        cout<<param<<": "<<val<<"\n";
	}
    cout<<endl;
}

int main(){
    FitFunctionMaker* myfitfunc = new FitFunctionMaker("param_825.csv");
    delete myfitfunc;
}
