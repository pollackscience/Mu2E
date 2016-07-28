#include "fiteval_c.h"

int main(){
    FitFunctionMaker* ffm       = new FitFunctionMaker("Mau10_800mm_long.csv");
    FitFunctionMaker* ffm_bad_m = new FitFunctionMaker("Mau10_bad_m_test_req.csv");
    vector<double> bxyz         = ffm->get_field(100,-500,10000, true);
    vector<double> bxyz_bad_m   = ffm_bad_m->get_field(100,-500,10000, true);

    cout<<"\tBx\t\tBy\tBz";
    cout<<"\nIdeal Map:\n";
    for(const auto& i: bxyz){
        cout<<i<<"\t";
    }
    cout<<"\nMeasurement Errors:\n";
    for(const auto& i: bxyz_bad_m){
        cout<<i<<"\t";
    }
    cout<<endl;
    delete ffm;
}
