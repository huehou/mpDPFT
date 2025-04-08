#include "Plugin_1pEx_Rho1p.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cfloat>

#include <vector>
#include <algorithm>
#include <numeric>

void init_rho1p(const size_t L, std::vector<std::vector<double>>& rho, const size_t M, const std::vector<size_t>& p, double num){


    // Clear rho in the region we will work on
    //
    for(size_t i = 0; i < L; i++){
        for(size_t j = 0; j < L; j++){
            rho[i][j] = 0.0;
        }
    }

    size_t K = fmax(round(num/2.0), 1.0);
    assert(K <= M);


    for(size_t k = 0; k < K; k++){
        size_t ik = p[k];
        rho[ik][ik] = 2.0/num;
        /*printf("rho[%zd][%zd] = %f\n", ik,ik,rho[ik][ik]);*/
    }
}

void normalize_rho1p(size_t L, std::vector<std::vector<double>>& rho, size_t M, std::vector<double>& occnum, std::vector<size_t>& p, double num){

    // Verify majorization step by step and adjust 
    // if necessary
    //

    double ssum = 0.0;
    double tsum = 0.0;

    for(size_t k = 0; k < M; k++){

        size_t ik = p[k];

        ssum += occnum[ik]/num;
        tsum += rho[ik][ik];

        double delta = ssum - tsum;
    
        // Majorization failed, distribute
        // the difference accross the elements
        // (This is guaranteed not to violate majorization 
        // and decreasing ordering of t)
        //
        if(delta > 0){

            double deltap = delta/(double)(k+1);

            for(size_t l = 0; l < (k+1); l++){
                int il = p[l];
                rho[il][il] += deltap;
            }

            tsum += delta;
        }
    }
    
    // Adjust norm
    double deltan = ssum - tsum;

    // We can not distribute over the range 
    // of eigenvalues, since the roundoff error 
    // might make the adjustement disappear.
    // It is generally expected to be small.
    // This however, might make some eigenvalue slightly negative
    
    rho[p[M-1]][p[M-1]] += deltan;
}

/*
 * Compute the worst case relative error bound
 */
double relerr_occnum(size_t M, std::vector<double>& occnum, const std::vector<size_t>& p){

    // Find the absolute max / min for occnum
    //
    double ksi = fabs(occnum[p[0]]);

    double absmin = 0.0;

    for(size_t k = M-1; k >= 0; k--){

        double occ = occnum[p[k]];

        if(occ == 0){
            continue;
        }

        // We hit first positive value, end of search
        if(occ > 0){

            if(absmin > occ || absmin == 0.0){
                absmin = occ;
                break;
            }
            else{
                if(fabs(occ) < absmin){
                    absmin = fabs(occ);
                }
            }
        }
    }

    ksi /= absmin;

    double lambda = 2*ksi*(M-1)*DBL_EPSILON;
    
    return lambda;

}

enum getrho1p_status {
    getrho1p_err_oom = -1,
    getrho1p_err_source_not_found = -2,
    getrho1p_err_target_less_source = -3,
    getrho1p_err_target_less_occ = -4,
    getrho1p_err_eta_inf = -5,
    getrho1p_err_eta_neg = -6,
    getrho1p_err_eta_nan = -7,
    getrho1p_err_eta_inv = -8,
    getrho1p_success = 0,
};

int GenerateRho1pMixer(std::vector<std::vector<double>>& rho, size_t L, std::vector<double>& occnum, double num){

    size_t M = occnum.size();

    assert(num > 0.0);
    assert(M >= num/2.0);
    assert(L >= M);

    /*
     * Generate sorting permutation for occnum 
     */

     /*
     * Generate the sorting permutation
     */

    std::vector<size_t> p(occnum.size(), 0);
    for(size_t i = 0 ; i != p.size() ; i++) {
        p[i] = i;
    }
    // Sort the occupation numbers into descending sequence
    // by generating the ordering.
    sort(p.begin(), p.end(),
        [&](const int& a, const int& b) {
            return (occnum[a] > occnum[b]);
        }
    );

    /*
     * Initialize rho
     */

    // Set num/2 eigenvalues to 2/num
    init_rho1p(L, rho, M, p, num);

    // Verify majorization and adjust 
    // if it does not hold 
    normalize_rho1p(L, rho, M, occnum, p, num);

    // DEBUG
//     printf("Initial rho: ");
//     for(int i = 0; i < L; i++){
//         printf("%f, ", rho[p[i]][p[i]]);
//     }
//     puts("\n");
    // Compute the maximum relative error 
    // for the given occnum
    //double lambda = relerr_occnum(M, occnum, p);//original
    //double lambda = 5.*relerr_occnum(M, occnum, p);
    double lambda = 0.;//MIT20220413

    /*printf("rho:\n");*/
    /*show_rho(L, rho);*/

    // Target index
    size_t T = 0; 
    
    // Source index
    size_t S = 0;

    // Indices in the ordered view
    size_t iT = T;
    size_t iS = p[S];


    for(T = 0; T < M-1; T++){

        iT = p[T];

        double occ = occnum[iT]/num;

        // Question: In Julia version this sometimes does not work
        if(fabs(rho[iT][iT] - occ) <= lambda*fabs(occ)){
            continue;
        }

        // Outside of tolerance
        if(rho[iT][iT] < occ) return getrho1p_err_target_less_occ;

        double tT = rho[iT][iT];
        
        // Find the source to within tolerance
        while(rho[iS][iS] > occ && fabs(occ - rho[iS][iS]) > lambda*fabs(occ)){

            S++;

            if(S >= M){
                return getrho1p_err_source_not_found;
            }

            assert(S < M);

            iS = p[S];
        }

        double tS = rho[iS][iS];

        // This can only happen 
        // when working with tolerated quantities 
        if(tT <= tS) return getrho1p_err_target_less_source;

        double eta = (occ - tS)/(tT - tS);

        // This is guaranteed by the algorithm 

        if(eta > 1.0) return getrho1p_err_eta_inv;
        if(eta == INFINITY || eta == -INFINITY) return getrho1p_err_eta_inf;
        if(eta == NAN) return getrho1p_err_eta_nan;

        // tS is greater than occ to within tolerance
        // in such case we simply swap tS and tT
        // and set target to the occupation number
        if( eta < 0.0 ){
            rho[iT][iT] = occ;
            rho[iS][iS] = tT + tS - occ;
        }
        else{
            double sum = tS + tT;
            double diff = tT - tS;

            rho[iT][iT] = occ;
            rho[iS][iS] = sum - occ;

            rho[iT][iS] = sqrt(eta*(1-eta))*diff;
            rho[iS][iT] = rho[iT][iS];

            // Matrix multiplication in linear time
            size_t iM = 0;

            for(size_t m = 0; m < M; m++){

                if(m == S || m == T){
                    continue;
                }

                iM = p[m];

                double x = rho[iM][iT];
                double y = rho[iM][iS];

                rho[iM][iT] = x*sqrt(eta) + y*sqrt(1.0 - eta);
                rho[iM][iS] = x*sqrt(1-eta) - y*sqrt(eta);

                rho[iS][iM] = rho[iM][iS];
                rho[iT][iM] = rho[iM][iT];
            }
        }

    }

    return 0;
}

