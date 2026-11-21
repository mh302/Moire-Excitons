#pragma once 
#include "types.hpp"
#include "functions.hpp"
#include "model.hpp"

#include <boost/math/tools/roots.hpp>
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>

#include <execution>
#include <random>

struct periodic_anderson_accel{
    const int m;
    const int a; //period length of anderson mixing
    const int b; //period length of linear mixing
    const ScalarType alpha;
    const ScalarType alpha_start;
    const int dim;
    ScalarType lambda; //regularization
    cMatrixType P;
    cMatrixType G;
    cMatrixType D;
    int head;
    int it_cnt;
    ScalarType err;

    
    Eigen::ColPivHouseholderQR<cMatrixType> qr;
    Eigen::LLT<cMatrixType> llt;

    explicit periodic_anderson_accel(const int _m, const int _a, const int _b, const ScalarType _beta, const ScalarType _beta_start, const int _dim);
    ScalarType compute(cMatrixType& siG, const cMatrixType& n_siG);
    void reset();
    private:
    void update_head();
    ScalarType compute_linmix(cMatrixType& siG, const cMatrixType& n_siG);
    ScalarType compute_aacc(cMatrixType& siG);
    ScalarType compute_reg_aacc(cMatrixType& siG);
};


cMatrixType IS_uncorrelated_trap(const particle_statistic pstat, const lattice& lat, const int& site_dof, const ScalarType  rt, const ScalarType N_loc);
cMatrixType IS_random(const particle_statistic pstat, const bool correlated, const ScalarType nu, const int dim, std::mt19937& rng);
cMatrixType IS_soft_wigner_filling(const particle_statistic pstat, const lattice& lat, const triang_WC_types twct, const int M, const ScalarType nu, const ScalarType softness);
cMatrixType trial_state(const lattice& lat, const int dim, std::mt19937& rng, ScalarType nu, const cMatrixType& siG, const ScalarType alpha, const ScalarType beta, const ScalarType softness, bool fluid);


template <typename Func>
bool solve_Eq_MUfixed_SHFsc(const particle_statistic ptype, const ScalarType gam, const ScalarType mu_eq, Func&& updateSHF, const MatrixType H, cMatrixType& siG, const ScalarType beta,
    int& itcnt_sc, const unsigned int maxit_sc, const anderson_accel_params aacc_ps, const unsigned int dim){
    const int sigma = particle_statistic_sigma(ptype);
    if (ptype==particle_statistic::BOSON && gam != 0.0) throw std::invalid_argument("No open system equilibrium implemented for bosons.");
    
    Eigen::SelfAdjointEigenSolver<cMatrixType>es(dim);


    cMatrixType SHF(dim,dim);
    updateSHF(SHF, siG);




    auto occ = [beta, mu_eq, sigma, gam](ScalarType e){
        if(gam==0.0){
            // ScalarType z = std::exp(beta*(mu_eq-e));
            // return z/(1.0-sigma*z);
            return 1.0/(std::exp(beta*(e-mu_eq))-sigma);
        }
        else{//open system only implemented for fermions, but the case boson+gam!=0 is checked at the start
            cScalarType a = 0.5*(1.0+1i*beta/std::numbers::pi*(e-mu_eq-1i*gam));
            return 0.5-digamma(a).imag()/std::numbers::pi;
        }
    };

    constexpr double tol_sc = 1e-4;
    // constexpr int maxit_sc = 3000;
    itcnt_sc = 0;
    

    periodic_anderson_accel aacc(aacc_ps.m, aacc_ps.a, aacc_ps.b, aacc_ps.alpha, aacc_ps.alpha_start, dim);
    VectorType f_evals(dim);
    cMatrixType n_SHF(dim,dim);

    ScalarType err_sc = std::numeric_limits<ScalarType>::infinity();
    cMatrixType siG_old = siG;
    do {        
        
        es.compute(H+SHF);

        siG_old = siG;     
        //compute siG
        f_evals = es.eigenvalues().unaryExpr(occ);

        siG.noalias() = es.eigenvectors()*f_evals.asDiagonal()*es.eigenvectors().adjoint();

        updateSHF(n_SHF, siG);
        
        err_sc = aacc.compute(SHF, n_SHF);

        itcnt_sc++;
    }
    while(err_sc>tol_sc && itcnt_sc < maxit_sc);
    

    if(itcnt_sc != maxit_sc) return true;
    else return false;
}

template <typename Func>
bool solve_Eq_Nfixed_SHFsc(const particle_statistic ptype, const ScalarType gam, ScalarType& mu_eq, Func&& updateSHF, const MatrixType H, cMatrixType& siG, const ScalarType beta, const ScalarType Np,
    int& itcnt_sc, const unsigned int maxit_sc, const anderson_accel_params aacc_ps, const unsigned int dim){
    const int sigma = particle_statistic_sigma(ptype);
    if (ptype==particle_statistic::BOSON && gam != 0.0) throw std::invalid_argument("No open system equilibrium implemented for bosons.");
    
    ScalarType mu = 0;

    Eigen::SelfAdjointEigenSolver<cMatrixType>es(dim);

    cMatrixType SHF(dim,dim);
    updateSHF(SHF, siG);



    auto occ = [beta, sigma, gam](ScalarType e, ScalarType mu){
        if(gam==0.0){
            // ScalarType z = std::exp(beta*(mu-e));
            // return z/(1.0-sigma*z);
            return 1.0/(std::exp(beta*(e-mu))-sigma);
            // if(sigma==1) return 0.5*(1.0/std::tanh(beta*(e-mu)/2.0)-1.0);
            // else return 0.5*(1.0-std::tanh(beta*(e-mu)/2.0));
        
        }
        else{//open system only implemented for fermions, but the case boson+gam!=0 is checked at the start
            cScalarType a = 0.5*(1.0+1i*beta/std::numbers::pi*(e-mu-1i*gam));
            return 0.5-digamma(a).imag()/std::numbers::pi;
        }
    };

    ScalarType xmin; //lower bound of search interval
    ScalarType xmax; //upper bound

    constexpr double tol_sc = 1e-4;
    itcnt_sc = 0;
    
    const boost::math::tools::eps_tolerance<ScalarType> tol_root(48);
    std::uintmax_t maxit_root = 50;

    periodic_anderson_accel aacc(aacc_ps.m, aacc_ps.a, aacc_ps.b, aacc_ps.alpha, aacc_ps.alpha_start, dim);
    VectorType f_evals(dim);
    cMatrixType n_SHF(dim,dim);

    ScalarType err_sc = std::numeric_limits<ScalarType>::infinity();
    ScalarType err_sc_old = std::numeric_limits<ScalarType>::infinity();

    cMatrixType siG_old = siG;
    
    do {        
        es.compute(H+SHF);
        auto nminusNp = [Np, &ens=es.eigenvalues(), &occ](const ScalarType mu_x){ //lambda function for the root finder                
            auto occ_mu = [mu_x, &occ](ScalarType e){
                return occ(e, mu_x);
            };
            return ens.unaryExpr(occ_mu).sum()-Np;
        };

        
        if(ptype==particle_statistic::BOSON){
            xmin = es.eigenvalues()(0)-std::log(1.0+dim/Np)/beta-1e-12;
            xmax = es.eigenvalues()(0)-std::log(1.0+1.0/Np)/beta+1e-12;
        }
        else{//bounds are derived by assuming dim-fold degenerate energy level. Thermal shift is exact, reservoir shift is what you get from gamma/beta>>1. If rootfinder fails, increase the interval
            ScalarType fill = Np/dim; //static_cast not needed, Np is ScalarType
            ScalarType shift = -std::log(1.0/fill-1.0)/beta-gam*std::tan(std::numbers::pi*(0.5-fill)); //thermal shift + reservoir shift 
            xmin = es.eigenvalues()(0)+shift;
            xmax = es.eigenvalues()(Eigen::placeholders::last)+shift;       
        }


        maxit_root = 50;
        auto root = boost::math::tools::toms748_solve(nminusNp, xmin, xmax, tol_root, maxit_root);
        mu = (root.first+root.second)/2.0;
        //compute siG
        f_evals = es.eigenvalues().unaryExpr([&](ScalarType e) { return occ(e, mu); });

        siG_old = siG;
        siG.noalias() = es.eigenvectors()*f_evals.asDiagonal()*es.eigenvectors().adjoint();
        

        err_sc_old = err_sc;
        updateSHF(n_SHF, siG);
        err_sc = aacc.compute(SHF, n_SHF);

        itcnt_sc++;

    }
    while(err_sc>tol_sc && itcnt_sc < maxit_sc);
    
    mu_eq = mu; //pass back the correct chemical potential

    if(itcnt_sc != maxit_sc) return true;
    else return false;
}