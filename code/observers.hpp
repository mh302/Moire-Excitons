#pragma once
#include "types.hpp"

template <lattice_type ltype, int ldim = lattice_dim(ltype)>
struct diameter_observer{
    const Eigen::Matrix<ScalarType, ldim, ldim> A;

    const int sigma;
    const int N;
    const int site_dof = 2;
    
    const int Ns;
    const int dim;
    Eigen::Vector<ScalarType, ldim> s0;
    const ScalarType R0sq;


    diameter_observer(particle_statistic ptype, int _N, int _site_dof, const Eigen::Ref<const cMatrixType>& iG0) :
    A(lattice_vectors<ltype>()),
    sigma(particle_statistic_sigma(ptype)),
    N(_N),
    site_dof(_site_dof),
    Ns(iPow(_N,ldim)),
    dim(site_dof*Ns),
    s0(init_s0(iG0)),
    R0sq(Rsq(iG0))
    {
        // std::cout << "s0: " << s0 << " R0sq: " << R0sq << std::endl;
    }

    ScalarType compute_radius(const Eigen::Ref<const cMatrixType>& iG){
        // return std::sqrt(Rsq(iG)-R0sq+1e-14);
        s0 = init_s0(iG);
        return Rsq(iG)-R0sq; //no sqrt (just convention)
    }

    private:
    Eigen::Vector<ScalarType, ldim> init_s0(const Eigen::Ref<const cMatrixType>& iG0){
        Eigen::Vector<ScalarType, ldim> res = Eigen::Vector<ScalarType, ldim>::Zero();
        ScalarType N_tot = 0;
        for(int n=0;n<Ns;++n){
            Eigen::Vector<ScalarType, ldim> r;
            for(int i=0;i<ldim;++i){
                r(i) = (n/iPow(N,i))%N;
            }
            ScalarType occ_n = 0;
            for(int m=0;m<site_dof;++m){
                occ_n += sigma*iG0(site_dof*n+m,site_dof*n+m).real();
            }
            N_tot += occ_n;
            res += occ_n*r;
        }
        return A*res/N_tot;
    }

    ScalarType Rsq(const Eigen::Ref<const cMatrixType>& iG){
        ScalarType res = 0;
        ScalarType N_tot = 0;
        for(int n=0;n<Ns;++n){
            Eigen::Vector<ScalarType, ldim> r;
            for(int i=0;i<ldim;++i){
                r(i) = (n/iPow(N,i))%N;
            }
            ScalarType occ_n = 0;
            for(int m=0;m<site_dof;++m){
                occ_n += sigma*iG(site_dof*n+m,site_dof*n+m).real();
            }
            N_tot += occ_n;
            res += occ_n*(A*r-s0).squaredNorm();
            // std::cout << "res: " << res  << " n: " << n << " occ_n: " << occ_n<< std::endl;
        }
        return res/N_tot;
    }
};


struct particle_number_observer{

    const int sigma;

    particle_number_observer(particle_statistic ptype) :
    sigma(particle_statistic_sigma(ptype))
    {}

    ScalarType compute_radius(const Eigen::Ref<const cMatrixType>& iG){
        return sigma*iG.trace().real();
    }

};