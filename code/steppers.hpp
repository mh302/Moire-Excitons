#pragma once
#include "types.hpp"

#include <chrono>
#include <execution>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp> 
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/generation.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp> 
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_resize.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra_dispatcher.hpp>
#include <boost/fusion/container.hpp>

//----stepper----//
namespace boost::numeric::odeint {
    template<>
    struct vector_space_norm_inf<state_type> {
        using result_type = double;
        double operator()(const state_type& x) const {
            return x.cwiseAbs().maxCoeff();
        }
    };
}

typedef boost::numeric::odeint::runge_kutta4<
    state_type, ScalarType, state_type, ScalarType,
    boost::numeric::odeint::vector_space_algebra
> stepper_type;

typedef boost::numeric::odeint::runge_kutta_dopri5<
    state_type, ScalarType, state_type, ScalarType,
    boost::numeric::odeint::vector_space_algebra
> error_stepper_type;

typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;


inline std::chrono::nanoseconds totalTimeCI(0);
inline std::chrono::nanoseconds totalTime_updateHHF(0);


template <typename Func>
struct SystemBF_markov{
    Func update_H;
    //BOSONS
    const int b_dim;
    const int b_dxd;
    cMatrixType b_Heff;
    cMatrixType b_RHS;

    //FERMIONS
    const int f_dim;
    cMatrixType f_Heff;
    cMatrixType f_RHS;
    

    SystemBF_markov(Func _update_H,
             int _b_dim,
             int _f_dim): 
    update_H(std::move(_update_H)),
    b_dim(_b_dim), b_dxd(_b_dim*_b_dim), b_Heff(_b_dim,_b_dim), b_RHS(_b_dim,_b_dim),
    f_dim(_f_dim), f_Heff(_f_dim,_f_dim), f_RHS(_f_dim, _f_dim)
    {};

    void operator()(const state_type &x , state_type &dxdt , ScalarType t)
    {
        //BOSONS
        Eigen::Map<const cMatrixType> b_iG(x.data(), b_dim, b_dim);
        Eigen::Map<cMatrixType> b_diG(dxdt.data(), b_dim, b_dim);
        //FERMIONS
        Eigen::Map<const cMatrixType> f_iG(x.data()+b_dxd, f_dim, f_dim);        
        Eigen::Map<cMatrixType> f_diG(dxdt.data()+b_dxd, f_dim, f_dim);

        update_H(b_Heff, f_Heff, b_iG, f_iG, t);

        b_RHS = -1i*b_Heff*b_iG;
        b_diG.noalias() = b_RHS+b_RHS.adjoint();

        f_RHS.noalias() = -1i*f_Heff*f_iG;
        f_diG.noalias() = f_RHS+f_RHS.adjoint();

    }
};