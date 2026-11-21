#pragma once
#include <complex>
#include <Eigen/Dense>
#include <numbers>

using namespace std::complex_literals;


using ScalarType = double;
using cScalarType = std::complex<ScalarType>;
using VectorType = Eigen::Vector<ScalarType, Eigen::Dynamic>;
using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
using cVectorType = Eigen::Vector<cScalarType, Eigen::Dynamic>;
using cMatrixType = Eigen::Matrix<cScalarType, Eigen::Dynamic, Eigen::Dynamic>;

using state_type = cVectorType; //everything is flattened into the vector and then reshaped used Eigen::Map<(const) Matrix>

enum struct lattice_type {LINEAR, SQUARE, TRIANGULAR};
enum struct boundary_conditions {OPEN, PERIODIC};
enum struct particle_statistic {BOSON, FERMION};
enum struct triang_WC_types {ONESIXTH, ONEFOURTH,ONETHIRD,ONEHALF,TWOTHIRDS,THREEFOURTHS};

constexpr int lattice_dim(lattice_type ltype) {
    switch(ltype) {
        case lattice_type::LINEAR: return 1;
        case lattice_type::SQUARE: return 2;
        case lattice_type::TRIANGULAR: return 2;
    }
}

template <lattice_type ltype, int ldim = lattice_dim(ltype)>
constexpr Eigen::Matrix<ScalarType, ldim, ldim> lattice_vectors() {//probably not evaluated at compile time because of eigen3
    constexpr ScalarType a = 1.0;
    constexpr ScalarType sqrt3 = 1.7320508075688772935274463415058723669428052538103806280558069794;
    if constexpr (ltype == lattice_type::LINEAR){
        return Eigen::Matrix<ScalarType, ldim, ldim>{a};}
    else if constexpr (ltype == lattice_type::SQUARE){
        return Eigen::Matrix<ScalarType, ldim, ldim>{{a,0},{0,a}};}
    else if constexpr (ltype == lattice_type::TRIANGULAR){
        return Eigen::Matrix<ScalarType, ldim, ldim>{{a,0.5*a},{0,0.5*sqrt3*a}};}
}

constexpr int particle_statistic_sigma(particle_statistic ptype) {
    switch(ptype) {
        case particle_statistic::BOSON: return 1;
        case particle_statistic::FERMION: return -1;
    }
}


struct SchurDec{
    cMatrixType U;
    cMatrixType T;

    SchurDec(const cMatrixType& iU, const cMatrixType& iT) 
        : U(iU), T(iT) {}
};

struct HermEigenDec{
    VectorType E;
    cMatrixType U;
    HermEigenDec(const VectorType& _E, const cMatrixType& _U) 
        : E(_E), U(_U) {}
};

struct iGR_HED{
    cMatrixType iGR_U_l;
    cMatrixType iGR_U_u;
    cMatrixType Uadj;
    iGR_HED(const cMatrixType& _iGR_U_l, const cMatrixType& _iGR_U_u, const cMatrixType& _Uadj) 
        : iGR_U_l(_iGR_U_l), iGR_U_u(_iGR_U_u), Uadj(_Uadj) {}
};

struct EigenDec{
    cVectorType E;
    cMatrixType V;
    cMatrixType Vinv;
    EigenDec(const cVectorType& _E, const cMatrixType& _V, const cMatrixType& _Vinv) 
        : E(_E), V(_V), Vinv(_Vinv) {}
};

struct iGR_ED{
    cMatrixType iGR_V_l;
    cMatrixType iGR_V_u;
    cMatrixType Vinv;
    iGR_ED(const cMatrixType& _iGR_V_l, const cMatrixType& _iGR_V_u, const cMatrixType& _Vinv) 
        : iGR_V_l(_iGR_V_l), iGR_V_u(_iGR_V_u), Vinv(_Vinv) {}
};


// struct LerchArgs{
//     const ScalarType z_t; 
//     const cScalarType L_pref;

//     LerchArgs(const ScalarType imu, const ScalarType t, const ScalarType inv_decaytime) :
//     z_t(std::exp(-inv_decaytime*t)),
//     L_pref(std::exp(cScalarType(0,imu*t))*z_t/std::numbers::pi)
//     {}
// };

struct anderson_accel_params{
    int m;
    int a;
    int b;
    ScalarType alpha;
    ScalarType alpha_start;
    anderson_accel_params(int _m, int _a, int _b, ScalarType _alpha, ScalarType _alpha_start)
        : m(_m), a(_a), b(_b), alpha(_alpha), alpha_start(_alpha_start) {}
};

//forward declaration of iPow 
template <typename T> constexpr T iPow(T x, unsigned int p);

struct lattice_params{
    boundary_conditions bcond;
    lattice_type ltype;
    int N;
    int ldim;
    int Ns;
    constexpr lattice_params(boundary_conditions _bcond, lattice_type _ltype, int _N)
        : bcond(_bcond), ltype(_ltype), N(_N), ldim(lattice_dim(_ltype)), Ns(iPow(_N, lattice_dim(_ltype))) {}
};
struct fermion_params{
    static inline int sigma = -1;

    int site_dof = 2;
    ScalarType filling;

    ScalarType j;
    ScalarType u;
    ScalarType v;
    ScalarType g; 
    ScalarType q_scr;

    ScalarType beta;
    ScalarType mu_eq;
    ScalarType mu;

    constexpr fermion_params(int _site_dof, ScalarType _filling, ScalarType _j, ScalarType _u, ScalarType _v, ScalarType _g,
               ScalarType _q_scr, ScalarType _beta)
    : site_dof(_site_dof), filling(_filling), j(_j), u(_u), v(_v), g(_g),
      q_scr(_q_scr), beta(_beta),
      mu_eq(std::numeric_limits<ScalarType>::quiet_NaN()), mu(std::numeric_limits<ScalarType>::quiet_NaN()) {}
};
struct boson_params{
    static inline int sigma = 1;

    int site_dof;
    ScalarType Np;

    ScalarType j;
    ScalarType u;
    ScalarType v;
    ScalarType g;
    ScalarType p;

    ScalarType rt;
    ScalarType ut;

    ScalarType beta;
    ScalarType mu_eq;
    ScalarType mu;

    constexpr boson_params(int _site_dof, ScalarType _Np, ScalarType _j, ScalarType _u,
             ScalarType _v, ScalarType _g, ScalarType _p, ScalarType _rt,
             ScalarType _ut, ScalarType _beta)
    : site_dof(_site_dof), Np(_Np), j(_j), u(_u),
      v(_v), g(_g), p(_p), rt(_rt),
      ut(_ut), beta(_beta),
      mu_eq(std::numeric_limits<ScalarType>::quiet_NaN()),
      mu(std::numeric_limits<ScalarType>::quiet_NaN()) {}
};

struct simulation_params{
    int n_cool;
    int m;
    ScalarType alpha;
    ScalarType t_0;
    ScalarType t_end;
    ScalarType t0_trap;
    ScalarType tau_trap;
    ScalarType t0_res;
    ScalarType tau_res;
    ScalarType del_t;
    ScalarType atol;
    ScalarType rtol;
    ScalarType max_stepsize;
    constexpr simulation_params(
                int _n_cool, int _m, ScalarType _alpha,
                ScalarType _t_0, ScalarType _t_end, ScalarType _t0_trap,
                ScalarType _tau_trap, ScalarType _t0_res, ScalarType _tau_res, ScalarType _del_t,
                ScalarType _atol, ScalarType _rtol, ScalarType _max_stepsize)
    :
    n_cool(_n_cool), m(_m), alpha(_alpha),
    t_0(_t_0), t_end(_t_end), t0_trap(_t0_trap),
    tau_trap(_tau_trap), t0_res(_t0_res), tau_res(_tau_res), del_t(_del_t),
    atol(_atol), rtol(_rtol), max_stepsize(_max_stepsize) {}
};