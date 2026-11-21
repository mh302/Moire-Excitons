#pragma once 
#include "types.hpp"



struct lattice {
    const lattice_type ltype;
    const boundary_conditions bcond;
    int dim; //real space dimensions of the lattice
    const int N; //number of sites per dimension
    int Ns; //number of sites = N^dim
    inline static double a = 1.0;
    
    
    MatrixType A;   //real lattice vectors
    MatrixType B; //reciprocal lattice vectors 

    lattice(lattice_type iltype, boundary_conditions ibcond, int iN);
    std::vector<int> nearN(int sidx) const;
    VectorType position(int sidx) const; 
    VectorType distance(int sidx1, int sidx2) const; 
};


struct bHubbardModel{
    const lattice lat; //type of lattice
    inline static particle_statistic particle_type = particle_statistic::BOSON;

    const MatrixType j;
    const VectorType delta;
    const MatrixType u;
    const MatrixType v;
    const MatrixType g; //reservoir coupling

    const int M; //number of orbital/band/spin DOF per site
    const int dim;
    const int p; //LR interaction ~1/r^p
    const ScalarType rt; //trap radius
    const ScalarType ut; //trap strength
    

    MatrixType H;
    MatrixType V;
    MatrixType H_eq;
    MatrixType Ht;
    MatrixType Gam;
    cMatrixType Heff_NI;
    cMatrixType Heff_NI_eq;
    // MatrixType U;


    bHubbardModel(lattice_type _ltype, boundary_conditions _bcond, int _N, MatrixType _j, VectorType _delta, MatrixType _u, MatrixType _v, MatrixType _g, int _p, ScalarType _rt, ScalarType _ut);
    void NNhopping();
    void Htrap();
    void LRint();
    ScalarType v_ij_periodic(VectorType r);
};



struct tVModel{
    const lattice lat; //type of lattice
    inline static particle_statistic particle_type = particle_statistic::FERMION;
    inline static int M = 1; //number of orbital/band/spin DOF per site
    const int dim;
    const ScalarType t; //hopping
    const ScalarType v; //coulomb interaction strength
    const ScalarType g; //reservoir coupling
    
    MatrixType H;
    MatrixType V;
    MatrixType Gam;
    cMatrixType Heff_NI;

    tVModel(lattice_type _ltype, boundary_conditions _bcond, int _N, ScalarType _t, ScalarType _v, ScalarType _g);
};

