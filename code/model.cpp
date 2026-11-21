#include "model.hpp"
#include "functions.hpp"
#include "types.hpp"
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <fftw3.h>
#include <unsupported/Eigen/KroneckerProduct>
using namespace std::complex_literals;


lattice::lattice(lattice_type iltype, boundary_conditions ibcond, int iN) 
    :ltype(iltype), bcond(ibcond), N(iN) {
    if(ltype == lattice_type::LINEAR){
        dim = 1;
        A = MatrixType::Zero(dim,dim);
        B = MatrixType::Zero(dim,dim);
        A << 1*a;
        B << 2*std::numbers::pi/A(0,0);
    }
    else if(ltype == lattice_type::SQUARE){
        dim = 2;
        A = MatrixType::Zero(dim,dim);
        B = MatrixType::Zero(dim,dim);
        A << 1, 0, 0, 1;
        A *= a;
        B = 2*std::numbers::pi*(A.inverse()).transpose();
    }
    else if(ltype == lattice_type::TRIANGULAR){
        dim = 2;
        A = MatrixType::Zero(dim,dim);
        B = MatrixType::Zero(dim,dim);
        A << 1, 0.5, 0, 0.5*sqrt(3);
        A *= a;
        B = 2*std::numbers::pi*(A.inverse()).transpose();
    }
    Ns = iPow(N,dim);
}

std::vector<int> lattice::nearN(int sidx) const{
    std::vector<int> NN(0);
if(bcond == boundary_conditions::OPEN){

        if(ltype == lattice_type::LINEAR){
            bool not_left  = (sidx!=0);//not left boundary
            bool not_right = (sidx!=N-1);//not right boundary

            if(not_left) NN.push_back(sidx-1); //(-1)
            if(not_right) NN.push_back(sidx+1); //(+1)
        }
        else if(ltype == lattice_type::SQUARE){
            bool not_lower = (mod((sidx+1),N)!=0);//not lower boundary
            bool not_upper = (mod(sidx,N)!=0);//not upper boundary
            bool not_left  = (sidx/N!=0);//not left boundary
            bool not_right = (sidx/N!=N-1);//not right boundary

            if(not_lower) NN.push_back(sidx+1); //(0,+1)
            if(not_upper) NN.push_back(sidx-1); //(0,-1)
            if(not_left) NN.push_back(sidx-N); //(-1,0)
            if(not_right) NN.push_back(sidx+N); //(+1,0)
        }
        else if(ltype == lattice_type::TRIANGULAR){
            bool not_lower = (mod((sidx+1),N)!=0);//not lower boundary
            bool not_upper = (mod(sidx,N)!=0);//not upper boundary
            bool not_left  = (sidx/N!=0);//not left boundary
            bool not_right = (sidx/N!=N-1);//not right boundary

            if(not_lower) NN.push_back(sidx+1); // (0, +1)
            if(not_upper) NN.push_back(sidx-1); // (0, -1)
            if(not_left)  NN.push_back(sidx-N); // (-1, 0)
            if(not_right) NN.push_back(sidx+N); // (+1, 0)
            if(not_left && not_lower)  NN.push_back(sidx-N+1); // (-1, +1)
            if(not_right && not_upper) NN.push_back(sidx+N-1); // (+1, -1)
        }
    }
    else if(bcond == boundary_conditions::PERIODIC){
        if(N<3) throw std::invalid_argument("For PERIODIC boundary conditions, choose N>2");
        if(ltype == lattice_type::LINEAR){
            NN.push_back(mod((sidx+1),N)); //(+1)
            NN.push_back(mod((sidx-1),N));//(-1)
        }
        else if(ltype == lattice_type::SQUARE){
            NN.push_back(mod((sidx+N),Ns));//(+1,0)
            NN.push_back(mod((sidx-N),Ns));//(-1,0)
            NN.push_back((sidx/N)*N+mod(sidx+1,N));//(0,+1)
            NN.push_back((sidx/N)*N+mod(sidx-1,N));//(0,-1)
        }
        else if(ltype == lattice_type::TRIANGULAR){
            NN.push_back(mod((sidx+N),Ns)); //(+1,0)
            NN.push_back(mod((sidx-N),Ns)); //(-1,0)
            NN.push_back((sidx/N)*N+mod(sidx+1,N)); //(0,+1)
            NN.push_back((sidx/N)*N+mod(sidx-1,N)); //(0,-1)
            NN.push_back(mod(((sidx/N)*N+mod((sidx+1),N)-N), Ns)); //(-1,+1)
            NN.push_back(mod(((sidx/N)*N+mod((sidx-1),N)+N), Ns));//(+1,-1)
        }
    }
    return NN;
}


VectorType lattice::position(int sidx) const{ 
    Eigen::Vector<double, Eigen::Dynamic> r(dim);
    for(int i=0;i<dim;++i){
        r(i) = (sidx/iPow(N,i))%N;
    }
    return A*r;
}


VectorType lattice::distance(int sidx1, int sidx2) const{ 
    Eigen::Vector<double, Eigen::Dynamic> r1(dim);
    Eigen::Vector<double, Eigen::Dynamic> r2(dim);
    for(int i=0;i<dim;++i){
        r1(i) = (sidx1/iPow(N,i))%N;
        r2(i) = (sidx2/iPow(N,i))%N;
    }
    return A*(r1-r2);
}



bHubbardModel::bHubbardModel(lattice_type _ltype, boundary_conditions _bcond, int _N, MatrixType _j, VectorType _delta, MatrixType _u, MatrixType _v, MatrixType _g, int _p, ScalarType _rt, ScalarType _ut)
    : 
    lat(_ltype, _bcond, _N),
    j(_j),
    delta(_delta),
    u(_u),
    v(_v),
    g(_g),
    M(static_cast<int>(_j.cols())),
    dim(lat.Ns*M),
    p(_p),
    rt(_rt),
    ut(_ut),
    H(dim,dim),
    V(dim,dim),
    H_eq(dim,dim),
    Ht(dim,M),
    Gam(dim,dim),
    Heff_NI(dim,dim),
    Heff_NI_eq(dim,dim)
{   
    // MatrixType H_spinless = MatrixType::Zero(lat.Ns, lat.Ns);
    NNhopping();
    // H = Eigen::KroneckerProduct(H_spinless, sigma_0);
    Htrap();
    LRint();

    //shift by band energies
    assert(delta.size() == M && "delta must have length M");
    H.diagonal() += delta.replicate(lat.Ns,1);


    Gam = Eigen::KroneckerProduct(MatrixType::Identity(lat.Ns,lat.Ns), g);
    Heff_NI = H-1i*Gam;
    Heff_NI_eq = Heff_NI;
    for(int i=0; i<lat.Ns;++i){
        Heff_NI_eq.block(M*i,M*i,M,M) += Ht.block(M*i,0,M,M);
    }

    H_eq = H;
    for(int i=0; i<lat.Ns;++i){
        H_eq.block(M*i,M*i,M,M) += Ht.block(M*i,0,M,M);
    }
};

void bHubbardModel::NNhopping(){
    MatrixType H_spinless = MatrixType::Zero(lat.Ns, lat.Ns);
    for(int n=0; n<lat.Ns; ++n){
        std::vector<int> NN = lat.nearN(n);
        for(auto m : NN){
            H_spinless(n,m) -= 0.5;
            H_spinless(m,n) -= 0.5;
        }
    }
    H = Eigen::KroneckerProduct(H_spinless, j);
}

void bHubbardModel::Htrap(){
    VectorType Ht_lat(lat.Ns);
    VectorType r_center(lat.dim); //vector to the central lattice site
    for(auto& x : r_center){
        x = (lat.N-1)/2.0; 
    }
    VectorType r(lat.dim);
    for(int n=0; n<lat.Ns; ++n){
        for(int i=0;i<lat.dim;++i){
            r(i) = (n/iPow(lat.N,i))%lat.N;
        }
        if((lat.A*(r-r_center)).norm()<rt+1e-9){
            Ht_lat(n) = 0; 
        }
        else {
            Ht_lat(n) = ut;
        }
    }
    Ht = Eigen::KroneckerProduct(Ht_lat, MatrixType::Identity(M,M));
}


void bHubbardModel::LRint(){
    MatrixType V_lat(lat.Ns, lat.Ns);
    if(lat.bcond==boundary_conditions::OPEN){
        for(int n=0;n<lat.Ns;++n){
            for(int m=n+1;m<lat.Ns;++m){
                ScalarType d = lat.distance(n,m).norm();
                ScalarType pot =  1.0/std::pow(d, p);
                if(abs(pot)>1e-14){
                    V_lat(n,m) = pot;
                    V_lat(m,n) = pot;
                }
                else{
                    V_lat(n,m) = 0.0;
                    V_lat(m,n) = 0.0;
                }
            }
        }
    }
    else if(lat.bcond==boundary_conditions::PERIODIC){

        VectorType vLR(lat.Ns);
        VectorType r(lat.dim);
        VectorType x(lat.dim);
        vLR(0)=0;
        for(int n=1; n<lat.Ns; ++n){//start at 1 to skip the diverging self interaction
            for(int i=0; i<lat.dim; ++i){
                r(i) = (n/iPow<int>(lat.N,i))%lat.N;
            }
            vLR(n) = v_ij_periodic(lat.A*r);
        }

        double* in = (double*) fftw_malloc(sizeof(double)*lat.Ns);
        fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*iPow<unsigned int>(lat.N,lat.dim-1)*(lat.N/2+1)); //allocate only non redundant values (out[n-i]=out[i]^*)
        fftw_plan pl;

        if(lat.dim==1){
            pl = fftw_plan_dft_r2c_1d(lat.N, in, out, FFTW_ESTIMATE);
        }
        else if(lat.dim==2){
            pl = fftw_plan_dft_r2c_2d(lat.N, lat.N, in, out, FFTW_ESTIMATE);
        }
        else throw std::invalid_argument("Lattice dimension can only be 1 or 2");

        for(int n=0; n<lat.Ns; ++n){
            in[n] = vLR[n];
        }
        fftw_execute(pl);

        cVectorType Ug(lat.Ns);
        for(int i=0;i<lat.Ns;++i){
            if(i%lat.N<(lat.N/2+1)){
                Ug(i) = cScalarType(out[(i/lat.N)*(lat.N/2+1)+i%lat.N][0],out[(i/lat.N)*(lat.N/2+1)+i%lat.N][1]);
            }
            else{
                int i2 = 0;
                for(int n=0;n<lat.dim;++n){ //the old formula is not applicable, because if i_n=(i/iPow<unsigned int>(N,n))%N=0, then N-i_n=N, but due to the periodicity it should be 0
                    i2+=iPow<unsigned int>(lat.N,n)*((lat.N-(i/iPow<unsigned int>(lat.N,n))%lat.N)%lat.N);
                }
                Ug(i) = cScalarType(out[(lat.N/2+1)*(i2/lat.N)+i2%lat.N][0],-out[(lat.N/2+1)*(i2/lat.N)+i2%lat.N][1]);
                }
        }
        fftw_destroy_plan(pl);
        fftw_free(in); fftw_free(out);

        cMatrixType P(lat.Ns,lat.Ns);
        VectorType R(lat.dim);
        VectorType G(lat.dim);
        for(int n=0; n<lat.Ns; ++n){
            for(int i=0;i<lat.dim;++i){
                R(i) = (n/iPow<unsigned int>(lat.N,i))%lat.N;
            }
                for(int m=0; m<lat.Ns; ++m){
                    for(int i=0;i<lat.dim;++i){
                        G(i) = (m/iPow<unsigned int>(lat.N,i))%lat.N;
                    }
                    std::complex<double> arg(0,-2*std::numbers::pi/lat.N*R.dot(G));
                    P(n,m) = exp(arg);
            }
        }
        P *= 1/sqrt(lat.Ns);
        V_lat = (P*Ug.asDiagonal()*P.adjoint()).real();
    }
    V = Eigen::KroneckerProduct(V_lat,v)+Eigen::KroneckerProduct(MatrixType::Identity(lat.Ns,lat.Ns),u);
}


ScalarType bHubbardModel::v_ij_periodic(VectorType r){
    constexpr int N_cells = 101;
    ScalarType vij = 0;
    VectorType R(lat.dim);
    for(int n=0; n<iPow<int>(N_cells,lat.dim); ++n){
        for(int i=0; i<lat.dim; ++i){
            R(i) = (n/iPow<int>(N_cells,i))%N_cells-N_cells/2;
        }
        ScalarType d = (r-lat.A*R*lat.N).norm();
        ScalarType tmp = 1.0/std::pow(d,p);
        vij += tmp;
        // if(tmp<=1e-9) break;
    }
    return vij;
}


tVModel::tVModel(lattice_type _ltype, boundary_conditions _bcond, int _N, ScalarType _t, ScalarType _v, ScalarType _g)
    : 
    lat(_ltype, _bcond, _N),
    dim(M*lat.Ns),
    t(_t),
    v(_v),
    g(_g),
    H(dim,dim),
    V(dim,dim),
    Gam(dim,dim),
    Heff_NI(dim,dim)
{
    MatrixType NearN = MatrixType::Zero(lat.Ns,lat.Ns);

    for(int n=0; n<lat.Ns; ++n){
        std::vector<int> NN = lat.nearN(n);
        for(auto m : NN){
            NearN(n,m) += 0.5;
            NearN(m,n) += 0.5;
        }
    }
    H = -t*NearN;
    V = v*NearN;
    Gam = g*MatrixType::Identity(dim,dim);
    Heff_NI = H-cScalarType(0,1)*Gam;
};
