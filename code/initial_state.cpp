#include "initial_state.hpp"



cMatrixType IS_uncorrelated_trap(const particle_statistic pstat, const lattice& lat, const int& site_dof, const ScalarType  rt, const ScalarType N_loc){
    if(pstat==particle_statistic::FERMION && N_loc>1.0)  throw std::invalid_argument("For Fermions, Pauli principle forbids N_loc>1");

    VectorType occs = VectorType::Zero(lat.Ns*site_dof);
    VectorType r_center(lat.dim); //vector to the central lattice site
    for(auto& x : r_center){
        x = (lat.N-1)/2.0; 
    }
    r_center = lat.A*r_center;
    for(int n=0; n<lat.Ns; ++n){
        if((lat.position(n)-r_center).norm()<rt+1e-9){
            occs(site_dof*n) = N_loc; //only occupy lowest band
        }
    }
    return particle_statistic_sigma(pstat)*occs.cast<cScalarType>().asDiagonal();
}

cMatrixType IS_random(const particle_statistic pstat, const bool correlated, const ScalarType nu, const int dim, std::mt19937& rng){
    VectorType occs(dim);
    std::uniform_real_distribution<double> U(0, 1.0);
    occs = VectorType::NullaryExpr(dim, [&](){ return U(rng);});

    const ScalarType r=0.2;

    ScalarType nu_min;
    ScalarType nu_max;
    if(nu<r){
        nu_min = 0;
        nu_max = 2.0*nu;
    }
    else if(nu<1-r){
        nu_min = nu-r;
        nu_max = nu+r;
    }
    else{
        nu_min = 2.0*nu-1.0;
        nu_max = 1.0;
    }

    occs *= (nu_max-nu_min);
    occs.array() += nu_min;


    if(pstat == particle_statistic::BOSON){
        constexpr ScalarType LARGE = 100.0;
        occs *= LARGE;
    }

    if(correlated == true){//may not be thread safe
        Eigen::ColPivHouseholderQR<cMatrixType> qr(dim, dim);
        cMatrixType rd_M = cMatrixType::Random(dim, dim);
        qr.compute(rd_M);
        cMatrixType Q = qr.matrixQ();
        return particle_statistic_sigma(pstat)*Q*occs.asDiagonal()*Q.adjoint();
    }

    return particle_statistic_sigma(pstat)*occs.cast<cScalarType>().asDiagonal();
}

cMatrixType IS_soft_wigner_filling(const particle_statistic pstat, const lattice& lat, const triang_WC_types twct, const int M, const ScalarType nu, const ScalarType softness) {
    int Norb = lat.Ns * M;
    VectorType occs = VectorType::Zero(Norb);
    

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> loc_dist(0,M-1); 
    
    auto filling_cond = [twct, &lat](int sidx){
        int xidx = sidx%lat.N;
        int yidx = (sidx/lat.N)%lat.N;
        if(twct==triang_WC_types::ONEFOURTH){
            return (xidx%2==0&&yidx%2==0);
        }
        else if(twct==triang_WC_types::THREEFOURTHS){
            return !(xidx%2==0 && yidx%2==0);
        }
        else if(twct==triang_WC_types::ONEHALF){
            return (xidx%2==0);
        }
        else if(twct==triang_WC_types::ONETHIRD){
            return ((xidx+2*yidx)%3==0);
        }
        else if(twct==triang_WC_types::TWOTHIRDS){
            return ((xidx+2*yidx)%3!=0);
        }
        else return false;
    };

    VectorType backbone = VectorType::Zero(Norb);
    for (int n=0; n<lat.Ns; ++n){
        if(filling_cond(n)){
            int s = loc_dist(gen); //pick a random spin/orbital
            backbone(M*n+s) = 1.0;
        }
    }


    ScalarType N_target = nu * Norb;
    ScalarType N_cryst = backbone.sum();
    ScalarType N_other = Norb - N_cryst;
    

    ScalarType nH_max = 0.0;
    ScalarType nL_max = 0.0;
    
    if (N_target >= N_cryst) {
        nH_max = 1.0;
        nL_max = (N_target-N_cryst)/N_other;
    } else {
        nH_max = N_target/N_cryst;
        nL_max = 0.0;
    }
    
    ScalarType nH = (1.0-softness)*nH_max+softness*nu;
    ScalarType nL = (1.0-softness)*nL_max+softness*nu;
    
    for (int i = 0; i < Norb; ++i) {
        if (backbone(i) > 0.5) {
            occs(i) = nH;
        } else {
            occs(i) = nL;
        }
    }
    
    return particle_statistic_sigma(pstat) * occs.cast<cScalarType>().asDiagonal();
}

cMatrixType trial_state(const lattice& lat, const int dim, std::mt19937& rng, ScalarType nu, const cMatrixType& siG, const ScalarType alpha, const ScalarType beta, const ScalarType softness, bool fluid){
    cMatrixType WC_intr(dim, dim);
    if(fluid==true){
        WC_intr = nu*cVectorType::Ones(dim).asDiagonal();
    }
    else if(nu<0.1){
        WC_intr = cMatrixType::Zero(dim,dim);
    }
    else if(nu<(1.0/3.0+1.0/4.0)/2.0){
        WC_intr = -IS_soft_wigner_filling(particle_statistic::FERMION,lat, triang_WC_types::ONEFOURTH, 1, nu, softness);
    }
    else if(nu<(1.0/3.0+1.0/2.0)/2.0){
        WC_intr = -IS_soft_wigner_filling(particle_statistic::FERMION,lat, triang_WC_types::ONETHIRD, 1, nu, softness);
    }
    else if(nu<(2.0/3.0+1.0/2.0)/2.0){
        WC_intr = -IS_soft_wigner_filling(particle_statistic::FERMION,lat, triang_WC_types::ONEHALF, 1, nu, softness);
    }
    else if(nu<(3.0/4.0+2.0/3.0)/2.0){
        WC_intr = -IS_soft_wigner_filling(particle_statistic::FERMION,lat, triang_WC_types::TWOTHIRDS, 1, nu, softness);
    }
    else if(nu<0.9){
        WC_intr = -IS_soft_wigner_filling(particle_statistic::FERMION,lat, triang_WC_types::THREEFOURTHS, 1, nu, softness);
    }
    else{
        WC_intr = cVectorType::Zero(dim).asDiagonal();
    }
    return alpha*siG+beta*WC_intr+(1.0-(alpha+beta))*(-IS_random(particle_statistic::FERMION, false, nu, dim, rng));
};


periodic_anderson_accel::periodic_anderson_accel(const int _m, const int _a, const int _b, const ScalarType _alpha, const ScalarType _alpha_start,  const int _dim) 
    : 
    m(_m),
    a(_a),
    b(_b),
    alpha(_alpha),
    alpha_start(_alpha_start),
    dim(_dim),
    P(cMatrixType::Zero(_dim*_dim,m)),
    G(cMatrixType::Zero(_dim*_dim,m)),
    D(cMatrixType::Zero(_dim*_dim,m)),
    qr(_dim*_dim,m-1),
    llt(m-1)
{
    lambda = 1e-7;
    head = 0;
    it_cnt = 0;
    err = 10000;
};

void periodic_anderson_accel::update_head(){
    head = (head+1)%m;
}

ScalarType periodic_anderson_accel::compute(cMatrixType& siG, const cMatrixType& n_siG){
    P.col(head) = siG.reshaped();
    G.col(head) = n_siG.reshaped();
    D.col(head).noalias() = n_siG.reshaped()-siG.reshaped();

    ScalarType loc_err = 0;

    if(m==1 || it_cnt==0 || (it_cnt+1)%(a+b)<b){
        loc_err = compute_linmix(siG, n_siG);
    }
    else{
        loc_err = compute_reg_aacc(siG);
    }
    update_head();
    it_cnt++;


    err = loc_err;
    return err/static_cast<ScalarType>(dim);
}
 
ScalarType periodic_anderson_accel::compute_linmix(cMatrixType& siG, const cMatrixType& n_siG){
    siG = (alpha_start*n_siG+(1.0-alpha_start)*siG);
    return D.col(head).norm();
}

ScalarType periodic_anderson_accel::compute_aacc(cMatrixType& siG){

    cMatrixType F(dim*dim,m);
    F.leftCols(head) = D.leftCols(head);
    F.rightCols(m-head) = D.rightCols(m-head);
    F.colwise() -= D.col(head);

    VectorType theta = VectorType::Zero(m);
   

    if(it_cnt<m+1){
        Eigen::ColPivHouseholderQR<cMatrixType> qr2(dim*dim,it_cnt);
        qr2.compute(F.leftCols(it_cnt));
        theta.head(it_cnt) = qr2.solve(-D.col(head)).real();
    }
    else{
        qr.compute(F);
        theta = qr.solve(-D.col(head)).real();
    }


    ScalarType alpha_head = 1.0-theta.sum();

    cVectorType P_new = P.col(head)*alpha_head+P.leftCols(head)*theta.head(head)+P.rightCols(m-head)*theta.tail(m-head);    
    cVectorType G_new = G.col(head)*alpha_head+G.leftCols(head)*theta.head(head)+G.rightCols(m-head)*theta.tail(m-head);
    cVectorType siG_vec;
    if(it_cnt<m){
        siG_vec = alpha_start*G_new+(1.0-alpha_start)*P_new;
    }
    else{
        siG_vec = alpha*G_new+(1.0-alpha)*P_new;
    }
    siG = siG_vec.reshaped(dim, dim);


    return D.col(head).norm();
}

ScalarType periodic_anderson_accel::compute_reg_aacc(cMatrixType& siG){

    cMatrixType F(dim*dim,m);
    F.leftCols(head) = D.leftCols(head);
    F.rightCols(m-head) = D.rightCols(m-head);
    F.colwise() -= D.col(head);

    VectorType theta = VectorType::Zero(m);

    // const ScalarType res = D.col(head).norm();
    // const ScalarType ratio = res/D.col(((head-1)+m)%m).norm(); // ratio of current and previous residual
    // constexpr ScalarType lambda_max = 1e-5;
    // constexpr ScalarType lambda_min = 1e-8;
    // if(ratio > 1.05){
    //     lambda *= 2.0;
    // }
    // else if(ratio < 0.95){
    //     lambda *= 0.5;
    // }
    // else{
    //     lambda *= 0.75;
    // }
    // lambda = std::clamp(lambda, lambda_min, lambda_max);

   
    if(it_cnt<m){
        Eigen::LLT<cMatrixType> llt2(it_cnt);
        cMatrixType FF_reg = F.leftCols(it_cnt).adjoint()*F.leftCols(it_cnt);
        FF_reg.diagonal().array() += lambda*D.col(head).squaredNorm();
        llt2.compute(FF_reg);
        theta.head(it_cnt) = llt2.solve(-F.leftCols(it_cnt).adjoint()*D.col(head)).real();
    }
    else{
        cMatrixType FF_reg = F.adjoint()*F;
        FF_reg.diagonal().array() += lambda*D.col(head).squaredNorm();
        llt.compute(FF_reg);
        theta = llt.solve(-F.adjoint()*D.col(head)).real();
    }

    ScalarType alpha_head = 1.0-theta.sum();

    cVectorType P_new = P.col(head)*alpha_head+P.leftCols(head)*theta.head(head)+P.rightCols(m-head)*theta.tail(m-head);    
    cVectorType G_new = G.col(head)*alpha_head+G.leftCols(head)*theta.head(head)+G.rightCols(m-head)*theta.tail(m-head);
    cVectorType siG_vec;
    if(it_cnt<m+1){
        siG_vec = alpha_start*G_new+(1.0-alpha_start)*P_new;
    }
    else{
        siG_vec = alpha*G_new+(1.0-alpha)*P_new;
    }
    siG = siG_vec.reshaped(dim, dim);


    return D.col(head).norm();
}

void periodic_anderson_accel::reset(){
    head = 0;
    it_cnt = 0;
    P.setZero();
    G.setZero();
    D.setZero();
}
