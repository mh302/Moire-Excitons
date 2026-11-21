#include <sys/stat.h>
#include <cmath>
#include <typeinfo>
#include <complex>
#include <random>
#include <chrono>
// #include <unsupported/Eigen/KroneckerProduct>
#include "model.hpp"
#include "write.hpp"
#include "functions.hpp"
#include "initial_state.hpp"
#include "steppers.hpp"
#include "observers.hpp"

const std::filesystem::path PROJECT_ROOT = PROJECT_ROOT_COMPILE;
uint32_t base_seed = 12;


void mu_sweep(){

  const lattice_params lp(
    boundary_conditions::PERIODIC,
    lattice_type::TRIANGULAR,
    30 //N
  );

  fermion_params fp(
    1, //site_dof
    1.0/3.0, //filling
    0.1, //j
    0.0, //u
    1.0, //v
    0.0, //gamma 
    2.0, //q_scr
    1.0/0.6 //beta //not bad: 1.0/1.5
  );

  tVModel f_mod(lp.ltype, lp.bcond, lp.N, fp.j, fp.v, fp.g);

  simulation_params sp(
    1, //n_cool
    1, //m (aacc)
    1.0, //alpha (aacc)
    1.0, //t0
    1.0, //t_end
    1.0, //t0_trap
    1.0, // tau_trap
    1.0, //t0_res
    1.0, //tau_res
    1.0, //del_t
    1.0, //atol
    1.0, //rtol
    1.0 //max_stepsize
  );
  

  auto updateSHF = [&V = f_mod.V](cMatrixType& SHF, const cMatrixType& siG){
    SHF = -V.cwiseProduct(siG); //Fock 
    SHF.diagonal() += V*siG.diagonal(); //Hartree
  };

  auto compute_GCP = [h=f_mod.H,V=f_mod.V, gam=fp.g, dim=f_mod.dim](const cMatrixType& siG, ScalarType beta, ScalarType mu){
    ScalarType Omega;
    cMatrixType Sigma_HF = -V.cwiseProduct(siG); //Fock 
    Sigma_HF.diagonal() += V*siG.diagonal(); //Hartree
    Omega = -0.5*(siG*Sigma_HF).trace().real();

    cMatrixType HHF_M = h+Sigma_HF;
    HHF_M.diagonal().array() -= mu;
    Omega += 0.5*HHF_M.trace().real();

    auto lngamterm = [&beta, &gam](ScalarType e){
      cScalarType a = 0.5*(1.0+1i*beta/std::numbers::pi*(e-1i*gam));
      return 2.0*lngamma(a).real();
    };

    Eigen::SelfAdjointEigenSolver<cMatrixType> es(HHF_M, Eigen::EigenvaluesOnly);
    VectorType evals = es.eigenvalues();
    Omega += evals.unaryExpr(lngamterm).sum()/beta;

    const ScalarType ln2pi = std::log(2.0*std::numbers::pi);
    Omega -= dim*ln2pi/beta;
    return Omega;
  };


  std::random_device rd;//change the seed of srand such that eigen3s ::Random produces different matrices each run
  std::srand(rd());//now you can see shifted Wigner Crystals 

  ScalarType mu_start = 0.0;
  ScalarType mu_stop = 7.0;
  int N_mu = 50;
  VectorType temp = VectorType::LinSpaced(N_mu, mu_start, mu_stop);
  std::vector<ScalarType> mus(temp.data(), temp.data()+N_mu);
  std::vector<ScalarType> nus{};


  std::vector<state_type> iGnu;
  cMatrixType siG = cMatrixType::Zero(f_mod.dim, f_mod.dim);
  std::vector<int> not_converged{};
  std::vector<ScalarType> converged{};


  //anderson accel parameters
  const int maxit_sc = 1000;
  const anderson_accel_params aacc_ps(//optimistic
    3, //m
    2, //aacc period 
    1, //linmix period
    0.85, //alpha
    0.05 //alpha_start
  );

  ScalarType last_conv_filling = 0.2;

  //book keeping for the random initial guesses
  int N_trials = 40;
  std::vector<cMatrixType> siG_trials(N_trials);
  std::vector<unsigned char> try_trials(N_trials);//apparently std::vector<bool> not theadsafe
  std::vector<ScalarType> GCP_trials(N_trials);


  for(int it=0; it<N_mu; ++it){
    #pragma omp parallel for
    for(int n=0;n<N_trials;++n){
      std::seed_seq seed{base_seed,static_cast<uint32_t>(it),static_cast<uint32_t>(n)};   
      std::mt19937 rng(seed);
      int itcnt_trial = 0;
      siG_trials[n] = trial_state(f_mod.lat, f_mod.dim, rng, last_conv_filling, cMatrixType::Zero(f_mod.dim, f_mod.dim),0.0, 0.8, 0.0, false);
      bool try_current = solve_Eq_MUfixed_SHFsc(particle_statistic::FERMION, fp.g, temp[it], updateSHF, f_mod.H,
                                  siG_trials[n], fp.beta, itcnt_trial, maxit_sc, aacc_ps, f_mod.dim);
      try_trials[n] = static_cast<unsigned char>(try_current);
      if(try_current){
        GCP_trials[n] = compute_GCP(siG_trials[n], fp.beta, temp[it]);
      }
      else {//not neccessary because only converged iterations are compared
        GCP_trials[n] = std::numeric_limits<ScalarType>::max();
      }
    }
    bool try_is = false;
    ScalarType GCP_min = std::numeric_limits<ScalarType>::max();
    int n_min = -1;
    for(int n=0;n<N_trials;++n){
      if(try_trials[n]!=0){
        if(GCP_trials[n]<=GCP_min){
          GCP_min = GCP_trials[n];
          n_min = n;
          try_is = true;
        }
      }
    }
    
    if(try_is){
      siG = siG_trials[n_min];
      last_conv_filling = siG.real().trace()/f_mod.dim;
      nus.push_back(last_conv_filling);
      iGnu.push_back(-siG.reshaped());

      converged.push_back(static_cast<ScalarType>(it));
      std::cout << "it = " << it+1 << "/" << N_mu << "\n";

    }
    else{
      not_converged.push_back(it);
      std::cout << "it = " << it+1 << "/" << N_mu << " not converged."<< "\n";
    }
  }

  for (auto it = not_converged.rbegin(); it!=not_converged.rend(); ++it) {//delete not converged iterations
    mus.erase(mus.begin()+*it);
    std::cout << *it << "\n";
  }
  std::cout << "number not converged it's: " << not_converged.size() << "\n";
  write_meta(PROJECT_ROOT/"data/f_meta.txt", lp, sp, fp);
  write_green_trajectory(PROJECT_ROOT/"data/f_G.txt", iGnu, nus);
  write_scalar_trajectory(PROJECT_ROOT/"data/converged_its.txt", converged, converged);
  write_scalar_trajectory(PROJECT_ROOT/"data/mus_nus.txt", mus, nus);
}

void solve_bfKBE_markov(int idx, std::string folder){

  constexpr lattice_params lp(
    boundary_conditions::OPEN,
    lattice_type::TRIANGULAR,
    30 //N
  );

  constexpr lattice_params f_lp(
    boundary_conditions::PERIODIC,
    lattice_type::TRIANGULAR,
    30 //N
  );
  boson_params bp(
    2, //site dof
    11.0, //Np
    1.0, //j
    0.1, //u
    0.12, //v
    0.1, //gamma
    3.0, //p
    4.0, //rt
    1e3, //ut
    1.0 //beta
  );
  
  const MatrixType j{{bp.j/10.0,0},{0,bp.j}};
  const MatrixType u{{bp.u,bp.u},{bp.u,bp.u}};
  const MatrixType v{{bp.v,bp.v},{bp.v,bp.v}};
  const MatrixType g{{bp.g,0},{0,3*bp.g}};
  const VectorType delta{{0.0, 20.0}};
  const ScalarType J_12 = 1.0*bp.j;
  const ScalarType U_xe = 20.0;
  

  bHubbardModel b_mod(lp.ltype, lp.bcond, lp.N, j, delta, u, v, g, static_cast<int>(bp.p), bp.rt, bp.ut);
  cMatrixType b_heff = b_mod.Heff_NI;
  for(int i=0;i<b_mod.lat.Ns;++i){//interband onsite hopping
      b_heff(2*i+1,2*i) -= J_12;
      b_heff(2*i,2*i+1) -= J_12;
  }

  simulation_params sp(
    1, //n_cool
    6, //m (aacc)
    0.3/bp.j, //alpha (aacc)
    0.0/bp.j, //t0
    3.0/bp.j, //t_end
    -100.0/bp.j, //t0_trap
    1.0/bp.j, // tau_trap
    std::numeric_limits<ScalarType>::quiet_NaN(), //t0_res
    std::numeric_limits<ScalarType>::quiet_NaN(), //tau_res
    100.0/bp.j, //del_t
    1e-4, //atol
    1e-4, //rtol,
    0.1 //max_stepsize
  );
  fermion_params fp(1,1.0,1.0,1.0,1.0,1.0,1.0,1.0);
  read_meta(PROJECT_ROOT/"data"/folder/"f_meta.txt", fp);
  tVModel f_mod(lp.ltype, boundary_conditions::PERIODIC, lp.N, fp.j, fp.v, fp.g);

  //Effective single-particle Hamiltonian, including mean-field interacitons and decay
  auto update_H = 
  [&b_heff, &V_xx = b_mod.V, &f_h = f_mod.Heff_NI, &V_ee = f_mod.V, &Ns = lp.Ns, &U_xe]
  (cMatrixType& b_Heff, cMatrixType& f_Heff, const cMatrixType& b_iG, const cMatrixType& f_iG, const ScalarType t){
    //BOSONS-BOSON
    b_Heff = b_heff;
    b_Heff += V_xx.cwiseProduct(b_iG);
    b_Heff.diagonal() += V_xx*b_iG.diagonal();
    //FERMIONS-FERMION
    f_Heff = f_h;
    f_Heff.diagonal() -= V_ee*f_iG.diagonal(); //Hartree
    f_Heff += V_ee.cwiseProduct(f_iG); //Fock
    //BOSON-FERMION
    for(int i=0;i<Ns;++i){
      cScalarType temp = U_xe*(-f_iG(i,i));
      b_Heff(2*i,2*i) += temp;
      temp = U_xe*(b_iG(2*i,2*i));
      f_Heff(i,i) += temp;
    }   
  };
  //initial states
  std::vector<state_type> iGmu;
  std::vector<ScalarType> nus;
  std::vector<ScalarType> mus;
  read_green_trajectory(PROJECT_ROOT/"data"/folder/"f_G.txt", iGmu);
  read_scalar_trajectory(PROJECT_ROOT/"data"/folder/"mus_nus.txt", mus, nus);
  //BOSONS
  cMatrixType b_iG0 = IS_uncorrelated_trap(particle_statistic::BOSON, b_mod.lat, bp.site_dof, bp.rt, 1.0);
  //FERMIONS
  cMatrixType f_iG0 = cMatrixType::Zero(f_mod.dim,f_mod.dim);
  f_iG0 = iGmu[idx].reshaped(f_mod.dim,f_mod.dim);
  fp.mu_eq = mus[idx];
  fp.mu = fp.mu_eq;


  SystemBF_markov sys(update_H, b_mod.dim, f_mod.dim);
  diameter_observer<lp.ltype> obs(particle_statistic::BOSON, lp.N, b_mod.M, b_iG0);

  int b_dxd = b_mod.dim*b_mod.dim;
  int f_dxd = f_mod.dim*f_mod.dim;
  state_type x = state_type::Zero(b_dxd+f_dxd);
  x.head(b_dxd) = b_iG0.reshaped();
  x.tail(f_dxd) = f_iG0.reshaped();

  auto constep = boost::numeric::odeint::make_controlled(sp.atol, sp.rtol, sp.max_stepsize, error_stepper_type{});

  ScalarType t = sp.t_0;
  ScalarType dt  = 1e-3;

  int it_cnt = 0;

  std::vector<VectorType> b_occs{b_iG0.diagonal().real()};
  std::vector<VectorType> f_occs{-f_iG0.diagonal().real()};
  std::vector<ScalarType> b_diams{0};
  std::vector<ScalarType> times{0};

  std::string file_end = std::to_string(idx);
  write_meta(PROJECT_ROOT/"data"/folder/("b_meta_"+file_end+".txt"), lp, sp, bp);
  write_meta(PROJECT_ROOT/"data"/folder/("f_meta_"+file_end+".txt"), lp, sp, fp);
  
  while (t < sp.t_end){
    if (t + dt > sp.t_end) {dt = sp.t_end - t;}
    auto result = constep.try_step(sys, x, t, dt);
    if (result == boost::numeric::odeint::success){
      Eigen::Map<const cMatrixType> b_iG(x.data(), b_mod.dim, b_mod.dim);      
      Eigen::Map<const cMatrixType> f_iG(x.data()+b_dxd, f_mod.dim, f_mod.dim);
      it_cnt++;
      if(it_cnt%1==0){
        std::cout << "t = " << t <<"\n"; 
        ScalarType d = obs.compute_radius(Eigen::Map<const cMatrixType>(x.data(), b_mod.dim, b_mod.dim));
        b_diams.push_back(d);
        b_occs.push_back(b_iG.diagonal().real());
        f_occs.push_back(-f_iG.diagonal().real());
        times.push_back(t);
      }
    }
  }
  write_vector_trajectory(PROJECT_ROOT/"data"/folder/("b_occs_"+file_end+".txt"), b_occs, times);
  write_vector_trajectory(PROJECT_ROOT/"data"/folder/("f_occs_"+file_end+".txt"), f_occs, times);
  write_scalar_trajectory(PROJECT_ROOT/"data"/folder/("b_diams_"+file_end+".txt"), b_diams, times);
}


int main(int argc,char *argv[]){
  
  mu_sweep(); //determines fermionic initial states for different chemical potentials (fillings)

  //The time evolution of the Bose-Fermi mixture (for all fillings)
  std::vector<ScalarType> conv_its{};
  std::vector<ScalarType> temp{};
  read_scalar_trajectory(PROJECT_ROOT/"data/converged_its.txt", conv_its, temp);
  for(int it=0; it<temp.size();++it){
    solve_bfKBE_markov(static_cast<int>(conv_its[it]), "");
    std::cout << it+1 << "/" << temp.size() << "\n";
  }

  return 0;

}

