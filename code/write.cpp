#include "write.hpp"



void csv_cM(const cMatrixType& M, const std::string fpath, const unsigned int N, const unsigned int ltype, const unsigned int ldim){
    std::ofstream out(fpath);
    const std::string delim = " ";
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open"+fpath);
    }
    out << N << delim << ltype << delim << ldim << "\n"; //header
    for(int i=0; i<M.rows(); i++){
        for(int j=0; j<M.cols(); j++){
            out << M(i,j);
            if(j!=M.cols()-1){
                out << delim;
            }
        }
        out << "\n";
    }
}


void save_binary(const std::string& path, const cMatrixType& M) {
    std::ofstream out(path, std::ios::binary);
    // Write dimensions first so Python knows the shape
    int rows = M.rows();
    int cols = M.cols();
    out.write(reinterpret_cast<char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<char*>(&cols), sizeof(int));
    // Write the raw buffer
    out.write(reinterpret_cast<const char*>(M.data()), M.size() * sizeof(cScalarType));
    out.close();
}

cMatrixType load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file for reading.");
    }

    int rows, cols;
    // Read the dimensions we saved at the beginning
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));

    cMatrixType M(rows, cols);
    
    // Read the raw buffer directly into the Eigen matrix
    in.read(reinterpret_cast<char*>(M.data()), rows * cols * sizeof(cScalarType));
    
    in.close();
    return M;
}



bool write_green_trajectory(const std::filesystem::path path, const std::vector<state_type>& iGt, const std::vector<ScalarType>& times){
                               
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    int32_t dim = static_cast<int32_t>(std::sqrt(iGt[0].rows()*iGt[0].cols())); //state_type is a vector, I just do rows*cols in case I switch to matrix state types
    int32_t time_steps = static_cast<int32_t>(iGt.size());


    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&time_steps), sizeof(time_steps));
    out.write(reinterpret_cast<const char*>(times.data()), times.size() * sizeof(ScalarType));
    for (const auto& state : iGt) {
        out.write(reinterpret_cast<const char*>(state.data()), 
                  state.size() * sizeof(cScalarType));
    }
    
    out.close();
    return true;
}


bool write_vector_trajectory(const std::filesystem::path path, const std::vector<VectorType>& vt, const std::vector<ScalarType>& times){
                               
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    int32_t dim = static_cast<int32_t>(vt[0].size()); //state_type is a vector, I just do rows*cols in case I switch to matrix state types
    int32_t time_steps = static_cast<int32_t>(vt.size());


    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&time_steps), sizeof(time_steps));
    out.write(reinterpret_cast<const char*>(times.data()), times.size() * sizeof(ScalarType));
    for (const auto& state : vt) {
        out.write(reinterpret_cast<const char*>(state.data()), 
                  state.size() * sizeof(ScalarType));
    }
    
    out.close();
    return true;
}

bool read_green_trajectory(const std::filesystem::path path, std::vector<state_type>& iGt, std::vector<ScalarType>& times){
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    int32_t dim;
    int32_t time_steps;

    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&time_steps), sizeof(time_steps));
    if (!in) return false;

    times.resize(time_steps);
    in.read(reinterpret_cast<char*>(times.data()), time_steps * sizeof(ScalarType));
    if (!in) return false;

    iGt.resize(time_steps);
    for (auto& state : iGt) {
        state.resize(dim*dim);
        in.read(reinterpret_cast<char*>(state.data()), dim*dim*sizeof(cScalarType));
        if (!in) return false;
    }

    return true;
}

bool read_green_trajectory(const std::filesystem::path path, std::vector<state_type>& iGt){
    std::vector<ScalarType> times;
    return read_green_trajectory(path, iGt, times);
}

bool write_scalar_trajectory(
    const std::filesystem::path& path,
    const std::vector<ScalarType>& ft,
    const std::vector<ScalarType>& times
)
{
    if (ft.size() != times.size())
        return false;

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        return false;

    int32_t time_steps = static_cast<int32_t>(ft.size());

    out.write(
        reinterpret_cast<const char*>(&time_steps),
        sizeof(time_steps)
    );

    out.write(
        reinterpret_cast<const char*>(times.data()),
        times.size() * sizeof(ScalarType)
    );

    out.write(
        reinterpret_cast<const char*>(ft.data()),
        ft.size() * sizeof(ScalarType)
    );

    return static_cast<bool>(out);
}

bool read_scalar_trajectory(
    const std::filesystem::path& path,
    std::vector<ScalarType>& ft,
    std::vector<ScalarType>& times
)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    int32_t time_steps;

    in.read(
        reinterpret_cast<char*>(&time_steps),
        sizeof(time_steps)
    );

    if (!in || time_steps < 0)
        return false;

    times.resize(static_cast<std::size_t>(time_steps));
    ft.resize(static_cast<std::size_t>(time_steps));

    in.read(
        reinterpret_cast<char*>(times.data()),
        times.size() * sizeof(ScalarType)
    );

    in.read(
        reinterpret_cast<char*>(ft.data()),
        ft.size() * sizeof(ScalarType)
    );

    if (!in)
        return false;

    return true;
}

bool write_meta(
    const std::filesystem::path path,
    const lattice_params& lat_parm,
    const simulation_params& sim_parm,
    const boson_params& bos_parm
)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;


    int32_t ltype;
    switch (lat_parm.ltype) {
        case lattice_type::LINEAR: ltype = 0; break;
        case lattice_type::SQUARE: ltype = 1; break;
        case lattice_type::TRIANGULAR: ltype = 2; break;
        default: return false;
    }

    int32_t bcond;
    switch (lat_parm.bcond) {
        case boundary_conditions::PERIODIC: bcond = 0; break;
        case boundary_conditions::OPEN: bcond = 1; break;
        default: return false;
    }



    int32_t lp[5] = {
        ltype,
        bcond,
        static_cast<int32_t>(lat_parm.N),
        static_cast<int32_t>(lat_parm.ldim),
        static_cast<int32_t>(lat_parm.Ns)
    };
    ScalarType sp[13] = {
        static_cast<ScalarType>(sim_parm.n_cool),
        static_cast<ScalarType>(sim_parm.m),
        sim_parm.alpha,
        sim_parm.t_0,
        sim_parm.t_end,
        sim_parm.t0_trap,
        sim_parm.tau_trap,
        sim_parm.t0_res,
        sim_parm.tau_res,
        sim_parm.del_t,
        sim_parm.atol,
        sim_parm.rtol,
        sim_parm.max_stepsize 
    };


    ScalarType bp[13]{
        static_cast<ScalarType>(bos_parm.sigma),
        static_cast<ScalarType>(bos_parm.site_dof),
        bos_parm.Np,
        bos_parm.j,
        bos_parm.u,
        bos_parm.v,
        bos_parm.g,
        bos_parm.p,
        bos_parm.rt,
        bos_parm.ut,
        bos_parm.beta,
        bos_parm.mu_eq,
        bos_parm.mu
    };

    out.write(reinterpret_cast<const char*>(lp), sizeof(lp));
    out.write(reinterpret_cast<const char*>(sp), sizeof(sp));
    out.write(reinterpret_cast<const char*>(bp), sizeof(bp));
    
    out.close();

    return static_cast<bool>(out);
}


bool write_meta(
    const std::filesystem::path path,
    const lattice_params& lat_parm,
    const simulation_params& sim_parm,
    const fermion_params& ferm_parm
)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;


    int32_t ltype;
    switch (lat_parm.ltype) {
        case lattice_type::LINEAR: ltype = 0; break;
        case lattice_type::SQUARE: ltype = 1; break;
        case lattice_type::TRIANGULAR: ltype = 2; break;
        default: return false;
    }

    int32_t bcond;
    switch (lat_parm.bcond) {
        case boundary_conditions::PERIODIC: bcond = 0; break;
        case boundary_conditions::OPEN: bcond = 1; break;
        default: return false;
    }



    int32_t lp[5] = {
        ltype,
        bcond,
        static_cast<int32_t>(lat_parm.N),
        static_cast<int32_t>(lat_parm.ldim),
        static_cast<int32_t>(lat_parm.Ns)
    };

    ScalarType sp[13] = {
        static_cast<ScalarType>(sim_parm.n_cool),
        static_cast<ScalarType>(sim_parm.m),
        sim_parm.alpha,
        sim_parm.t_0,
        sim_parm.t_end,
        sim_parm.t0_trap,
        sim_parm.tau_trap,
        sim_parm.t0_res,
        sim_parm.tau_res,
        sim_parm.del_t,
        sim_parm.atol,
        sim_parm.rtol,
        sim_parm.max_stepsize 
    };

    ScalarType fp[11]{
        static_cast<ScalarType>(ferm_parm.sigma),
        static_cast<ScalarType>(ferm_parm.site_dof),
        ferm_parm.filling,
        ferm_parm.j,
        ferm_parm.u,
        ferm_parm.v,
        ferm_parm.g,
        ferm_parm.q_scr,
        ferm_parm.beta,
        ferm_parm.mu_eq,
        ferm_parm.mu
    };

    out.write(reinterpret_cast<const char*>(lp), sizeof(lp));
    out.write(reinterpret_cast<const char*>(sp), sizeof(sp));
    out.write(reinterpret_cast<const char*>(fp), sizeof(fp));
        

    return static_cast<bool>(out);
}

bool read_meta(
    const std::filesystem::path& path,
    lattice_params& lat_parm,
    simulation_params& sim_parm,
    boson_params& bos_parm
)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    int32_t lp[5];
    ScalarType sp[13];
    ScalarType bp[13];

    in.read(reinterpret_cast<char*>(lp), sizeof(lp));
    in.read(reinterpret_cast<char*>(sp), sizeof(sp));
    in.read(reinterpret_cast<char*>(bp), sizeof(bp));

    if (!in) return false;

    // Lattice parameters
    switch (lp[0]) {
        case 0:
            lat_parm.ltype = lattice_type::LINEAR;
            break;
        case 1:
            lat_parm.ltype = lattice_type::SQUARE;
            break;
        case 2:
            lat_parm.ltype = lattice_type::TRIANGULAR;
            break;
        default:
            return false;
    }

    switch (lp[1]) {
        case 0:
            lat_parm.bcond = boundary_conditions::PERIODIC;
            break;
        case 1:
            lat_parm.bcond = boundary_conditions::OPEN;
            break;
        default:
            return false;
    }

    lat_parm.N    = static_cast<decltype(lat_parm.N)>(lp[2]);
    lat_parm.ldim = static_cast<decltype(lat_parm.ldim)>(lp[3]);
    lat_parm.Ns   = static_cast<decltype(lat_parm.Ns)>(lp[4]);


    // Simulation parameters
    sim_parm.n_cool       = static_cast<decltype(sim_parm.n_cool)>(sp[0]);
    sim_parm.m            = static_cast<decltype(sim_parm.m)>(sp[1]);
    sim_parm.alpha        = sp[2];
    sim_parm.t_0          = sp[3];
    sim_parm.t_end        = sp[4];
    sim_parm.t0_trap      = sp[5];
    sim_parm.tau_trap     = sp[6];
    sim_parm.t0_res       = sp[7];
    sim_parm.tau_res      = sp[8];
    sim_parm.del_t        = sp[9];
    sim_parm.atol         = sp[10];
    sim_parm.rtol         = sp[11];
    sim_parm.max_stepsize = sp[12];


    // Boson parameters
    bos_parm.sigma    = static_cast<decltype(bos_parm.sigma)>(bp[0]);
    bos_parm.site_dof = static_cast<decltype(bos_parm.site_dof)>(bp[1]);
    bos_parm.Np       = bp[2];
    bos_parm.j        = bp[3];
    bos_parm.u        = bp[4];
    bos_parm.v        = bp[5];
    bos_parm.g        = bp[6];
    bos_parm.p        = bp[7];
    bos_parm.rt       = bp[8];
    bos_parm.ut       = bp[9];
    bos_parm.beta     = bp[10];
    bos_parm.mu_eq    = bp[11];
    bos_parm.mu       = bp[12];

    return true;
}


bool read_meta(
    const std::filesystem::path& path,
    lattice_params& lat_parm,
    simulation_params& sim_parm,
    fermion_params& ferm_parm
)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    int32_t lp[5];
    ScalarType sp[13];
    ScalarType fp[11];

    in.read(reinterpret_cast<char*>(lp), sizeof(lp));
    in.read(reinterpret_cast<char*>(sp), sizeof(sp));
    in.read(reinterpret_cast<char*>(fp), sizeof(fp));

    if (!in) return false;

    // Lattice parameters
    switch (lp[0]) {
        case 0:
            lat_parm.ltype = lattice_type::LINEAR;
            break;
        case 1:
            lat_parm.ltype = lattice_type::SQUARE;
            break;
        case 2:
            lat_parm.ltype = lattice_type::TRIANGULAR;
            break;
        default:
            return false;
    }

    switch (lp[1]) {
        case 0:
            lat_parm.bcond = boundary_conditions::PERIODIC;
            break;
        case 1:
            lat_parm.bcond = boundary_conditions::OPEN;
            break;
        default:
            return false;
    }

    lat_parm.N    = static_cast<decltype(lat_parm.N)>(lp[2]);
    lat_parm.ldim = static_cast<decltype(lat_parm.ldim)>(lp[3]);
    lat_parm.Ns   = static_cast<decltype(lat_parm.Ns)>(lp[4]);


    // Simulation parameters
    sim_parm.n_cool       = static_cast<decltype(sim_parm.n_cool)>(sp[0]);
    sim_parm.m            = static_cast<decltype(sim_parm.m)>(sp[1]);
    sim_parm.alpha        = sp[2];
    sim_parm.t_0          = sp[3];
    sim_parm.t_end        = sp[4];
    sim_parm.t0_trap      = sp[5];
    sim_parm.tau_trap     = sp[6];
    sim_parm.t0_res       = sp[7];
    sim_parm.tau_res      = sp[8];
    sim_parm.del_t        = sp[9];
    sim_parm.atol         = sp[10];
    sim_parm.rtol         = sp[11];
    sim_parm.max_stepsize = sp[12];


    // Fermion parameters
    ferm_parm.sigma    = static_cast<decltype(ferm_parm.sigma)>(fp[0]);
    ferm_parm.site_dof = static_cast<decltype(ferm_parm.site_dof)>(fp[1]);
    ferm_parm.filling  = fp[2];
    ferm_parm.j        = fp[3];
    ferm_parm.u        = fp[4];
    ferm_parm.v        = fp[5];
    ferm_parm.g        = fp[6];
    ferm_parm.q_scr    = fp[7];
    ferm_parm.beta     = fp[8];
    ferm_parm.mu_eq    = fp[9];
    ferm_parm.mu       = fp[10];

    return true;
}

bool read_meta(
    const std::filesystem::path& path,
    fermion_params& ferm_parm
)
{
    lattice_params lat_dummy{boundary_conditions::PERIODIC, lattice_type::LINEAR, 1};
    simulation_params sim_dummy{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return read_meta(path, lat_dummy, sim_dummy, ferm_parm);
}

bool read_meta(
    const std::filesystem::path& path,
    boson_params& bos_parm
)
{
    lattice_params lat_dummy{boundary_conditions::PERIODIC, lattice_type::LINEAR, 1};
    simulation_params sim_dummy{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return read_meta(path, lat_dummy, sim_dummy, bos_parm);
}