#pragma once 
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include "types.hpp"

template <typename T1, typename T2> void csv_xy(const T1& x, const T2& y, const std::string& fpath){
    std::ofstream out(fpath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }
    out << "#x,y\n";
    for(auto it_x = std::begin(x), it_y = std::begin(y); it_x != std::end(x) && it_y != std::end(y); ++it_x, ++it_y){
        out << *it_x << "," << *it_y << "\n";
    }
}

void csv_cM(const cMatrixType& M, const std::string fpath, const unsigned int N, const unsigned int ltype, const unsigned int ldim);

void save_binary(const std::string& path, const cMatrixType& M);

cMatrixType load_binary(const std::string& path);

bool write_green_trajectory(const std::filesystem::path path, const std::vector<state_type>& iGt, const std::vector<ScalarType>& times);
bool write_vector_trajectory(const std::filesystem::path path, const std::vector<VectorType>& vt, const std::vector<ScalarType>& times);

bool read_green_trajectory(const std::filesystem::path path, std::vector<state_type>& iGt, std::vector<ScalarType>& times);
bool read_green_trajectory(const std::filesystem::path path, std::vector<state_type>& iGt);

bool write_scalar_trajectory(const std::filesystem::path& path, const std::vector<ScalarType>& ft, const std::vector<ScalarType>& times);
bool read_scalar_trajectory(const std::filesystem::path& path, std::vector<ScalarType>& ft, std::vector<ScalarType>& times);

bool write_meta(
    const std::filesystem::path path,
    const lattice_params& lat_parm,
    const simulation_params& sim_parm,
    const fermion_params& ferm_parm
);

bool write_meta(
    const std::filesystem::path path,
    const lattice_params& lat_parm,
    const simulation_params& sim_parm,
    const boson_params& bos_parm
);

bool read_meta(
    const std::filesystem::path& path,
    lattice_params& lat_parm,
    simulation_params& sim_parm,
    fermion_params& ferm_parm
);

bool read_meta(
    const std::filesystem::path& path,
    lattice_params& lat_parm,
    simulation_params& sim_parm,
    boson_params& bos_parm
);

bool read_meta(
    const std::filesystem::path& path,
    fermion_params& ferm_parm
);

bool read_meta(
    const std::filesystem::path& path,
    boson_params& bos_parm
);