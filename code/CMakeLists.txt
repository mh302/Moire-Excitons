cmake_minimum_required(VERSION 3.24)
project(kbe_timestepper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
# find_package(FFTW3 CONFIG REQUIRED)
find_package(GSL REQUIRED)

# Prefer pkg-config, which works with the cluster FFTW installation.
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3_PC QUIET IMPORTED_TARGET fftw3)
endif()

if(TARGET PkgConfig::FFTW3_PC)
    set(FFTW3_TARGET PkgConfig::FFTW3_PC)
    message(STATUS "FFTW3 found through pkg-config")
else()
    # Fallback for systems such as a desktop installation providing
    # FFTW3Config.cmake, for example through a package manager.
    find_package(FFTW3 CONFIG REQUIRED)
    set(FFTW3_TARGET FFTW3::fftw3)
    message(STATUS "FFTW3 found through its CMake package")
endif()


set(SOURCE_FILES
    main.cpp
    model.cpp
    write.cpp
    initial_state.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

get_filename_component(PROJECT_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_ROOT_COMPILE="${PROJECT_ROOT_PATH}")


option(ENABLE_SANITIZERS "Enable Clang/GCC sanitizers" OFF)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        /permissive-

        $<$<CONFIG:Debug>:
            /Od
            /Zi
            /bigobj
        >

        $<$<CONFIG:Release>:
            /O2
            /arch:AVX2
            /DNDEBUG
            # /fp:fast
        >
    >

    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wpedantic

        $<$<CONFIG:Debug>:
            -O0
            -g3
        >

        $<$<CONFIG:Release>:
            -O3
            -march=native
            -DNDEBUG
            # -ffast-math
        >
    >
)

if(
    ENABLE_SANITIZERS
    AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang"
)
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O1
        -g
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize=address,undefined
        -fno-sanitize-recover=undefined
    )

    target_link_options(${PROJECT_NAME} PRIVATE
        -fsanitize=address,undefined
    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    Boost::boost
    OpenMP::OpenMP_CXX
    # FFTW3::fftw3
    ${FFTW3_TARGET}
    GSL::gsl
    GSL::gslcblas
) 


target_precompile_headers(${PROJECT_NAME} PRIVATE
    functions.hpp
    write.hpp
    initial_state.hpp
)



include(CTest)
enable_testing()

message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")