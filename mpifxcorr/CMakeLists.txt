cmake_minimum_required(VERSION 3.13)
project(mpifxcorr LANGUAGES CUDA CXX)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)
find_package(CUDA  REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# Generate maximum debug information (Dwarf version 4 for folly backtrace support)
set(CMAKE_CXX_FLAGS_DEBUG " -g -O2 -rdynamic -gdwarf-4 -fpermissive -w ")
set(CMAKE_C_FLAGS_DEBUG " -g -O2 -rdynamic -gdwarf-4 -fpermissive -w ")

include_directories(
        build
        build/src
        /opt/intel/oneapi/ipp/latest/include
        ../install/include
        .
        ../libraries/mark5access
        ../libraries/mark6sg
        ../libraries/vdifio/src

        "${CUDA_INCLUDE_DIRS}"
)

link_directories(
        /opt/intel/oneapi/ipp/latest/lib/intel64
        ../install/lib
)

# Define the source files
set(
        SOURCE_FILES
        src/alert.cpp
        src/alert.h
        src/configuration.cpp
        src/configuration.h
        src/core.cpp
        src/core.h
        src/cpumode.cpp
        src/cpumode.h
        src/datamuxer.cpp
        src/datamuxer.h
        src/datastream.cpp
        src/datastream.h
        src/fxmanager.cpp
        src/fxmanager.h
        src/gpucore.cu
        src/gpucore.cuh
        src/gpumode.cu
        src/gpumode.cuh
        src/gpumode_kernels.cuh
        src/mpifxcorr.cpp
        src/mpifxcorr.h
        src/mark5bfile.cpp
        src/mark5bfile.h
        src/mark5bmark5_stubs.cpp
        src/mathutil.cpp
        src/mathutil.h
        src/mk5.cpp
        src/mk5.h
        src/mk5mode.cpp
        src/mk5mode.h
        src/mk5mode_gpu.cpp
        src/mk5mode_gpu.h
        src/mode.cpp
        src/mode.h
        src/model.cpp
        src/model.h
        src/mpifxcorr.h
        src/mpifxcorr.cpp
        src/nativemk5_stubs.cpp
        src/pcal.cpp
        src/pcal.h
        src/pcal_impl.h
        src/polyco.cpp
        src/polyco.h
        src/pthreadbarrier_osx.cpp
        src/pthreadbarrier_osx.h
        src/switchedpower.cpp
        src/switchedpower.h
        src/sysutil.cpp
        src/sysutil.h
        src/vdiffake.cpp
        src/vdiffake.h
        src/vdiffile.cpp
        src/vdiffile.h
        src/vdifnetwork.cpp
        src/vdifnetwork.h
        src/visibility.cpp
        src/visibility.h
)

# Set the main executable and source files
add_executable(
        mpifxcorr
        ${SOURCE_FILES}
)

# Set the link libraries
set(
        LINK_LIBRARIES
        difxmessage
        vdifio
        mark5access

        cuda
        cudart
        cufft

        ippse9
        ippvme9
        ippcore

        mpi
        mpi_cxx
)

set_property(TARGET mpifxcorr PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(mpifxcorr ${LINK_LIBRARIES})
add_definitions(-DIPP9)