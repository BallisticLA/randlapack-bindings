# funNystrom++ benchmark — external baselines via FetchContent.
#
# The benchmark compares RandLAPACK's FunNystromPP against external reference
# implementations. Those are NOT vendored in this repository; they are fetched
# at CMake-configure time, pinned to a commit each for reproducibility. The
# sources are MATLAB (.m), so there is no build step — FetchContent only
# downloads them, and we write their locations into external_paths.m, which the
# benchmark scripts addpath().
#
# Acknowledgements / licenses (see ../../../THIRD_PARTY_LICENSES.md):
#   Epperly  https://github.com/eepperly/Ethan-Epperly-Thesis   MIT
#            XTrace / XNysTrace / resphering trace estimators.
#   Persson  https://github.com/davpersson/funNystrom           license pending
#            funNystrom++ reference (nystrom.m, funnystrompp.m, block_lanczos.m).
#            No license file at time of writing: used here with attribution,
#            fetched (not redistributed); a license is being requested from the
#            author. block_lanczos.m therein is Meyer's HutchPlusPlus.

include(FetchContent)

FetchContent_Declare(epperly_trace
    GIT_REPOSITORY https://github.com/eepperly/Ethan-Epperly-Thesis.git
    GIT_TAG        2d33ee40577caff18b16fed3454a147d08cbe16e)   # main, 2026-06
FetchContent_Declare(persson_funnystrom
    GIT_REPOSITORY https://github.com/davpersson/funNystrom.git
    GIT_TAG        0427bde1e3c238ce53cc184600482481490c2a48)   # master, 2026-06

# Download only (no add_subdirectory: these are MATLAB sources, no CMake build).
FetchContent_MakeAvailable(epperly_trace persson_funnystrom)

set(EPPERLY_DIR "${epperly_trace_SOURCE_DIR}")
set(PERSSON_DIR "${persson_funnystrom_SOURCE_DIR}")
configure_file(external_paths.m.in
               "${CMAKE_CURRENT_SOURCE_DIR}/external_paths.m" @ONLY)

message(STATUS "funNystrom++ benchmark baselines fetched (not vendored):")
message(STATUS "  Epperly (MIT):              ${EPPERLY_DIR}")
message(STATUS "  Persson (license pending):  ${PERSSON_DIR}")
message(STATUS "  -> external_paths.m written for the benchmark scripts")
