cmake_minimum_required(VERSION 3.16)

project(MicroGradCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(MicroGradCpp main.cpp

    value.h
    stdafx.h
    stdafx.cpp)

target_link_libraries(MicroGradCpp "${TORCH_LIBRARIES}")

# ----------------------------------------------------------------------
# Add the Precompiled Header Compiler Flags
# ----------------------------------------------------------------------
target_precompile_headers(MicroGradCpp PRIVATE
    # This instructs the compiler to use stdafx.h as the PCH file.
    # The list below must match the headers you included in stdafx.h
    <torch/torch.h>
)

include(GNUInstallDirs)
install(TARGETS MicroGradCpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
