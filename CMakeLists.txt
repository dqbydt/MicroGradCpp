cmake_minimum_required(VERSION 3.16)

project(MicroGradCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MicroGradCpp main.cpp

    value.h)

include(GNUInstallDirs)
install(TARGETS MicroGradCpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
