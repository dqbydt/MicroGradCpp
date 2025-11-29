cmake_minimum_required(VERSION 3.19)
project(MicroGradCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# === Precompiled headers source (stdafx.cpp + stdafx.h) ===
add_library(pch_obj OBJECT stdafx.cpp)
target_link_libraries(pch_obj PUBLIC "${TORCH_LIBRARIES}")

# === Main app ===
add_executable(MicroGradCpp main.cpp value.h)
target_link_libraries(MicroGradCpp PRIVATE pch_obj "${TORCH_LIBRARIES}")
# Apply PCH to the real executable
target_precompile_headers(MicroGradCpp PRIVATE <torch/torch.h>)

# === Catch2 ===
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# === Tests ===
add_executable(mgcpp_tests tests/test_value.cpp value.h)
target_link_libraries(mgcpp_tests PRIVATE pch_obj Catch2::Catch2WithMain "${TORCH_LIBRARIES}")
# Apply PCH on the test executable also
target_precompile_headers(mgcpp_tests PRIVATE <torch/torch.h>)

# === Testing ===
enable_testing()
include(CTest)
add_test(NAME mgcpp_tests COMMAND mgcpp_tests)

# === Uncomment for AUTOMATIC DLL COPY ===
# Needed to run exe in external terminal
#if(WIN32)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    foreach(exe MicroGradCpp mgcpp_tests)
#        add_custom_command(TARGET ${exe} POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                ${TORCH_DLLS}
#                "$<TARGET_FILE_DIR:${exe}>"
#            COMMENT "Copying libtorch DLLs → ${exe} output folder"
#            VERBATIM
#        )
#    endforeach()
#endif()

install(TARGETS MicroGradCpp mgcpp_tests
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
