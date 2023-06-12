cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(FluidSimulation2D)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(current-output-path ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}.spv)

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})

    add_custom_command(
           OUTPUT ${current-output-path}
           COMMAND ${GLSLC} --target-env=vulkan1.0 -o ${current-output-path} ${current-shader-path}
           DEPENDS ${current-shader-path}
           IMPLICIT_DEPENDS CXX ${current-shader-path}
           VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_shader)

add_subdirectory(libs)

find_package(Vulkan REQUIRED)

set( SOURCE_FILE 
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})

add_shader(${PROJECT_NAME} shaders/advectVelocity.comp)
add_shader(${PROJECT_NAME} shaders/displayVelocity.comp)
add_shader(${PROJECT_NAME} shaders/extrapolateU.comp)
add_shader(${PROJECT_NAME} shaders/extrapolateV.comp)

target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan glfw glm)
target_compile_definitions(${PROJECT_NAME} PRIVATE)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

