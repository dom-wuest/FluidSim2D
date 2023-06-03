cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(FluidSimulation2D)

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

add_shader(${PROJECT_NAME} shaders/shader.comp)

target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan glfw glm)
target_compile_definitions(${PROJECT_NAME} PRIVATE)

