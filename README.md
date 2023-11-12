<p float="left">
  <img src="./images/windtunnel-pressure.png" width="32.5%" />
  <img src="./images/windtunnel.png" width="33%" />
  <img src="./images/pressurebox.png" width="33%" />
</p>

# 2D Fluid Simulation on the GPU

## About this project
This project is focused on implementing and optimizing a fluid simulation using Vulkan compute shaders. A grid-based approach has been taken as this maps to GPU memory efficiently and allows for optimizations later on.
The simulation domain is limited to 2D as this simplifies visualization. In this project, compute and present queues are created, and the simulation steps are synchronized to be executed in the correct order. A simulation step is split into advection, pressure projection and visualization. These steps are implemented in different compute shaders which require synchronization as well. A naive implementation of these shaders was used for reference when optimizing performance. For this, a detailed performance analysis was performed using Nvidia Nsight Graphics. Besides optimizing shader dispatches in general by supporting multiple frames in flight, the pressure projection shader was optimized specifically with memory throughput in mind. The performance analysis showed that this shader dominated the duration of a simulation step. The shader itself is memory-bound as it propagates pressure values through the grid iteratively. Here we see a lot of potential for optimization and implemented different techniques such as ghost zoning, kernel decomposition and loop unrolling to increase performance. For technical details and the results of our performance analysis, refer to the [technical report](#technical-report).

## Key features and optimizations

Feature | Status
--------|-------
fluid advection | ✔️
resolve fluid incompressibility | ✔️
scenes with different boundary conditions | ✔️
mouse interaction | ✔️
cli | ✔️
pressure visualization | ✔️
windows support | ✔️
linux support | ✔️

The performance of a naive implementation is far from optimal. To compute more simulation steps per second, the performance was analyzed using Nvidia Nsight Graphics, and the most significant bottlenecks were identified. 
Overall the simulation is memory-bound as most shader dispatches require data from neighbouring grid cells as well. Multiple optimization techniques have been used to increase GPU utilization and throughput.

Optimization | Status | Details
-------------|--------|----------
reduce required memory bandwidth | ✔️ | optimized pressure projection step as it dominates the simulation time
support multiple frames in flight | ✔️ | don't wait for simulation step to finish, start next step asap
decouple simulation resolution | ✔️ | support different resolutions for simulation grid and output buffer


## Requirements

- Install [Vulkan SDK](https://vulkan.lunarg.com)
- Install [cmake](https://cmake.org) 3.16 or later
- Install [Visual Studio](https://visualstudio.microsoft.com) 2019 or later (windows) or [GCC](https://gcc.gnu.org) (linux)

## Compiling

Clone this repo from GitLab. Make sure to clone the submodules as well, f.e. by using `git clone --recursive`. Navigate to the fluid-sim-2d folder and run:

```sh
mkdir build
cd build
cmake ..
```
This will create a visual studio .sln inside the build folder you just created. You can build the solution in visual studio or use cmake:

```sh
cmake --build . --config=Release
./Release/FluidSimulation2D.exe
```

## Usage

The simulation can be configured using a command line interface. By default the Windtunnel scene is simulated. If you run the `FluidSimulation2D.exe` without any arguments, the result should look like the left image:

![scenes](./images/scenes.png)
 
You can change the scene and simulation settings by passing additional arguments when starting the application. For example you can run the Pressurebox scene (right) at a resolution of 512 and in a window of 1200x800 pixels use:
```sh
./FluidSimulation2D.exe --scene=Pressurebox --width=1200 --height=800 --res=512
```

Note that you can pause the animation by pressing spacebar and interact with the fluid by moving the mouse while pressing left mouse button.
To see all available arguments and a list of available scenes use the `--help` option:
```
An interactive Vulkan-based fluid simulation in 2D
--------------------------------------------------
Controls:
  SPACEBAR:        pause / continue simulation
  R:               restart simulation
  O:               toggle output
  LEFT MOUSE BTN:  apply force and dye at cursor
--------------------------------------------------
Usage:
  FluidSimulation2D [OPTION...]

  -s, --scene arg   Scene to simulate: [Paint, Pressurebox, Windtunnel] (default: Windtunnel)
  -i, --iter arg    Number of iterations for pressure projection (default: 20)
  -w, --width arg   Width of output window (default: 1000)
  -h, --height arg  Height of output window (default: 600)
  -r, --res arg     Resolution of the simulation (vertical) (default: 512)
  -t, --dt arg      Time per simulation step [s] (default: 0.001)
  -o, --output arg  Output to visualize: [Dye, Pressure] (default: Dye)
      --help        Print usage
```

By default the colored dye is displayed (left), but the underlying pressure can be visualized as well (right). Toggle the output by pressing `O` on your keyboard:

![output dye or pressure](./images/output-dye-pressure.png)

## Technical report

An important goal of this project is to learn the skills necessary to write high-performance GPU code. To achieve this the simulation is implemented using the Vulkan. The naive implementation was analyzed thoroughly using [Nvidia Nsigh Graphics](https://developer.nvidia.com/nsight-graphics). Slow shaders and bottlenecks within the shaders were identified and the main performance limiters were optimized to improve overall performance:

![Analysis using Nsigh Graphics](./images/nsight-optimized.png)

For details about the implementation and optimization of the simulation take a look at the [technical report](./docs/TechnicalReport.pdf) or the [slides](./docs/PresentationSlides.pdf). These contain information about the underlying physics of fluids and a description of the simulation steps that were implemented. In the technical report, there is a section about performance analysis and optimization on the GPU. For this project dependent texture reads (DTRs) were removed, the workgroup size was optimized and ghost zoning was implemented to make use of shared memory (left). The improvements were analyzed and compared to the naive implementation (right).

![ghost zoning and performance gain](./images/optimization.png)

## Acknowledgments
Special thanks to Mike and Reiner for supervising the project, providing important feedback and motivating the implementation of ghost zoning for a significant performance gain.
