# 2D Fluid Simulation on the GPU



## Requirements

- Install [Vulkan SDK](https://vulkan.lunarg.com)
- Install [cmake](https://cmake.org) 3.16 or later

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

The simulation can be configured using a command line interface. By default the Windtunnel scene is simulated. If you run the `FluidSimulation2D.exe` without any arguments, the result should look like this:

![windtunnel scene](./images/windtunnel.png)
 
You can change the scene and simulation settings by passing additional arguments when starting the application. For example you can run the Pressurebox scene at a resolution of 512 and in a window of 1200x800 pixels use:
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
  LEFT MOUSE BTN:  apply force and dye at cursor
--------------------------------------------------
Usage:
  FluidSimulation2D [OPTION...]

  -s, --scene arg   Scene to simulate: [Paint, Pressurebox, Windtunnel] (default: Windtunnel)
  -i, --iter arg    Number of iterations for pressure projection (default: 8)
  -w, --width arg   Width of output window (default: 1000)
  -h, --height arg  Height of output window (default: 600)
  -r, --res arg     Resolution of the simulation (vertical) (default: 512)
  -t, --dt arg      Time per simulation step [s] (default: 0.0003)
      --help        Print usage
```