# 2D Fluid Simulation on the GPU



## Requirements

- Vulkan SDK
- cmake

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

The simulation can be configured using a command line interface. By default the Windtunnel scene is simulated. To run the Pressurebox scene at a resolution of 512 and in a window of 1200x800 pixels use:

```sh
./FluidSimulation2D.exe --scene=Pressurebox --width=1200 --height=800 --res=512
```

To see all available arguments and a list of available scenes use the `--help` option.