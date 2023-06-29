#include "pressurebox.h"

bool PressureBoxSceneBuilder::build(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
	for (unsigned int i = 0; i < width; i++) {
		for (unsigned int j = 0; j < height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == height - 1 || i == width-1) {
				s = 0; // solid
			}

			if (i == width / 4 && j >= height / 4 && j < 3 * height / 4) {
				s = 0; // solid
			}

			if (i >= width / 4 && j == height / 4 && i < 3 * width / 4) {
				s = 0; // solid
			}

			if (i >= width / 4 && j == 3 * height / 4 && i < 3 * width / 4) {
				s = 0; // solid
			}
			solids[i + width * j] = s;
		}
	}

	for (int j = 1; j < height/2; j++) {
		u[1 + (width + 1) * j] = 1.0f; // initial velocity
	}
	return true;
}
