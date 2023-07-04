#include "pressurebox.h"

bool PressureBoxSceneBuilder::fillDisplayBuffers(std::vector<glm::vec4>& dye)
{
	for (int j = height / 6; j < 3 * height / 10; j++) {
		dye[0 + (width)*j] = glm::vec4(0.724, 0.197, 0.537, 1.0); // initial dye color
	}

	for (int j = 3 * height / 10; j < 4 * height / 10; j++) {
		dye[0 + (width)*j] = glm::vec4(0.327, 0.006, 0.646, 1.0); // initial dye color
	}
	return true;
}

bool PressureBoxSceneBuilder::fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
	for (unsigned int i = 0; i < sim_width; i++) {
		for (unsigned int j = 0; j < sim_height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == sim_height - 1 || i == sim_width - 1) {
				s = 0; // solid
			}

			if (i == sim_width / 4 && j >= sim_height / 4 && j < 3 * sim_height / 4) {
				s = 0; // solid
			}

			if (i >= sim_width / 4 && j == sim_height / 4 && i < 3 * sim_width / 4) {
				s = 0; // solid
			}

			if (i >= sim_width / 4 && j == 3 * sim_height / 4 && i < 3 * sim_width / 4) {
				s = 0; // solid
			}
			solids[i + sim_width * j] = s;
		}
	}

	for (int j = 1; j < sim_height / 2; j++) {
		u[1 + (sim_width + 1) * j] = 0.4f; // initial velocity
		v[0 + (sim_width)*j] = 0.2f; // initial velocity
	}

	return true;
}

glm::vec4 PressureBoxSceneBuilder::dyeColor(uint32_t frameIdx)
{
	return glm::vec4(0.3,0.3,0.3,1.0);
}
