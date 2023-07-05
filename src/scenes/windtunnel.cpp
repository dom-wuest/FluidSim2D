#include "windtunnel.h"

bool WindTunnelSceneBuilder::fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
	unsigned int obstacleX = sim_height / 4;
	unsigned int obstacleY = sim_height / 2;
	unsigned int obstacleR = sim_height / 7;

	for (unsigned int i = 0; i < sim_width; i++) {
		for (unsigned int j = 0; j < sim_height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == sim_height - 1) {
				s = 0; // solid
			}

			int dx = i - obstacleX;
			int dy = j - obstacleY;

			if (dx * dx + dy * dy < obstacleR * obstacleR) {
				s = 0; // solid
			}

			solids[i + sim_width * j] = s;

		}
	}

	for (int j = 1; j < sim_height - 1; j++) {
		u[0 + (sim_width + 1) * j] = 1.0f; // initial velocity
		u[1 + (sim_width + 1) * j] = 1.0f; // initial velocity
	}

	return true;
}

bool WindTunnelSceneBuilder::fillDisplayBuffers(std::vector<glm::vec4>& dye)
{
	for (int j = 4 * height / 9; j < 5 * height / 9; j++) {
		dye[0 + (width)*j] = glm::vec4(0.994, 0.738, 0.167, 1.0); // initial dye color
	}

	return true;
}

glm::vec4 WindTunnelSceneBuilder::dyeColor(uint32_t frameIdx)
{
	return glm::vec4(0.4,0.0,0.0,1.0);
}
