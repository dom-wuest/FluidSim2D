#include "windtunnel.h"

bool WindTunnelSceneBuilder::fillBuffers(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v, std::vector<glm::vec4>& dye)
{
	unsigned int obstacleX = width / 4;
	unsigned int obstacleY = height / 2 + 2;
	unsigned int obstacleR = height / 6;

	for (unsigned int i = 0; i < width; i++) {
		for (unsigned int j = 0; j < height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == height - 1) {
				s = 0; // solid
			}

			int dx = i - obstacleX;
			int dy = j - obstacleY;

			if (dx * dx + dy * dy < obstacleR * obstacleR) {
				s = 0; // solid
			}

			solids[i + width * j] = s;

		}
	}

	
	for (int j = 1; j < height / 2; j++) {
		u[0 + (width + 1) * j] = 1.0f; // initial velocity
		u[1 + (width + 1) * j] = 1.0f; // initial velocity
	}

	for (int j = 2 * height / 5; j < 3 * height / 5; j++) {
		dye[0 + (width)*j] = glm::vec4(0.994, 0.738, 0.167, 1.0); // initial dye color
	}

	return true;
}