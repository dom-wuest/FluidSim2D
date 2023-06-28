#include "windtunnel.h"

bool WindTunnelSceneBuilder::build(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
	unsigned int obstacleX = width / 4;
	unsigned int obstacleY = height / 2 + 1;
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

	
	for (int j = 1; j < height-1; j++) {
		u[0 + (width + 1) * j] = .99f; // initial velocity
		u[1 + (width + 1) * j] = .99f; // initial velocity
	}

	return true;
}