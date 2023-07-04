#include "paint.h"

bool PaintSceneBuilder::fillDisplayBuffers(std::vector<glm::vec4>& dye)
{
	return true;
}

bool PaintSceneBuilder::fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
	for (unsigned int i = 0; i < sim_width; i++) {
		for (unsigned int j = 0; j < sim_height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == sim_height - 1 || i == sim_width - 1) {
				s = 0; // solid
			}

			solids[i + sim_width * j] = s;

		}
	}

	return true;
}

glm::vec4 PaintSceneBuilder::dyeColor(uint32_t frameIdx)
{
	float hue = float((32 + 117 * frameIdx) % 360);
	glm::vec4 hsv(hue, 1.0, 1.0, 1.0);
	return Utils::hsv2rgb(hsv);
}
