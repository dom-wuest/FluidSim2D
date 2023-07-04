#include "paint.h"

bool PaintSceneBuilder::fillBuffers(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v, std::vector<glm::vec4>& dye)
{
	for (unsigned int i = 0; i < width; i++) {
		for (unsigned int j = 0; j < height; j++) {
			int s = 1; // fluid
			if (i == 0 || j == 0 || j == height - 1 || i == width - 1) {
				s = 0; // solid
			}

			solids[i + width * j] = s;

		}
	}

	return true;
}