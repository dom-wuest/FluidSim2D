#pragma once

#include "scenes.h"

class PressureBoxSceneBuilder : public Scenes::SceneBuilder {
public:
	bool fillBuffers(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v, std::vector<glm::vec4>& dye);
	virtual glm::vec4 dyeColor(uint32_t frameIdx);
};

REGISTER_SCENE("Pressurebox", PressureBoxSceneBuilder)