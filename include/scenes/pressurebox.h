#pragma once

#include "scenes.h"

class PressureBoxSceneBuilder : public Scenes::SceneBuilder {
public:
	bool fillDisplayBuffers(std::vector<glm::vec4>& dye);
	bool fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v);
	virtual glm::vec4 dyeColor(uint32_t frameIdx);
};

REGISTER_SCENE("Pressurebox", PressureBoxSceneBuilder)