#pragma once

#include "scenes.h"

class WindTunnelSceneBuilder : public Scenes::SceneBuilder {
public:
	bool fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v);
	bool fillDisplayBuffers(std::vector<glm::vec4>& dye);
	virtual glm::vec4 dyeColor(uint32_t frameIdx);
};

REGISTER_SCENE("Windtunnel", WindTunnelSceneBuilder)