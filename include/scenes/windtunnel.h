#pragma once

#include "scenes.h"

class WindTunnelSceneBuilder : public Scenes::SceneBuilder {
public:
	bool build(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v);
};

REGISTER_SCENE("Windtunnel", WindTunnelSceneBuilder)