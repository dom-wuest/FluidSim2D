#include "scenes.h"

bool Scenes::SceneManager::reg(std::string name, Scenes::AbstractBuilder builder)
{
    scenes[name] = builder;
	return true;
}

std::unique_ptr<Scenes::SceneBuilder> Scenes::SceneManager::createScene(std::string name)
{
    if (scenes.count(name) <= 0) {
        std::string msg = "Scene '" + name + "' does not exist";
        throw new std::runtime_error(msg.c_str());
    }
    auto scene = scenes.at(name)();
    std::unique_ptr<SceneBuilder> ptr;
    ptr.reset(scene);
    return ptr;
}

std::vector<std::string> Scenes::SceneManager::availableScenes()
{
    std::vector<std::string> names;
    for (auto entry : scenes)
    {
        names.push_back(entry.first);
    }
    return names;
}

void Scenes::SceneBuilder::setSimulationSize(uint32_t width, uint32_t height)
{
    sim_width = width;
    sim_height = height;
}

void Scenes::SceneBuilder::setDisplaySize(uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
}
