#include "scenes.h"

bool Scenes::SceneManager::reg(std::string name, Scenes::AbstractBuilder builder)
{
    scenes[name] = builder;
	return true;
}

bool Scenes::SceneManager::createScene(std::string name, uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v)
{
    if (scenes.count(name) <= 0) {
        return false; // scene does not exist
    }
    auto scene = scenes.at(name)();
    bool result = scene->build(width,height,solids,u,v);
    delete scene;
    return result;
}