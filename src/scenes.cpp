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
        throw std::exception(msg.c_str());
    }
    auto scene = scenes.at(name)();
    std::unique_ptr<SceneBuilder> ptr;
    ptr.reset(scene);
    return ptr;
}

std::vector<std::string> Scenes::SceneManager::availableScenes()
{
    std::vector<std::string> names;
    for each (auto entry in scenes)
    {
        names.push_back(entry.first);
    }
    return names;
}