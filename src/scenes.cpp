#include "scenes.h"

bool Scenes::SceneManager::reg(std::string name, Scenes::AbstractBuilder builder)
{
    scenes[name] = builder;
	return true;
}

std::unique_ptr<Scenes::SceneBuilder> Scenes::SceneManager::createScene(std::string name)
{
    if (scenes.count(name) <= 0) {
         // scene does not exist
        return std::unique_ptr<SceneBuilder>(nullptr);
    }
    auto scene = scenes.at(name)();
    std::unique_ptr<SceneBuilder> ptr;
    ptr.reset(scene);
    return ptr;
}