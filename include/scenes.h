#pragma once

#include <string>
#include <functional>
#include <map>
#include <vector>
#include <memory>

#include <glm/glm.hpp>



namespace Scenes {
	class SceneBuilder {
	public:
		virtual bool fillBuffers(uint32_t width, uint32_t height, std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v, std::vector<glm::vec4>& dye) = 0;
	};

	typedef std::function<SceneBuilder* ()> AbstractBuilder;

	class SceneManager {
	public:
		static SceneManager& instance() {
			static SceneManager INSTANCE;
			return INSTANCE;
		}
		bool reg(std::string name, AbstractBuilder builder);
		std::unique_ptr<SceneBuilder> createScene(std::string name);
		std::vector<std::string> availableScenes();

	private:
		std::map<std::string, AbstractBuilder> scenes;
	};

	template<typename Implementation>
	SceneBuilder* builder() {
		return new Implementation();
	}

	
}

#define REGISTER_SCENE(name, impl) const bool res = Scenes::SceneManager::instance().reg(name, Scenes::builder<impl>);