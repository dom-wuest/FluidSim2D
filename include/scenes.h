#pragma once

#include <string>
#include <functional>
#include <stdexcept>
#include <map>
#include <vector>
#include <memory>

#include <glm/glm.hpp>



namespace Scenes {
	class SceneBuilder {
	protected:
		uint32_t width;
		uint32_t height;
		uint32_t sim_width;
		uint32_t sim_height;

	public:
		void setSimulationSize(uint32_t width, uint32_t height);
		void setDisplaySize(uint32_t width, uint32_t height);
		virtual bool fillSimulationBuffers(std::vector<int>& solids, std::vector<float>& u, std::vector<float>& v) = 0;
		virtual bool fillDisplayBuffers(std::vector<glm::vec4>& dye) = 0;
		virtual glm::vec4 dyeColor(uint32_t frameIdx) = 0;
		virtual unsigned int circels(glm::vec2& pos, float& radius) { return 0; }
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