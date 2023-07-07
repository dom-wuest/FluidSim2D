#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include <glm/glm.hpp>

#define CAST(a) static_cast<uint32_t>(a.size())

namespace Utils {
	struct ComputeShader {
		VkDescriptorSetLayout descLayout;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
	};

	void createBuffer(VkPhysicalDevice& pDevice, VkDevice& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	void copyBuffer(VkDevice& device, VkBuffer& srcBuffer, VkBuffer& dstBuffer, VkDeviceSize& size, VkCommandPool& commandPool, VkQueue& commandQueue);
	uint32_t findMemoryType(VkPhysicalDevice& pDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void addStorageBuffer(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding);
	VkShaderModule createShaderModule(VkDevice& device, const std::vector<char>& code);
	static std::vector<char> readFile(const std::string& filename);
	void createPipeline(VkDevice& device, const std::string& shaderFile, ComputeShader& computeShader, uint32_t pushConstantSize = 0, glm::ivec2 workgroupsize=glm::ivec2(32));
	void addStorageImage(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding, VkSampler *sampler);
	void createDescriptorSetLayout(VkDevice& device, std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout& layout);
	void bindBuffer(VkDevice& device, VkBuffer& buffer, VkDescriptorSet& set, uint32_t binding);
	void bindImage(VkDevice& device, VkImageView& imageView, VkSampler& imageSampler, VkDescriptorSet& set, uint32_t binding);
	void createImageSampler(VkDevice& device, VkSampler& imageSampler);
	glm::vec4 hsv2rgb(glm::vec4 hsv);
}