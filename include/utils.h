#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>

#define CAST(a) static_cast<uint32_t>(a.size())

namespace Utils {
	void createBuffer(VkPhysicalDevice& pDevice, VkDevice& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	void copyBuffer(VkDevice& device, VkBuffer& srcBuffer, VkBuffer& dstBuffer, VkDeviceSize& size, VkCommandPool& commandPool, VkQueue& commandQueue);
	uint32_t findMemoryType(VkPhysicalDevice& pDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void addStorageBuffer(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding);
	void addStorageImage(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding, VkSampler *sampler);
	void createDescriptorSetLayout(VkDevice& device, std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout& layout);
	void bindBuffer(VkDevice& device, VkBuffer& buffer, VkDescriptorSet& set, uint32_t binding);
	void bindImage(VkDevice& device, VkImageView& imageView, VkSampler& imageSampler, VkDescriptorSet& set, uint32_t binding);
	void createImageSampler(VkDevice& device, VkSampler& imageSampler);
}