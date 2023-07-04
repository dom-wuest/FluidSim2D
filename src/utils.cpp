#include "utils.h"

namespace Utils {

    void createBuffer(VkPhysicalDevice& pDevice, VkDevice& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(pDevice, memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkDevice &device, VkBuffer &srcBuffer, VkBuffer &dstBuffer, VkDeviceSize &size, VkCommandPool &commandPool, VkQueue &commandQueue) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(commandQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(commandQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    uint32_t findMemoryType(VkPhysicalDevice &pDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(pDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    void addStorageImage(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding, VkSampler *sampler)
    {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorCount = 1;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBinding.pImmutableSamplers = sampler;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(layoutBinding);
    }
    void createDescriptorSetLayout(VkDevice& device, std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout& layout)
    {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = CAST(bindings);
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void bindBuffer(VkDevice& device, VkBuffer& buffer, VkDescriptorSet& set, uint32_t binding)
    {
        VkDescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = buffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &storageBufferInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void bindImage(VkDevice& device, VkImageView& imageView, VkSampler& imageSampler, VkDescriptorSet& set, uint32_t binding)
    {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = imageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.sampler = imageSampler;

        VkWriteDescriptorSet descriptorWrites{};

        // swapchain image
        descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.dstSet = set;
        descriptorWrites.dstBinding = binding;
        descriptorWrites.dstArrayElement = 0;
        descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites.descriptorCount = 1;
        descriptorWrites.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrites, 0, nullptr);
    }

    void createImageSampler(VkDevice& device, VkSampler& imageSampler)
    {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        vkCreateSampler(device, &samplerInfo, nullptr, &imageSampler);
    }

    glm::vec4 hsv2rgb(glm::vec4 hsv)
    {
        // CODE FOR THIS FUNCTION PROVIDED BY David H. 
        // at https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
        double      hh, p, q, t, ff;
        long        i;
        glm::vec4   rgb(hsv.a);

        hh = hsv.r;
        if (hh >= 360.0) hh = 0.0;
        hh /= 60.0;
        i = (long)hh;
        ff = hh - i;
        p = hsv.b * (1.0 - hsv.g);
        q = hsv.b * (1.0 - (hsv.g * ff));
        t = hsv.b * (1.0 - (hsv.g * (1.0 - ff)));

        switch (i) {
        case 0:
            rgb.r = hsv.b;
            rgb.g = t;
            rgb.b = p;
            break;
        case 1:
            rgb.r = q;
            rgb.g = hsv.b;
            rgb.b = p;
            break;
        case 2:
            rgb.r = p;
            rgb.g = hsv.b;
            rgb.b = t;
            break;

        case 3:
            rgb.r = p;
            rgb.g = q;
            rgb.b = hsv.b;
            break;
        case 4:
            rgb.r = t;
            rgb.g = p;
            rgb.b = hsv.b;
            break;
        case 5:
        default:
            rgb.r = hsv.b;
            rgb.g = p;
            rgb.b = q;
            break;
        }
        return rgb;
    }

    void addStorageBuffer(std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t binding)
    {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorCount = 1;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.pImmutableSamplers = nullptr;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(layoutBinding);
    }

    VkShaderModule createShaderModule(VkDevice& device, const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            std::string msg = "failed to open file: " + filename;
            throw std::runtime_error(msg);
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    void createPipeline(VkDevice& device, const std::string& shaderFile, ComputeShader& computeShader, uint32_t pushConstantSize) {
        
        auto shader = readFile(shaderFile + ".spv");
        VkShaderModule shaderModule = createShaderModule(device, shader);

        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = pushConstantSize;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1; // Optional
        pipelineLayoutInfo.pSetLayouts = &computeShader.descLayout; // Optional
        if (pushConstantSize > 0) {
            pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
            pipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional
        }
        else {
            pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        }

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeShader.pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.layout = computeShader.pipelineLayout;
        info.basePipelineIndex = -1;
        info.basePipelineHandle = VK_NULL_HANDLE;
        info.stage = shaderStageInfo;


        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &computeShader.pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline!");
        }

        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
}