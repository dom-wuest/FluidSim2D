#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include <array>

#include "utils.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const uint32_t SIM_WIDTH = 80;
const uint32_t SIM_HEIGHT = 60;

const size_t NUM_STORAGE_BUFFERS = 4;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices {
	std::optional<uint32_t> computeFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return computeFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct PushConstants {
	unsigned int width;
	unsigned int height;
	unsigned int sim_width;
	unsigned int sim_height;
	float deltaTime;
};

class FluidSimApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue computeQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	//std::vector<VkImage> renderTargetImages;
	std::vector<VkImageView> renderTargetImageViews;
	VkDeviceMemory renderTargetDeviceMemory;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	VkPipelineLayout displayPipelineLayout;
	VkPipeline displayPipeline;

	VkPipelineLayout extrapolatePipelineLayoutU;
	VkPipeline extrapolatePipelineU;

	VkPipelineLayout extrapolatePipelineLayoutV;
	VkPipeline extrapolatePipelineV;

	VkPipelineLayout advectVelPipelineLayout;
	VkPipeline advectVelPipeline;

	VkPipelineLayout calcDivergencePipelineLayout;
	VkPipeline calcDivergencePipeline;

	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkBuffer> solidBuffers;
	std::vector<VkDeviceMemory> solidBuffersMemory;

	std::vector<VkBuffer> velocityUBuffers;
	std::vector<VkDeviceMemory> velocityUBuffersMemory;

	std::vector<VkBuffer> velocityVBuffers;
	std::vector<VkDeviceMemory> velocityVBuffersMemory;

	std::vector<VkBuffer> divergenceBuffers;
	std::vector<VkDeviceMemory> divergenceBuffersMemory;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;
	size_t currentFrame = 0;
	float lastFrameTime = 0.0f;
	double lastTime = 0.0;

	VkDescriptorSetLayout displayDescriptorSetLayout;
	std::vector<VkDescriptorSet> displayDescriptorSets;

	VkDescriptorSetLayout extrapolateDescriptorSetLayout;
	std::vector<VkDescriptorSet> extrapolateDescriptorSetsU;
	std::vector<VkDescriptorSet> extrapolateDescriptorSetsV;

	VkDescriptorSetLayout advectVelDescriptorSetLayout;
	std::vector<VkDescriptorSet> advectVelDescriptorSets;

	VkDescriptorSetLayout calcDivergenceDescriptorSetLayout;
	std::vector<VkDescriptorSet> calcDivergenceDescriptorSets;

	VkDescriptorPool descriptorPool;

	VkSampler imageSampler;

	bool framebufferResized = false;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Fluid Sim 2D", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		lastTime = glfwGetTime();
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<FluidSimApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();

		createSwapChain();
		createImageViews();

		createDisplayDescriptorSetLayout();
		createDisplayPipeline();

		createExtrapolateDescriptorSetLayout();
		createExtrapolatePipelineU();
		createExtrapolatePipelineV();

		createAdvectVelDescriptorSetLayout();
		createAdvectVelPipeline();

		createCalcDivergenceDescriptorSetLayout();
		createCalcDivergencePipeline();

		createCommandPool();

		createShaderStorageBuffers();
		createDescriptorPool();
		createDisplayDescriptorSets();
		createExtrapolateDescriptorSets();
		createAdvectVelDescriptorSets();
		createCalcDivergenceDescriptorSets();
		createCommandBuffers();

		createSyncObjects();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();

			double currentTime = glfwGetTime();
			lastFrameTime = (currentTime - lastTime) * 1000.0;
			lastTime = currentTime;
		}

		vkDeviceWaitIdle(device);
	}

	void cleanupSwapChain() {

		vkFreeMemory(device, renderTargetDeviceMemory, nullptr);
		for (auto imageView : renderTargetImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		//for (auto image : renderTargetImages) {
		//    vkDestroyImage(device, image, nullptr);
		//}
		vkDestroySampler(device, imageSampler, nullptr);

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

		vkDestroyPipeline(device, displayPipeline, nullptr);
		vkDestroyPipelineLayout(device, displayPipelineLayout, nullptr);

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void cleanup() {

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyBuffer(device, solidBuffers[i], nullptr);
			vkFreeMemory(device, solidBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, velocityUBuffers[i], nullptr);
			vkFreeMemory(device, velocityUBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, velocityVBuffers[i], nullptr);
			vkFreeMemory(device, velocityVBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, divergenceBuffers[i], nullptr);
			vkFreeMemory(device, divergenceBuffersMemory[i], nullptr);
		}

		cleanupSwapChain();

		vkDestroyPipeline(device, extrapolatePipelineU, nullptr);
		vkDestroyPipelineLayout(device, extrapolatePipelineLayoutU, nullptr);

		vkDestroyPipeline(device, extrapolatePipelineV, nullptr);
		vkDestroyPipelineLayout(device, extrapolatePipelineLayoutV, nullptr);

		vkDestroyPipeline(device, advectVelPipeline, nullptr);
		vkDestroyPipelineLayout(device, advectVelPipelineLayout, nullptr);

		vkDestroyDescriptorSetLayout(device, displayDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, extrapolateDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, advectVelDescriptorSetLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createDescriptorPool();
		createDisplayDescriptorSets();
		createDisplayPipeline();
		createExtrapolateDescriptorSets();
		createExtrapolatePipelineU();
		createExtrapolatePipelineV();
		createAdvectVelPipeline();
		createCommandBuffers();

	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Fluid Sim 2D";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.computeFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}


	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		for (auto& format : swapChainSupport.formats) {

			if (CheckFormatSupport(physicalDevice, format.format,
				VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
				VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
				surfaceFormat.format = format.format;
				break;
			}
		}


		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.computeFamily.value(), indices.presentFamily.value() };

		if (indices.computeFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());


		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;


			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createDisplayPipeline() {
		auto shader = readFile("../shaders/displayVelocity.comp.spv");
		VkShaderModule shaderModule = createShaderModule(shader);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";

		VkPushConstantRange pcr{};
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo displayPipelineLayoutInfo{};
		displayPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		displayPipelineLayoutInfo.setLayoutCount = 1; // Optional
		displayPipelineLayoutInfo.pSetLayouts = &displayDescriptorSetLayout; // Optional
		displayPipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		displayPipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional

		if (vkCreatePipelineLayout(device, &displayPipelineLayoutInfo, nullptr, &displayPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create displayPipeline layout!");
		}

		VkComputePipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.layout = displayPipelineLayout;
		info.basePipelineIndex = -1;
		info.basePipelineHandle = VK_NULL_HANDLE;
		info.stage = shaderStageInfo;


		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &displayPipeline) != VK_SUCCESS) {
			throw std::runtime_error("compute shader");
		}

		vkDestroyShaderModule(device, shaderModule, nullptr);
	}

	void createExtrapolatePipelineU() {
		auto shader = readFile("../shaders/extrapolateU.comp.spv");
		VkShaderModule shaderModule = createShaderModule(shader);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";

		VkPushConstantRange pcr{};
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo extrapolatePipelineLayoutInfo{};
		extrapolatePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		extrapolatePipelineLayoutInfo.setLayoutCount = 1; // Optional
		extrapolatePipelineLayoutInfo.pSetLayouts = &extrapolateDescriptorSetLayout; // Optional
		extrapolatePipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		extrapolatePipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional

		if (vkCreatePipelineLayout(device, &extrapolatePipelineLayoutInfo, nullptr, &extrapolatePipelineLayoutU) != VK_SUCCESS) {
			throw std::runtime_error("failed to create extrapolatePipeline layout!");
		}

		VkComputePipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.layout = extrapolatePipelineLayoutU;
		info.basePipelineIndex = -1;
		info.basePipelineHandle = VK_NULL_HANDLE;
		info.stage = shaderStageInfo;


		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &extrapolatePipelineU) != VK_SUCCESS) {
			throw std::runtime_error("compute shader");
		}

		vkDestroyShaderModule(device, shaderModule, nullptr);
	}

	void createExtrapolatePipelineV() {
		auto shader = readFile("../shaders/extrapolateV.comp.spv");
		VkShaderModule shaderModule = createShaderModule(shader);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";

		VkPushConstantRange pcr{};
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo extrapolatePipelineLayoutInfo{};
		extrapolatePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		extrapolatePipelineLayoutInfo.setLayoutCount = 1; // Optional
		extrapolatePipelineLayoutInfo.pSetLayouts = &extrapolateDescriptorSetLayout; // Optional
		extrapolatePipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		extrapolatePipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional

		if (vkCreatePipelineLayout(device, &extrapolatePipelineLayoutInfo, nullptr, &extrapolatePipelineLayoutV) != VK_SUCCESS) {
			throw std::runtime_error("failed to create extrapolatePipeline layout!");
		}

		VkComputePipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.layout = extrapolatePipelineLayoutV;
		info.basePipelineIndex = -1;
		info.basePipelineHandle = VK_NULL_HANDLE;
		info.stage = shaderStageInfo;


		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &extrapolatePipelineV) != VK_SUCCESS) {
			throw std::runtime_error("compute shader");
		}

		vkDestroyShaderModule(device, shaderModule, nullptr);
	}

	void createAdvectVelPipeline() {
		auto shader = readFile("../shaders/advectVelocity.comp.spv");
		VkShaderModule shaderModule = createShaderModule(shader);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";

		VkPushConstantRange pcr{};
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1; // Optional
		pipelineLayoutInfo.pSetLayouts = &advectVelDescriptorSetLayout; // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		pipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &advectVelPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create advectPipeline layout!");
		}

		VkComputePipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.layout = advectVelPipelineLayout;
		info.basePipelineIndex = -1;
		info.basePipelineHandle = VK_NULL_HANDLE;
		info.stage = shaderStageInfo;


		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &advectVelPipeline) != VK_SUCCESS) {
			throw std::runtime_error("compute shader");
		}

		vkDestroyShaderModule(device, shaderModule, nullptr);
	}

	void createCalcDivergencePipeline() {
		auto shader = readFile("../shaders/calcDivergence.comp.spv");
		VkShaderModule shaderModule = createShaderModule(shader);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";

		VkPushConstantRange pcr{};
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1; // Optional
		pipelineLayoutInfo.pSetLayouts = &calcDivergenceDescriptorSetLayout; // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		pipelineLayoutInfo.pPushConstantRanges = &pcr; // Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &calcDivergencePipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create advectPipeline layout!");
		}

		VkComputePipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.layout = calcDivergencePipelineLayout;
		info.basePipelineIndex = -1;
		info.basePipelineHandle = VK_NULL_HANDLE;
		info.stage = shaderStageInfo;


		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &calcDivergencePipeline) != VK_SUCCESS) {
			throw std::runtime_error("compute shader");
		}

		vkDestroyShaderModule(device, shaderModule, nullptr);
	}

	void createShaderStorageBuffers() {
		// create scene
		// solid cells
		std::vector<int> solids(SIM_WIDTH * SIM_HEIGHT);
		// horizontal velocity
		std::vector<float> u(SIM_WIDTH * SIM_HEIGHT, 0.0f);
		// vertical velocity
		std::vector<float> v(SIM_WIDTH * SIM_HEIGHT, 0.0f);

		unsigned int obstacleX = SIM_WIDTH / 4;
		unsigned int obstacleY = SIM_HEIGHT / 2;
		unsigned int obstacleR = SIM_HEIGHT / 6;

		for (unsigned int i = 0; i < SIM_WIDTH; i++) {
			for (unsigned int j = 0; j < SIM_HEIGHT; j++) {
				int s = 1; // fluid
				if (i == 0 || j == 0 || j == SIM_HEIGHT - 1) {
					s = 0; // solid
				}

				if (i == j) s = 1;
				if (i == SIM_WIDTH - 1) s = 1;

				int dx = i - obstacleX;
				int dy = j - obstacleY;

				if (dx * dx + dy * dy < obstacleR * obstacleR) {
					s = 0; // solid
				}

				solids[i + SIM_WIDTH * j] = s;

				if (i == 1) {
					u[i + SIM_WIDTH * j] = 1.0f;
				}
			}
		}

		// copy solids to GPU
		{
			solidBuffers.resize(swapChainImages.size());
			solidBuffersMemory.resize(swapChainImages.size());

			VkDeviceSize bufferSize = sizeof(int) * SIM_WIDTH * SIM_HEIGHT;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, solids.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, solidBuffers[i], solidBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, solidBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// copy velocity u to GPU
		{

			velocityUBuffers.resize(swapChainImages.size());
			velocityUBuffersMemory.resize(swapChainImages.size());

			VkDeviceSize bufferSize = sizeof(float) * SIM_WIDTH * SIM_HEIGHT;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, u.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityUBuffers[i], velocityUBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, velocityUBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// copy velocity v to GPU
		{
			velocityVBuffers.resize(swapChainImages.size());
			velocityVBuffersMemory.resize(swapChainImages.size());

			VkDeviceSize bufferSize = sizeof(float) * SIM_WIDTH * SIM_HEIGHT;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, v.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityVBuffers[i], velocityVBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, velocityVBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// divergence
		{
			divergenceBuffers.resize(swapChainImages.size());
			divergenceBuffersMemory.resize(swapChainImages.size());

			VkDeviceSize bufferSize = sizeof(float) * SIM_WIDTH * SIM_HEIGHT;
			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, divergenceBuffers[i], divergenceBuffersMemory[i]);
			}
		}

	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void recordImageBarrier(VkCommandBuffer buffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
		VkAccessFlags scrAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcBind, VkPipelineStageFlags dstBind) {
		VkImageMemoryBarrier barrier{};
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcAccessMask = scrAccess;
		barrier.dstAccessMask = dstAccess;
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		VkImageSubresourceRange sub{};
		sub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		sub.baseArrayLayer = 0;
		sub.baseMipLevel = 0;
		sub.layerCount = VK_REMAINING_MIP_LEVELS;
		sub.levelCount = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange = sub;

		vkCmdPipelineBarrier(buffer, srcBind, dstBind,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	void createCommandBuffers() {
		commandBuffers.resize(swapChainImages.size());

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		//for (size_t i = 0; i < commandBuffers.size(); i++) {
		//    recordCommandBuffer(i);
		//}
	}

	void recordCommandBuffer(size_t imageIdx) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffers[imageIdx], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		PushConstants pc;
		pc.width = WIDTH;
		pc.height = HEIGHT;
		pc.sim_width = SIM_WIDTH;
		pc.sim_height = SIM_HEIGHT;
		pc.deltaTime = (float)lastFrameTime;

		vkCmdPushConstants(commandBuffers[imageIdx], advectVelPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, calcDivergencePipeline);
		vkCmdBindDescriptorSets(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, calcDivergencePipelineLayout, 0, 1, &calcDivergenceDescriptorSets[imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[imageIdx], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		VkMemoryBarrier barrier;
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.pNext = NULL;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffers[imageIdx], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VkDependencyFlags(), 1, &barrier, 0, nullptr, 0, nullptr);

		vkCmdPushConstants(commandBuffers[imageIdx], advectVelPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, advectVelPipeline);
		vkCmdBindDescriptorSets(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, advectVelPipelineLayout, 0, 1, &advectVelDescriptorSets[imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[imageIdx], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		VkMemoryBarrier barrier_advect;
		barrier_advect.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier_advect.pNext = NULL;
		barrier_advect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier_advect.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffers[imageIdx], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VkDependencyFlags(), 1, &barrier_advect, 0, nullptr, 0, nullptr);

		vkCmdPushConstants(commandBuffers[imageIdx], extrapolatePipelineLayoutU, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineU);
		vkCmdBindDescriptorSets(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineLayoutU, 0, 1, &extrapolateDescriptorSetsU[imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[imageIdx], (SIM_HEIGHT + 63) / 64, 1, 1);

		vkCmdPushConstants(commandBuffers[imageIdx], extrapolatePipelineLayoutV, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineV);
		vkCmdBindDescriptorSets(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineLayoutV, 0, 1, &extrapolateDescriptorSetsV[imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[imageIdx], (SIM_WIDTH + 63) / 64, 1, 1);

		recordImageBarrier(commandBuffers[imageIdx], swapChainImages[imageIdx],
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(commandBuffers[imageIdx], displayPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, displayPipeline);
		vkCmdBindDescriptorSets(commandBuffers[imageIdx], VK_PIPELINE_BIND_POINT_COMPUTE, displayPipelineLayout, 0, 1, &displayDescriptorSets[imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[imageIdx], (WIDTH + 31) / 32, (HEIGHT + 31) / 32, 1);

		recordImageBarrier(commandBuffers[imageIdx], swapChainImages[imageIdx],
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		if (vkEndCommandBuffer(commandBuffers[imageIdx]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	void createDisplayDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		Utils::addStorageImage(layoutBindings, 0, &imageSampler);
		Utils::addStorageBuffer(layoutBindings, 1);
		Utils::addStorageBuffer(layoutBindings, 2);
		Utils::addStorageBuffer(layoutBindings, 3);

		Utils::createDescriptorSetLayout(device, layoutBindings, displayDescriptorSetLayout);
	}

	void createExtrapolateDescriptorSetLayout() {

		std::array<VkDescriptorSetLayoutBinding, 1> layoutBindings{};

		// velocity u buffer
		layoutBindings[0].binding = 0;
		layoutBindings[0].descriptorCount = 1;
		layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[0].pImmutableSamplers = nullptr;
		layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = layoutBindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &extrapolateDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

	}

	void createAdvectVelDescriptorSetLayout() {

		std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings{};

		for (int i = 0; i < 5; i++) {
			layoutBindings[i].binding = i;
			layoutBindings[i].descriptorCount = 1;
			layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			layoutBindings[i].pImmutableSamplers = nullptr;
			layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 5;
		layoutInfo.pBindings = layoutBindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &advectVelDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

	}

	void createCalcDivergenceDescriptorSetLayout() {

		std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};

		for (int i = 0; i < 4; i++) {
			layoutBindings[i].binding = i;
			layoutBindings[i].descriptorCount = 1;
			layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			layoutBindings[i].pImmutableSamplers = nullptr;
			layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 4;
		layoutInfo.pBindings = layoutBindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &calcDivergenceDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

	}

	void createDescriptorPool() {

		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size() * 13);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size() * 5);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}

	}

	void createDisplayDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), displayDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		displayDescriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, displayDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		Utils::createImageSampler(device, imageSampler);

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			Utils::bindImage(device, swapChainImageViews[i], imageSampler, displayDescriptorSets[i], 0);
			Utils::bindBuffer(device, solidBuffers[i], displayDescriptorSets[i], 1);
			Utils::bindBuffer(device, velocityUBuffers[i], displayDescriptorSets[i], 2);
			Utils::bindBuffer(device, velocityVBuffers[i], displayDescriptorSets[i], 3);
		}
	}

	void createExtrapolateDescriptorSets() {
		{
			std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), extrapolateDescriptorSetLayout);
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
			allocInfo.pSetLayouts = layouts.data();

			extrapolateDescriptorSetsU.resize(swapChainImages.size());
			if (vkAllocateDescriptorSets(device, &allocInfo, extrapolateDescriptorSetsU.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::bindBuffer(device, velocityUBuffers[i], extrapolateDescriptorSetsU[i], 0);
			}
		}
		{
			std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), extrapolateDescriptorSetLayout);
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
			allocInfo.pSetLayouts = layouts.data();

			extrapolateDescriptorSetsV.resize(swapChainImages.size());
			if (vkAllocateDescriptorSets(device, &allocInfo, extrapolateDescriptorSetsV.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				Utils::bindBuffer(device, velocityVBuffers[i], extrapolateDescriptorSetsV[i], 0);
			}
		}

	}

	void createAdvectVelDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), advectVelDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		advectVelDescriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, advectVelDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {

			Utils::bindBuffer(device, solidBuffers[i], advectVelDescriptorSets[i], 0);

			// i_old is index of previous frame
			size_t i_old = (i - 1) % swapChainImages.size();

			Utils::bindBuffer(device, velocityUBuffers[i_old], advectVelDescriptorSets[i], 1);
			Utils::bindBuffer(device, velocityVBuffers[i_old], advectVelDescriptorSets[i], 2);

			Utils::bindBuffer(device, velocityUBuffers[i], advectVelDescriptorSets[i], 3);
			Utils::bindBuffer(device, velocityVBuffers[i], advectVelDescriptorSets[i], 4);
		}
	}

	void createCalcDivergenceDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), calcDivergenceDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		calcDivergenceDescriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, calcDivergenceDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {

			Utils::bindBuffer(device, solidBuffers[i], calcDivergenceDescriptorSets[i], 0);

			// i_old is index of previous frame
			size_t i_old = (i - 1) % swapChainImages.size();
			Utils::bindBuffer(device, velocityUBuffers[i_old], calcDivergenceDescriptorSets[i], 1);
			Utils::bindBuffer(device, velocityVBuffers[i_old], calcDivergenceDescriptorSets[i], 2);

			Utils::bindBuffer(device, divergenceBuffers[i], calcDivergenceDescriptorSets[i], 3);
		}
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
			vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
		}
		imagesInFlight[imageIndex] = inFlightFences[currentFrame];

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[imageIndex], /*VkCommandBufferResetFlagBits*/ 0);
		recordCommandBuffer(imageIndex);


		if (vkQueueSubmit(computeQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
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

	bool CheckFormatSupport(VkPhysicalDevice gpu, VkFormat format, VkFormatFeatureFlags requestedSupport) {
		VkFormatProperties vkFormatProperties;
		vkGetPhysicalDeviceFormatProperties(gpu, format, &vkFormatProperties);
		return (vkFormatProperties.optimalTilingFeatures & requestedSupport) == requestedSupport;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
				indices.computeFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};

int main() {
	FluidSimApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}