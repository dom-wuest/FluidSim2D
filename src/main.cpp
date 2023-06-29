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
#include "scenes.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const uint32_t SIM_WIDTH = 160;
const uint32_t SIM_HEIGHT = 120;

const int MAX_FRAMES_IN_FLIGHT = 2;

const int PRESSURE_ITERATIONS = 41;

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

	Utils::ComputeShader displayShader;
	Utils::ComputeShader advectionShader;
	Utils::ComputeShader divergenceShader;
	Utils::ComputeShader pressureShader;
	Utils::ComputeShader applyPressureShader;
	Utils::ComputeShader clearPressureShader;

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

	std::vector<VkBuffer> pressureBuffers;
	std::vector<VkDeviceMemory> pressureBuffersMemory;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;
	size_t currentFrame = 0;
	float lastFrameTime = 0.0f;
	double lastTime = 0.0;

	std::vector<VkDescriptorSet> displayDescriptorSets;
	std::vector<VkDescriptorSet> advectionDescriptorSets;
	std::vector<VkDescriptorSet> divergenceDescriptorSets;
	std::vector<VkDescriptorSet> pressureDescriptorSets;
	std::vector<VkDescriptorSet> applyPressureDescriptorSets;
	std::vector<VkDescriptorSet> clearPressureDescriptorSets;

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

		createAdvectionDescriptorSetLayout();
		createAdvectionPipeline();

		createDivergenceDescriptorSetLayout();
		createDivergencePipeline();

		createPressureDescriptorSetLayout();
		createPressurePipeline();

		createClearPressureDescriptorSetLayout();
		createClearPressurePipeline();

		createApplyPressureDescriptorSetLayout();
		createApplyPressurePipeline();

		createCommandPool();

		createShaderStorageBuffers();
		createDescriptorPool();
		createDisplayDescriptorSets();
		createAdvectionDescriptorSets();
		createDivergenceDescriptorSets();
		createPressureDescriptorSets();
		createClearPressureDescriptorSets();
		createApplyPressureDescriptorSets();

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

		vkDestroyPipeline(device, displayShader.pipeline, nullptr);
		vkDestroyPipelineLayout(device, displayShader.pipelineLayout, nullptr);

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void cleanup() {

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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

		vkDestroyPipeline(device, advectionShader.pipeline, nullptr);
		vkDestroyPipelineLayout(device, advectionShader.pipelineLayout, nullptr);

		vkDestroyDescriptorSetLayout(device, displayShader.descLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, advectionShader.descLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, divergenceShader.descLayout, nullptr);

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
		createAdvectionPipeline();
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
		Utils::createPipeline(device, "displayVelocity.comp", displayShader, sizeof(PushConstants));
	}

	void createAdvectionPipeline() {
		Utils::createPipeline(device, "advectVelocity.comp", advectionShader, sizeof(PushConstants));
	}

	void createDivergencePipeline() {
		Utils::createPipeline(device, "calcDivergence.comp", divergenceShader, sizeof(PushConstants));
	}

	void createPressurePipeline() {
		Utils::createPipeline(device, "projectPressure.comp", pressureShader, sizeof(PushConstants));
	}

	void createClearPressurePipeline() {
		Utils::createPipeline(device, "clear.comp", clearPressureShader, sizeof(PushConstants));
	}

	void createApplyPressurePipeline() {
		Utils::createPipeline(device, "applyPressure.comp", applyPressureShader, sizeof(PushConstants));
	}

	void createShaderStorageBuffers() {
		// create scene
		// solid cells
		std::vector<int> solids(SIM_WIDTH * SIM_HEIGHT);
		// horizontal velocity
		std::vector<float> u((SIM_WIDTH + 1) * (SIM_HEIGHT), 0.0f);
		// vertical velocity
		std::vector<float> v((SIM_WIDTH) * (SIM_HEIGHT + 1), 0.0f);
		// pressure
		std::vector<float> p(SIM_WIDTH * SIM_HEIGHT, 0.0f);

		Scenes::SceneManager::instance().createScene("Pressurebox" /*"Windtunnel"*/, SIM_WIDTH, SIM_HEIGHT, solids, u, v);
		
		// copy solids to GPU
		{
			solidBuffers.resize(MAX_FRAMES_IN_FLIGHT);
			solidBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

			VkDeviceSize bufferSize = sizeof(int) * SIM_WIDTH * SIM_HEIGHT;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, solids.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, solidBuffers[i], solidBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, solidBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// copy velocity u to GPU
		{

			velocityUBuffers.resize(MAX_FRAMES_IN_FLIGHT * 2);
			velocityUBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT * 2);

			VkDeviceSize bufferSize = sizeof(float) * (SIM_WIDTH + 1) * (SIM_HEIGHT);

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, u.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT * 2; i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityUBuffers[i], velocityUBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, velocityUBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// copy velocity v to GPU
		{
			velocityVBuffers.resize(MAX_FRAMES_IN_FLIGHT * 2);
			velocityVBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT * 2);

			VkDeviceSize bufferSize = sizeof(float) * (SIM_WIDTH) * (SIM_HEIGHT + 1);

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, v.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT * 2; i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityVBuffers[i], velocityVBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, velocityVBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		// divergence
		{
			divergenceBuffers.resize(MAX_FRAMES_IN_FLIGHT);
			divergenceBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

			VkDeviceSize bufferSize = sizeof(float) * SIM_WIDTH * SIM_HEIGHT;
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, divergenceBuffers[i], divergenceBuffersMemory[i]);
			}
		}

		// pressure
		{
			pressureBuffers.resize(MAX_FRAMES_IN_FLIGHT * 2);
			pressureBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT * 2);

			VkDeviceSize bufferSize = sizeof(float) * SIM_WIDTH * SIM_HEIGHT;
			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, p.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			// Copy initial data to all storage buffers
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT * 2; i++) {
				Utils::createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pressureBuffers[i], pressureBuffersMemory[i]);
				Utils::copyBuffer(device, stagingBuffer, pressureBuffers[i], bufferSize, commandPool, computeQueue);
			}

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
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
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

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

	void recordCommandBuffer(size_t currentFrame, size_t imageIdx) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		PushConstants pc;
		pc.width = WIDTH;
		pc.height = HEIGHT;
		pc.sim_width = SIM_WIDTH;
		pc.sim_height = SIM_HEIGHT;
		pc.deltaTime = (float)lastFrameTime / 1000.0;
		//pc.deltaTime = 0.003;

		vkCmdPushConstants(commandBuffers[currentFrame], divergenceShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, divergenceShader.pipeline);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, divergenceShader.pipelineLayout, 0, 1, &divergenceDescriptorSets[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		vkCmdPushConstants(commandBuffers[currentFrame], clearPressureShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, clearPressureShader.pipeline);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, clearPressureShader.pipelineLayout, 0, 1, &clearPressureDescriptorSets[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		for (int i = 0; i < PRESSURE_ITERATIONS; i++) {
			int pingpong = i % 2;
			VkMemoryBarrier barrier;
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			barrier.pNext = NULL;
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			

			vkCmdPipelineBarrier(commandBuffers[currentFrame], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VkDependencyFlags(), 1, &barrier, 0, nullptr, 0, nullptr);

			vkCmdPushConstants(commandBuffers[currentFrame], pressureShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

			vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pressureShader.pipeline);
			vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pressureShader.pipelineLayout, 0, 1, &pressureDescriptorSets[2 * currentFrame + pingpong], 0, nullptr);
			vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		}
		
		VkMemoryBarrier barrier2;
		barrier2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier2.pNext = NULL;
		barrier2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffers[currentFrame], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VkDependencyFlags(), 1, &barrier2, 0, nullptr, 0, nullptr);

		vkCmdPushConstants(commandBuffers[currentFrame], applyPressureShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, applyPressureShader.pipeline);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, applyPressureShader.pipelineLayout, 0, 1, &applyPressureDescriptorSets[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);

		VkMemoryBarrier barrier_advect;
		barrier_advect.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier_advect.pNext = NULL;
		barrier_advect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier_advect.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffers[currentFrame], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VkDependencyFlags(), 1, &barrier_advect, 0, nullptr, 0, nullptr);

		
		vkCmdPushConstants(commandBuffers[currentFrame], advectionShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, advectionShader.pipeline);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, advectionShader.pipelineLayout, 0, 1, &advectionDescriptorSets[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 31) / 32, (SIM_HEIGHT + 31) / 32, 1);
		
				/*
		vkCmdPushConstants(commandBuffers[currentFrame], extrapolatePipelineLayoutU, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineU);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineLayoutU, 0, 1, &extrapolateDescriptorSetsU[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_HEIGHT + 63) / 64, 1, 1);

		vkCmdPushConstants(commandBuffers[currentFrame], extrapolatePipelineLayoutV, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineV);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, extrapolatePipelineLayoutV, 0, 1, &extrapolateDescriptorSetsV[currentFrame], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (SIM_WIDTH + 63) / 64, 1, 1);
		*/
		recordImageBarrier(commandBuffers[currentFrame], swapChainImages[imageIdx],
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(commandBuffers[currentFrame], displayShader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, displayShader.pipeline);
		vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, displayShader.pipelineLayout, 0, 1, &displayDescriptorSets[currentFrame * swapChainImages.size() + imageIdx], 0, nullptr);
		vkCmdDispatch(commandBuffers[currentFrame], (WIDTH + 31) / 32, (HEIGHT + 31) / 32, 1);

		recordImageBarrier(commandBuffers[currentFrame], swapChainImages[imageIdx],
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
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
		Utils::addStorageBuffer(layoutBindings, 4);

		Utils::createDescriptorSetLayout(device, layoutBindings, displayShader.descLayout);
	}

	void createAdvectionDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		for (int i = 0; i < 5; i++) {
			Utils::addStorageBuffer(layoutBindings, i);
		}

		Utils::createDescriptorSetLayout(device, layoutBindings, advectionShader.descLayout);
	}

	void createDivergenceDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		for (int i = 0; i < 4; i++) {
			Utils::addStorageBuffer(layoutBindings, i);
		}

		Utils::createDescriptorSetLayout(device, layoutBindings, divergenceShader.descLayout);
	}

	void createPressureDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		for (int i = 0; i < 4; i++) {
			Utils::addStorageBuffer(layoutBindings, i);
		}

		Utils::createDescriptorSetLayout(device, layoutBindings, pressureShader.descLayout);
	}

	void createClearPressureDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		for (int i = 0; i < 1; i++) {
			Utils::addStorageBuffer(layoutBindings, i);
		}

		Utils::createDescriptorSetLayout(device, layoutBindings, clearPressureShader.descLayout);
	}

	void createApplyPressureDescriptorSetLayout() {

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		for (int i = 0; i < 6; i++) {
			Utils::addStorageBuffer(layoutBindings, i);
		}

		Utils::createDescriptorSetLayout(device, layoutBindings, applyPressureShader.descLayout);
	}

	void createDescriptorPool() {

		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size() * MAX_FRAMES_IN_FLIGHT);

		// divergence: 4
		// clear: 1
		// projection: 4*2
		// application: 6
		// advection: 5
		// display: 4*SWAP_CHAIN_SIZE
		const size_t NUM_BUFFERS = 4 + 1 + (4 * 2) + 6 + 5 + (4 * swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * NUM_BUFFERS);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * (6 + swapChainImages.size()));

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}

	}

	void createDisplayDescriptorSets() {
		// each frame in flight can write to each swap chain image
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size() * MAX_FRAMES_IN_FLIGHT, displayShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size() * MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		displayDescriptorSets.resize(swapChainImages.size() * MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, displayDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		Utils::createImageSampler(device, imageSampler);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			for (size_t j = 0; j < swapChainImages.size(); j++) {
				size_t idx = i * swapChainImages.size() + j;
				Utils::bindImage(device, swapChainImageViews[j], imageSampler, displayDescriptorSets[idx], 0);
				Utils::bindBuffer(device, solidBuffers[i], displayDescriptorSets[idx], 1);
				Utils::bindBuffer(device, velocityUBuffers[MAX_FRAMES_IN_FLIGHT + i], displayDescriptorSets[idx], 2);
				Utils::bindBuffer(device, velocityVBuffers[MAX_FRAMES_IN_FLIGHT + i], displayDescriptorSets[idx], 3);
				Utils::bindBuffer(device, pressureBuffers[i], displayDescriptorSets[idx], 4);
			}
		}
	}

	void createAdvectionDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, advectionShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		advectionDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, advectionDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

			Utils::bindBuffer(device, solidBuffers[i], advectionDescriptorSets[i], 0);

			Utils::bindBuffer(device, velocityUBuffers[i], advectionDescriptorSets[i], 1);
			Utils::bindBuffer(device, velocityVBuffers[i], advectionDescriptorSets[i], 2);

			Utils::bindBuffer(device, velocityUBuffers[MAX_FRAMES_IN_FLIGHT + i], advectionDescriptorSets[i], 3);
			Utils::bindBuffer(device, velocityVBuffers[MAX_FRAMES_IN_FLIGHT + i], advectionDescriptorSets[i], 4);
		}
	}

	void createDivergenceDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, divergenceShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		divergenceDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, divergenceDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

			Utils::bindBuffer(device, solidBuffers[i], divergenceDescriptorSets[i], 0);

			// i_old is index of previous frame
			size_t i_old = (i - 1) % MAX_FRAMES_IN_FLIGHT;
			Utils::bindBuffer(device, velocityUBuffers[MAX_FRAMES_IN_FLIGHT + i_old], divergenceDescriptorSets[i], 1);
			Utils::bindBuffer(device, velocityVBuffers[MAX_FRAMES_IN_FLIGHT + i_old], divergenceDescriptorSets[i], 2);

			Utils::bindBuffer(device, divergenceBuffers[i], divergenceDescriptorSets[i], 3);
		}
	}

	void createPressureDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT * 2, pressureShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);
		allocInfo.pSetLayouts = layouts.data();

		pressureDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT * 2);
		if (vkAllocateDescriptorSets(device, &allocInfo, pressureDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

			Utils::bindBuffer(device, solidBuffers[i], pressureDescriptorSets[2 * i], 0);
			Utils::bindBuffer(device, divergenceBuffers[i], pressureDescriptorSets[2 * i], 1);
			Utils::bindBuffer(device, pressureBuffers[i], pressureDescriptorSets[2 * i], 2);
			Utils::bindBuffer(device, pressureBuffers[MAX_FRAMES_IN_FLIGHT + i], pressureDescriptorSets[2 * i], 3);

			// pingpong
			Utils::bindBuffer(device, solidBuffers[i], pressureDescriptorSets[2 * i + 1], 0);
			Utils::bindBuffer(device, divergenceBuffers[i], pressureDescriptorSets[2 * i + 1], 1);
			Utils::bindBuffer(device, pressureBuffers[MAX_FRAMES_IN_FLIGHT + i], pressureDescriptorSets[2 * i + 1], 2);
			Utils::bindBuffer(device, pressureBuffers[i], pressureDescriptorSets[2 * i + 1], 3);
		}
	}

	void createClearPressureDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, clearPressureShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		clearPressureDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, clearPressureDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			Utils::bindBuffer(device, pressureBuffers[i], clearPressureDescriptorSets[i], 0);
		}
	}

	void createApplyPressureDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, applyPressureShader.descLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		applyPressureDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, applyPressureDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

			Utils::bindBuffer(device, solidBuffers[i], applyPressureDescriptorSets[i], 0);

			Utils::bindBuffer(device, pressureBuffers[i], applyPressureDescriptorSets[i], 1);

			// i_old is index of previous frame
			size_t i_old = (i - 1) % MAX_FRAMES_IN_FLIGHT;
			Utils::bindBuffer(device, velocityUBuffers[MAX_FRAMES_IN_FLIGHT + i_old], applyPressureDescriptorSets[i], 2);
			Utils::bindBuffer(device, velocityVBuffers[MAX_FRAMES_IN_FLIGHT + i_old], applyPressureDescriptorSets[i], 3);

			Utils::bindBuffer(device, velocityUBuffers[i], applyPressureDescriptorSets[i], 4);
			Utils::bindBuffer(device, velocityVBuffers[i], applyPressureDescriptorSets[i], 5);
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
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		//vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
		recordCommandBuffer(currentFrame, imageIndex);


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