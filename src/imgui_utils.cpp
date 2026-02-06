#include "imgui_utils.h"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

void imInit(GLFWwindow* window, VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, uint32_t queue_family, VkQueue queue, uint32_t sc_images, VkFormat sc_format)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;

	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	// May trigger validation assertion due to the incorrect sync. implementation on ImGui side
	// See https://github.com/ocornut/imgui/issues/8795
	// io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	// Makes all elements transparent, so the background is seen even when window is open
	ImGui::GetStyle().Alpha = 0.75F;

	ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplVulkan_InitInfo init_info = {};

	init_info.ApiVersion = VK_API_VERSION_1_4;
	init_info.Instance = instance;
	init_info.PhysicalDevice = physical_device;
	init_info.Device = device;
	init_info.QueueFamily = queue_family;
	init_info.Queue = queue;
	init_info.PipelineCache = VK_NULL_HANDLE;
	init_info.UseDynamicRendering = true;
	init_info.MinAllocationSize = 1024 * 1024;
	init_info.DescriptorPool = VK_NULL_HANDLE;
	init_info.MinImageCount = 2;
	init_info.ImageCount = sc_images;
	init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;
	init_info.Allocator = nullptr;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {
		VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		nullptr,
		0,
		1,
		&sc_format,
	};

	ImGui_ImplVulkan_Init(&init_info);
}

void imShutdown()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void imBeginFrame()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void imEndAndRender(VkCommandBuffer command_buffer, VkImageView sc_img_view, VkRect2D viewport)
{
	ImGui::Render();

	VkRenderingAttachmentInfo color_attachment_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
	color_attachment_info.imageView = sc_img_view;
	color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	color_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	color_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
	rendering_info.renderArea = viewport;
	rendering_info.layerCount = 1;
	rendering_info.colorAttachmentCount = 1;
	rendering_info.pColorAttachments = &color_attachment_info;

	vkCmdBeginRendering(command_buffer, &rendering_info);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
	vkCmdEndRendering(command_buffer);

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
	}
}

bool imWantCaptureMouse()
{
	return ImGui::GetIO().WantCaptureMouse;
}

bool imWantCaptureKeyboard()
{
	return ImGui::GetIO().WantCaptureKeyboard;
}
