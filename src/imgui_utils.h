#pragma once

#include <volk.h>

struct GLFWwindow;

void imInit(GLFWwindow* window, VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, uint32_t queue_family, VkQueue queue, uint32_t sc_images, VkFormat sc_format);
void imShutdown();

void imBeginFrame();
void imEndAndRender(VkCommandBuffer command_buffer, VkImageView sc_img_view, VkRect2D viewport);

bool imWantCaptureMouse();
bool imWantCaptureKeyboard();
