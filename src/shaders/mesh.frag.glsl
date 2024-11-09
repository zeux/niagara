#version 450

layout(location = 0) out vec4 outputColor;

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(binding = 0, set = 1) uniform sampler2D textures[];

void main()
{
	outputColor = color;
}
