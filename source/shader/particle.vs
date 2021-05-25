#version 430 core

layout(location = 0) in vec3 in_Quad;
layout(location = 1) in vec2 in_Tex;
layout(location = 2) in vec3 in_Color;
layout(location = 3) in mat4 in_Model;

uniform mat4 vp;

out vec2 ex_Tex;
out vec3 ex_Color;

void main(void) {
	//Pass Texture Coordinates
	ex_Tex = in_Tex;
	ex_Color = in_Color;

	//Actual Position in Space
	gl_Position = vp * in_Model * vec4(in_Quad, 1.0f);
}
