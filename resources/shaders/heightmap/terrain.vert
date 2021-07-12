#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(location=1) out vec3 rnormal;
layout(location=2) out vec3 rpos;
layout(location=3) out vec2 rtex;
void main() {
    mat4 worldview = uniforms.view;// * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

    rpos = position;
    rnormal = normal;
    rtex = texcoord;
}

