#version 450

layout(location = 0) in vec3 position;

layout(location = 3) in vec2 position_offset;
layout(location = 4) in vec4 object_id;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(location=1) out vec4 out_color;

void main() {
    mat4 worldview = uniforms.view;

    vec4 s_pos = vec4(
    position.x + position_offset.x,
    position.y,
    position.z - position_offset.y,
    1.0
    );

    gl_Position = uniforms.proj * worldview * s_pos;
    out_color = object_id;
}
