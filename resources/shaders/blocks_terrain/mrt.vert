#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 3) in vec2 position_offset;
layout(location = 5) in vec4 highlight;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(location=1) out vec3 rnormal;
layout(location=2) out vec3 rpos;
layout(location=3) out vec3 out_color;
layout(location=5) out vec4 out_hightlight;

void main() {
    mat4 worldview = uniforms.view;// * uniforms.world;

    vec3 s_pos = position;
    s_pos.x += position_offset.x;
    s_pos.z -= position_offset.y;

    gl_Position = uniforms.proj * worldview * vec4(s_pos, 1.0);

    rpos = s_pos;
    rnormal = normal;
    out_color = color;
    out_hightlight = highlight;
}
