#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_position;


layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_world;
layout(location = 3) in vec3 in_color;
layout(location=5) in vec4 in_hightlight;

void main() {
    f_color = vec4(in_color, 1.0) * in_hightlight.x + vec4(0.0, 0.0, 1.0, 1.0) * (1 - in_hightlight.x);
    f_normal = vec4(in_normal, 1.0);
    f_position = vec4(in_world, 1.0);
}
