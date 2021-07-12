#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_world;
layout(location = 3) in vec2 in_tex;

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    vec3 light_pos = normalize(vec3(0.2, 0.2, 0.2));
    float light_percent = max(-dot(light_pos, in_normal), 0.0);

    f_color = texture(tex, in_tex / 25.0) * min(0.35+light_percent, 1.0);
}

