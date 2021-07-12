#version 450

layout(binding = 0) uniform sampler2DMS tex;

layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec4 f_color;

layout(location = 0) out vec4 Target0;

void main() {
    vec3 ret = vec3(0.0);
    ivec2 sz = textureSize(tex);
    for (int i = 0; i < 4; i++) {
        ret += texelFetch(tex, ivec2(f_uv.xy * sz), i).rgb;
    }

    Target0 = vec4(ret / 4, 1.0);
}
