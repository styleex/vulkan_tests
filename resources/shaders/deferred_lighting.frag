#version 450

// The `color_input` parameter of the `draw` method.
layout(set = 0, binding = 0) uniform sampler2DMS u_diffuse;

layout(push_constant) uniform PushConstants {
// The `ambient_color` parameter of the `draw` method.
    vec4 color;
} push_constants;

layout(location = 0) out vec4 f_color;
layout (constant_id = 0) const int NUM_SAMPLES = 8;

layout (location = 1) in vec2 inUV;


void main() {
    vec4 result = vec4(0.0);
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        vec4 val = texelFetch(u_diffuse, ivec2(gl_FragCoord.xy), i);
        result += val;
    }
    // Average resolved samples
    result = result / float(NUM_SAMPLES);

    f_color.rgb = push_constants.color.rgb * result.rgb;
    f_color.a = 1.0;
}
