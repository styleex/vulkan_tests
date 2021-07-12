#version 450

layout(location = 0) in vec2 position;
layout (location = 1) out vec2 outUV;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

}
