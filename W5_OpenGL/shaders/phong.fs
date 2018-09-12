#version 420                                            // Keeping you on the bleeding edge!
#extension GL_EXT_gpu_shader4 : enable

// Attributes passed on from the vertex shader
smooth in vec3 o_VertexPosition;
smooth in vec3 o_VertexNormal;
smooth in vec2 o_TexCoord;

// Structure for holding light parameters
struct LightInfo {
    vec4 Position; // Light position in eye coords.
    vec3 La; // Ambient light intensity
    vec3 Ld; // Diffuse light intensity
    vec3 Ls; // Specular light intensity
};

// We'll have a single light in the scene
uniform LightInfo u_Light;

// The material properties of our object
struct MaterialInfo {
    vec3 Ka; // Ambient reflectivity
    vec3 Kd; // Diffuse reflectivity
    vec3 Ks; // Specular reflectivity
    float Shininess; // Specular shininess factor
};

// The object has a material
uniform MaterialInfo u_Material;

// The texture to be mapped
uniform sampler2D u_Texture;

// This is no longer a built-in variable
out vec4 o_FragColor;

void main() {
    // Calculate the normal
    vec3 n = normalize( o_VertexNormal );

    // Calculate the light vector
    vec3 s = normalize( vec3(u_Light.Position) - o_VertexPosition );

    // Calculate the vertex position
    vec3 v = normalize(vec3(-o_VertexPosition));

    // Reflect the light about the surface normal
    vec3 r = reflect( -s, n );

    // Compute the light from the ambient, diffuse and specular components
    vec3 lightColor = (
            u_Light.La * u_Material.Ka +
            u_Light.Ld * u_Material.Kd * max( dot(s, n), 0.0 ) +
            u_Light.Ls * u_Material.Ks * pow( max( dot(r,v), 0.0 ), u_Material.Shininess ));

    // Set the output color of our current pixel
    o_FragColor = vec4(lightColor,1.0); // * texture(u_Texture, o_TexCoord);
}
