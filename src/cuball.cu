#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <vector>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

glm::vec3 cameraPos;
glm::vec3 cameraTarget;

glm::vec3 lightPos;

const char *vertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aNormal;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "out vec3 FragPos;\n"
    "out vec3 Normal;\n"
    "void main() {\n"
    "  gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
    "  Normal = mat3(transpose(inverse(model))) * aNormal;"
    "  FragPos = vec3(model * vec4(aPos, 1.0));\n"
    "}\0";

const char *fragmentShaderSource =
    "#version 330 core\n"
    "uniform vec3 lightPos;\n"
    "in vec3 FragPos;\n"
    "in vec3 Normal;\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "  vec3 lightColor = vec3(0.5, 0.1, 0.5);\n"
    "  vec3 lightDir = normalize(lightPos - FragPos);\n"
    "  float diff = max(0.0, dot(Normal, lightDir));\n"
    "  vec3 diffuse = diff * lightColor;\n"
    "  float ambientStrength = 0.5;\n"
    "  vec3 ambient = vec3(0.0, 0.0, 0.0);//ambientStrength * lightColor;\n"
    "  FragColor = vec4(ambient + diffuse, 0.0);\n"
    "}\0";

int main(int argc, char *argp[]) {
  if (argc < 2) {
    fprintf(stderr, "Provide a .stl model as an argument\n");
    return -1;
  }
  Assimp::Importer importer;
  const aiScene *scene =
      importer.ReadFile(std::string(argp[1]),
                        aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                            aiProcess_JoinIdenticalVertices |
                            aiProcess_SortByPType | aiProcess_GenSmoothNormals);
  if (!scene || scene->mNumMeshes != 1 || scene->mMeshes[0]->mNormals == NULL) {
    fprintf(stderr, "Failed to import %s: %s\n", argp[1],
            importer.GetErrorString());
    return -1;
  }

  aiMesh *mesh = scene->mMeshes[0];
  fprintf(stdout, "Loaded model '%s': faces: %d vertices: %d\n", argp[1],
          mesh->mNumFaces, mesh->mNumVertices);

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CUBALL", NULL, NULL);
  if (window == NULL) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    fprintf(stderr, "Failed to initialize GLAD\n");
    return -1;
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  unsigned int vertexShader;
  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);

  int success = 0;
  char infoLog[512];

  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    fprintf(stdout, "vertex shader compilation log: %s\n", infoLog);
  }

  unsigned int fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    fprintf(stdout, "fragment shader compilation log: %s\n", infoLog);
  }

  unsigned int shader;
  shader = glCreateProgram();
  glAttachShader(shader, vertexShader);
  glAttachShader(shader, fragmentShader);
  glLinkProgram(shader);

  glGetProgramiv(shader, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shader, 512, NULL, infoLog);
    fprintf(stdout, "shader linking log: %s\n", infoLog);
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  std::vector<float> vVertices;
  vVertices.reserve(mesh->mNumVertices * 3 * 2);
  for (int i = 0; i < mesh->mNumVertices; ++i) {
    for (int j = 0; j < 3; ++j)
      vVertices.push_back(mesh->mVertices[i][j]);
    for (int j = 0; j < 3; ++j)
      vVertices.push_back(mesh->mNormals[i][j]);
  }
  float *vertices = vVertices.data();

  std::vector<unsigned int> vIndices;
  vIndices.reserve(mesh->mNumFaces * 3);
  for (auto i = 0; i < mesh->mNumFaces; ++i) {
    for (auto j = 0; j < 3; ++j)
      vIndices.push_back(mesh->mFaces[i].mIndices[j]);
  }

  unsigned int *indices = vIndices.data();
  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(vertices[0]) * mesh->mNumVertices,
               vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               3 * sizeof(indices[0]) * mesh->mNumFaces, indices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glm::mat4 projection;
  projection = glm::perspective(glm::radians(60.0f),
                                (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 1000.0f);
  glm::mat4 model = glm::mat4(1.0);

  cameraPos = glm::vec3(0.0f, -100.0f, 43.0f);
  cameraTarget = glm::vec3(0.0f, 0.0f, 43.0f);

  lightPos = cameraPos;

  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader);

    model =
        glm::rotate(glm::mat4(1.0), (float)glfwGetTime() * glm::radians(40.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));

    // camera stuff
    glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    glm::vec3 cameraUp = glm::cross(cameraDirection, cameraRight);
    glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                       glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE,
                       glm::value_ptr(model));

    glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1,
                 glm::value_ptr(lightPos));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, mesh->mNumFaces * 3, GL_UNSIGNED_INT, 0);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteBuffers(1, &VBO);
  glfwTerminate();
  return 0;
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  float r = sqrt(cameraPos.x * cameraPos.x + cameraPos.y * cameraPos.y);
  float phi = atan2(cameraPos.y, cameraPos.x);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    r -= 0.3;
    cameraPos.x = r * cos(phi);
    cameraPos.y = r * sin(phi);
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    r += 0.3;
    cameraPos.x = r * cos(phi);
    cameraPos.y = r * sin(phi);
  }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}