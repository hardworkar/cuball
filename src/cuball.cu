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

// clang-format off
#define CHECK(e) \
do { if (!(e)) { fprintf(stderr, "! %s:%d\n", __FILE__, __LINE__); exit(-1); } } while (0)
// clang-format on

namespace globals {
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
glm::vec3 cameraPos = glm::vec3(0.0f, -200.0f, 00.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 00.0f);
int numOfModels = 100;
int discretization = 20;
} // namespace globals

std::pair<std::vector<float>, std::vector<unsigned int>>
importModel(const std::string &iFilename, bool iRescale) {
  static Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      iFilename, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                     aiProcess_JoinIdenticalVertices | aiProcess_SortByPType |
                     aiProcess_GenSmoothNormals);
  if (!scene || scene->mNumMeshes != 1 || scene->mMeshes[0]->mNormals == NULL) {
    fprintf(stderr, "Failed to import %s: %s\n", iFilename.c_str(),
            importer.GetErrorString());
    CHECK(false);
  }
  aiMesh *mesh = scene->mMeshes[0];
  CHECK(mesh);
  fprintf(stdout, "Loaded model '%s': faces: %d vertices: %d\n",
          iFilename.c_str(), mesh->mNumFaces, mesh->mNumVertices);

  std::pair<glm::vec3, glm::vec3> bbox = {glm::vec3(1e6, 1e6, 1e6),
                                          glm::vec3(-1e6, -1e6, -1e6)};
  for (int i = 0; i < mesh->mNumVertices; ++i) {
    for (int j = 0; j < 3; ++j) {
      bbox.first[j] = std::min(bbox.first[j], mesh->mVertices[i][j]);
      bbox.second[j] = std::max(bbox.second[j], mesh->mVertices[i][j]);
    }
  }
  glm::vec3 bboxCenter = (bbox.second + bbox.first) / 2.0f;
  std::vector<float> vertices;
  vertices.reserve(mesh->mNumVertices * 3 * 2);
  for (int i = 0; i < mesh->mNumVertices; ++i) {
    for (int j = 0; j < 3; ++j)
      vertices.push_back(mesh->mVertices[i][j] - bboxCenter[j]);
    for (int j = 0; j < 3; ++j)
      vertices.push_back(mesh->mNormals[i][j]);
  }

  std::vector<unsigned int> indices;
  indices.reserve(mesh->mNumFaces * 3);
  for (auto i = 0; i < mesh->mNumFaces; ++i) {
    for (auto j = 0; j < 3; ++j)
      indices.push_back(mesh->mFaces[i].mIndices[j]);
  }
  return {vertices, indices};
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

GLFWwindow *initGLFW() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(globals::SCR_WIDTH, globals::SCR_HEIGHT,
                                        "CUBALL", NULL, NULL);
  CHECK(window);

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  CHECK(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress));

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  return window;
}

GLuint compileShader(const char *iVertexShaderSource,
                     const char *iFragmentShaderSource) {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &iVertexShaderSource, NULL);
  glCompileShader(vertexShader);

  int success = 0;
  char infoLog[512];

  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    fprintf(stdout, "vertex shader compilation log: %s\n", infoLog);
    CHECK(false);
  }

  GLuint fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &iFragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    fprintf(stdout, "fragment shader compilation log: %s\n", infoLog);
    CHECK(false);
  }

  GLuint shader = glCreateProgram();
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

  return shader;
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  auto &cameraPos = globals::cameraPos;
  float r = sqrt(cameraPos.x * cameraPos.x + cameraPos.y * cameraPos.y);
  float phi = atan2(cameraPos.y, cameraPos.x);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    r -= 0.3;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    r += 0.3;
  cameraPos.x = r * cos(phi);
  cameraPos.y = r * sin(phi);
}

std::pair<GLuint, GLuint> bindModel(std::vector<float> &vertices,
                                    std::vector<unsigned int> &indices,
                                    std::vector<glm::mat4> &models) {
  GLuint VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(),
               vertices.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(),
               indices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  unsigned int instanceVBO;
  glGenBuffers(1, &instanceVBO);
  glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * models.size(), &models[0],
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
  for (int pos = 0; pos < 4; ++pos) {
    glEnableVertexAttribArray(pos + 2);
    glVertexAttribPointer(pos + 2, 4, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat) * 4 * 4,
                          (void *)(sizeof(float) * pos * 4));
    glVertexAttribDivisor(pos + 2, 1);
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  return {VAO, VBO};
}

std::pair<glm::vec3, glm::vec3>
computeBBox(const std::vector<float> &iVertices) {
  std::pair<glm::vec3, glm::vec3> bbox = {glm::vec3(1e6, 1e6, 1e6),
                                          glm::vec3(-1e6, -1e6, -1e6)};
  for (int i = 0; i < iVertices.size() / 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      bbox.first[j] = std::min(bbox.first[j], iVertices[3 * i + j]);
      bbox.second[j] = std::max(bbox.second[j], iVertices[3 * i + j]);
    }
  }
  return bbox;
}

int main(int argc, char *argp[]) {
  CHECK(argc == 3);

  auto [bearVertices, bearIndices] = importModel(std::string(argp[1]), false);
  auto [ballVertices, ballIndices] = importModel(std::string(argp[2]), false);

  GLFWwindow *window = initGLFW();
  CHECK(window);

  std::pair<glm::vec3, glm::vec3> bearBBox = computeBBox(bearVertices);
  std::pair<glm::vec3, glm::vec3> ballBBox = computeBBox(ballVertices);
  float ballScale = 0.0f;
  for (int i = 0; i < 3; ++i) {
    ballScale = std::max(ballScale, ballBBox.second[i] - ballBBox.first[i]);
  }

  float vxlSize = 100000.0f;
  for (int i = 0; i < 3; ++i) {
    vxlSize = std::min(vxlSize, (bearBBox.second[i] - bearBBox.first[i]) /
                                    globals::discretization);
  }
  std::vector<glm::mat4> ballsMatrices;
  for (int i = 0; i < (bearBBox.second[0] - bearBBox.first[0]) / vxlSize; ++i) {
    for (int j = 0; j < (bearBBox.second[1] - bearBBox.first[1]) / vxlSize;
         ++j) {
      for (int k = 0; k < (bearBBox.second[2] - bearBBox.first[2]) / vxlSize;
           ++k) {
        ballsMatrices.push_back(glm::scale(
            glm::translate(glm::mat4(1.0),
                           glm::vec3(bearBBox.first[0] + (i + 0.5f) * vxlSize,
                                     bearBBox.first[1] + (j + 0.5f) * vxlSize,
                                     bearBBox.first[2] + (k + 0.5f) * vxlSize)),
            glm::vec3(vxlSize / ballScale)));
      }
    }
  }

  std::string vertexShaderSource =
      "#version 330 core\n"
      "layout (location = 0) in vec3 aPos;\n"
      "layout (location = 1) in vec3 aNormal;\n"
      "layout (location = 2) in mat4 aModel;\n"
      "uniform mat4 view;\n"
      "uniform mat4 projection;\n"
      "out vec3 FragPos;\n"
      "out vec3 Normal;\n"
      "void main() {\n"
      "  gl_Position = projection * view * aModel * vec4(aPos, 1.0);\n"
      "  Normal = mat3(transpose(inverse(aModel))) * aNormal;"
      "  FragPos = vec3(aModel * vec4(aPos, 1.0));\n"
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
      "  vec3 ambient = ambientStrength * lightColor;\n"
      "  FragColor = vec4(ambient + diffuse, 0.0);\n"
      "}\0";

  GLuint shader =
      compileShader(vertexShaderSource.c_str(), fragmentShaderSource);

  std::vector<glm::mat4> bearMatrices{glm::mat4(1.0)};
  const auto [bearVAO, bearVBO] =
      bindModel(bearVertices, bearIndices, bearMatrices);
  const auto [ballVAO, ballVBO] =
      bindModel(ballVertices, ballIndices, ballsMatrices);
  glm::mat4 projection = glm::perspective(
      glm::radians(60.0f), (float)globals::SCR_WIDTH / globals::SCR_HEIGHT,
      0.1f, 1000.0f);

  glm::vec3 lightPos = globals::cameraPos;

  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::vec3 cameraDirection =
        glm::normalize(globals::cameraPos - globals::cameraTarget);
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    glm::vec3 cameraUp = glm::cross(cameraDirection, cameraRight);
    glm::mat4 view =
        glm::lookAt(globals::cameraPos, globals::cameraTarget, cameraUp);
    lightPos = globals::cameraPos;

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                       glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1,
                 glm::value_ptr(lightPos));

    glBindVertexArray(bearVAO);
    glDrawElementsInstanced(GL_TRIANGLES, bearIndices.size(), GL_UNSIGNED_INT,
                            0, bearMatrices.size());

    glBindVertexArray(ballVAO);
    glDrawElementsInstanced(GL_TRIANGLES, ballIndices.size(), GL_UNSIGNED_INT,
                            0, ballsMatrices.size());

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteBuffers(1, &bearVBO);
  glDeleteBuffers(1, &ballVBO);
  glfwTerminate();
  return 0;
}