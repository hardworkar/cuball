#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#undef NDEBUG // assertation supremacy.
#include <cassert>
#include <chrono>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/string_cast.hpp>
#include <limits>
#include <stdio.h>
#include <vector>

namespace g {
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
glm::vec3 cameraPos = glm::vec3(0.0f, -200.0f, 00.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 00.0f);
int discretization = 30;
} // namespace g

std::pair<glm::vec3, glm::vec3>
computeBBox(const std::vector<float> &iVerticesWithNormals) {
  std::pair<glm::vec3, glm::vec3> b = {glm::vec3(1e6), glm::vec3(-1e6)};
  for (int i = 0; i < iVerticesWithNormals.size() / 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      b.first[j] = std::min(b.first[j], iVerticesWithNormals[6 * i + j]);
      b.second[j] = std::max(b.second[j], iVerticesWithNormals[6 * i + j]);
    }
  }
  return b;
}

std::pair<std::vector<float>, std::vector<unsigned int>>
importModel(const std::string &iFilename) {
  static Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      iFilename, aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices |
                     aiProcess_SortByPType | aiProcess_GenSmoothNormals);
  assert(scene && scene->mNumMeshes == 1 &&
         scene->mMeshes[0]->mNormals != NULL);
  aiMesh *mesh = scene->mMeshes[0];
  assert(mesh);
  printf("Loaded model '%s': faces: %d vertices: %d\n", iFilename.c_str(),
         mesh->mNumFaces, mesh->mNumVertices);

  std::vector<float> vertices;
  vertices.reserve(mesh->mNumVertices * 3 * 2);
  for (int i = 0; i < mesh->mNumVertices; ++i) {
    for (int j = 0; j < 3; ++j)
      vertices.push_back(mesh->mVertices[i][j]);
    for (int j = 0; j < 3; ++j)
      vertices.push_back(mesh->mNormals[i][j]);
  }
  std::pair<glm::vec3, glm::vec3> bbox = computeBBox(vertices);
  glm::vec3 bboxCenter = (bbox.first + bbox.second) / 2.0f;
  for (int i = 0; i < mesh->mNumVertices; ++i)
    for (int j = 0; j < 3; ++j)
      vertices[6 * i + j] -= bboxCenter[j];

  std::vector<unsigned int> indices;
  indices.reserve(mesh->mNumFaces * 3);
  for (int i = 0; i < mesh->mNumFaces; ++i)
    for (int j = 0; j < 3; ++j)
      indices.push_back(mesh->mFaces[i].mIndices[j]);
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
  GLFWwindow *window =
      glfwCreateWindow(g::SCR_WIDTH, g::SCR_HEIGHT, "CUBALL", NULL, NULL);
  assert(window);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  assert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress));
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  return window;
}

GLuint compileShader(const char *ivertexShaderSrc,
                     const char *ifragmentShaderSrc) {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &ivertexShaderSrc, NULL);
  glCompileShader(vertexShader);

  int success = 0;
  char infoLog[512];

  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    printf("vertex shader compilation log: %s\n", infoLog);
    assert(false);
  }

  GLuint fragmentShader;
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &ifragmentShaderSrc, NULL);
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    printf("fragment shader compilation log: %s\n", infoLog);
    assert(false);
  }

  GLuint shader = glCreateProgram();
  glAttachShader(shader, vertexShader);
  glAttachShader(shader, fragmentShader);
  glLinkProgram(shader);

  glGetProgramiv(shader, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shader, 512, NULL, infoLog);
    printf("shader linking log: %s\n", infoLog);
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return shader;
}

unsigned int showBearBalls = 0b01;
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  float r2 = g::cameraPos.x * g::cameraPos.x + g::cameraPos.y * g::cameraPos.y;
  float phi = atan2(g::cameraPos.y, g::cameraPos.x);

  float rDiff = (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) -
                (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS);
  float r = sqrt(r2) + rDiff * 0.3;
  float phiDiff = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) -
                  (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
  phi += phiDiff * 0.1;
  g::cameraPos = glm::vec3(r * cos(phi), r * sin(phi), g::cameraPos.z);

  static bool lastBState = true;
  bool currState = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
  showBearBalls += currState && !lastBState;
  showBearBalls = showBearBalls > 0b11 ? 0b01 : showBearBalls;
  lastBState = currState;
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

__device__ bool rayTriangleIntersect(const glm::vec3 &orig,
                                     const glm::vec3 &dir,
                                     const glm::vec3 vert[3], float &t) {
  glm::vec3 N = glm::cross(vert[1] - vert[0], vert[2] - vert[0]);
  float NdotRayDirection = glm::dot(N, dir);
  if (fabs(NdotRayDirection) < 1e-6)
    return false;

  float d = glm::dot(-N, vert[0]);

  t = -(glm::dot(N, orig) + d) / NdotRayDirection;
  if (t < 0)
    return false;

  glm::vec3 P = orig + t * dir;
  for (int v = 0; v < 3; ++v) {
    glm::vec3 C = glm::cross(vert[(v + 1) % 3] - vert[v], P - vert[v]);
    if (glm::dot(N, C) < 0)
      return false;
  }
  return true;
}

__global__ void raycast(size_t indicesSize, float *gpuVertices,
                        unsigned int *gpuIndices, unsigned char *gpuOutput,
                        unsigned int mi, unsigned int mj, unsigned int mk,
                        glm::vec3 offset, float vxlSize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n = mi * mj * mk;

  for (int v = index; v < n; v += stride) {
    int i = v / mj / mk;
    int j = (v / mk) % mj;
    int k = v % mk;
    glm::vec3 pos = offset + (glm::vec3(i, j, k) + glm::vec3(0.5f)) * vxlSize;

    glm::vec3 vert[3];
    for (int idx = 0; idx < indicesSize; idx += 3) {
      for (int verti = 0; verti < 3; ++verti)
        for (int vertj = 0; vertj < 3; ++vertj)
          vert[verti][vertj] = gpuVertices[3 * gpuIndices[idx + verti] + vertj];
      float t;
      gpuOutput[v] += rayTriangleIntersect(pos, glm::vec3(0, 0, 1), vert, t);
    }
  }
}

int main(int argc, char *argp[]) {
  assert(argc == 3);
  auto [bearVertexData, bearIndices] = importModel(std::string(argp[1]));
  auto [ballVertexData, ballIndices] = importModel(std::string(argp[2]));

  GLFWwindow *window = initGLFW();
  assert(window);

  std::pair<glm::vec3, glm::vec3> bearBBox = computeBBox(bearVertexData);
  std::pair<glm::vec3, glm::vec3> ballBBox = computeBBox(ballVertexData);
  float ballScale = std::numeric_limits<float>::min();
  for (int i = 0; i < 3; ++i)
    ballScale = std::max(ballScale, ballBBox.second[i] - ballBBox.first[i]);

  float vxlSize = std::numeric_limits<float>::max();
  for (int i = 0; i < 3; ++i)
    vxlSize = std::min(vxlSize, (bearBBox.second[i] - bearBBox.first[i]) /
                                    g::discretization);

  float *gpuVertices;
  unsigned int *gpuIndices;
  cudaMallocManaged(&gpuVertices, bearVertexData.size() / 2 * sizeof(float));
  cudaMallocManaged(&gpuIndices, bearIndices.size() * sizeof(unsigned int));
  for (int i = 0; i < bearVertexData.size() / 6; ++i)
    for (int j = 0; j < 3; ++j)
      gpuVertices[i * 3 + j] = bearVertexData[i * 6 + j];
  std::memcpy(gpuIndices, bearIndices.data(),
              bearIndices.size() * sizeof(bearIndices[0]));

  std::vector<glm::mat4> relativePositions;
  glm::ivec3 dims = glm::ivec3((bearBBox.second - bearBBox.first) / vxlSize);

  int N = dims[0] * dims[1] * dims[2];
  unsigned char *gpuOutput;
  cudaMallocManaged(&gpuOutput, N * sizeof(unsigned char));
  memset(gpuOutput, 0, N * sizeof(unsigned char));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  raycast<<<numBlocks, blockSize>>>(bearIndices.size(), gpuVertices, gpuIndices,
                                    gpuOutput, dims[0], dims[1], dims[2],
                                    bearBBox.first, vxlSize);
  cudaDeviceSynchronize();

  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int k = 0; k < dims[2]; ++k) {
        glm::vec3 pos =
            bearBBox.first + (glm::vec3(i, j, k) + glm::vec3(0.5f)) * vxlSize;
        if (gpuOutput[i * dims[1] * dims[2] + j * dims[2] + k] % 2)
          relativePositions.push_back(
              glm::scale(glm::translate(glm::mat4(1.0), pos),
                         glm::vec3(vxlSize / ballScale)));
      }
    }
  }
  std::vector<glm::mat4> bearMatrices{glm::mat4(1.0)};

  std::string vertexShaderSrc =
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
      "  vec3 normal = normalize(aNormal);\n"
      "  Normal = mat3(transpose(inverse(aModel))) * normal;"
      "  FragPos = vec3(aModel * vec4(aPos, 1.0));\n"
      "}\0";

  const char *fragmentShaderSrc =
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

  GLuint shader = compileShader(vertexShaderSrc.c_str(), fragmentShaderSrc);

  auto [bearVAO, bearVBO] =
      bindModel(bearVertexData, bearIndices, bearMatrices);
  auto [ballVAO, ballVBO] =
      bindModel(ballVertexData, ballIndices, relativePositions);
  glm::mat4 projection = glm::perspective(
      glm::radians(60.0f), (float)g::SCR_WIDTH / g::SCR_HEIGHT, 0.1f, 1000.0f);

  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::vec3 cameraDir = glm::normalize(g::cameraPos - g::cameraTarget);
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDir));
    glm::vec3 cameraUp = glm::cross(cameraDir, cameraRight);
    glm::mat4 view = glm::lookAt(g::cameraPos, g::cameraTarget, cameraUp);

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                       glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1,
                 glm::value_ptr(g::cameraPos));

    if (showBearBalls & 0b01) {
      glBindVertexArray(bearVAO);
      glDrawElementsInstanced(GL_TRIANGLES, bearIndices.size(), GL_UNSIGNED_INT,
                              0, bearMatrices.size());
    }

    if (showBearBalls & 0b10) {
      glBindVertexArray(ballVAO);
      glDrawElementsInstanced(GL_TRIANGLES, ballIndices.size(), GL_UNSIGNED_INT,
                              0, relativePositions.size());
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteBuffers(1, &bearVBO);
  glDeleteBuffers(1, &ballVBO);
  glfwTerminate();
  return 0;
}