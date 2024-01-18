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
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <limits>
#include <stdio.h>
#include <vector>

namespace g {
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
glm::vec3 cameraPos = glm::vec3(0.0f, -200.0f, 00.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 00.0f);
float aspect = (float)SCR_WIDTH / SCR_HEIGHT;
glm::mat4 projection =
    glm::perspective(glm::radians(60.0f), aspect, 0.1f, 1000.0f);
int discretization = 30;
} // namespace g

const char *vertexShaderSrc =
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

glm::mat4 getCameraView() {
  glm::vec3 cameraDir = glm::normalize(g::cameraPos - g::cameraTarget);
  glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
  glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDir));
  glm::vec3 cameraUp = glm::cross(cameraDir, cameraRight);
  glm::mat4 view = glm::lookAt(g::cameraPos, g::cameraTarget, cameraUp);
  return view;
}

const float MAX_FLOAT = std::numeric_limits<float>::infinity();
const float MIN_FLOAT = -std::numeric_limits<float>::infinity();

float max3(const glm::vec3 &v) { return std::max(std::max(v.x, v.y), v.z); }
float min3(const glm::vec3 &v) { return std::min(std::min(v.x, v.y), v.z); }

struct Box3 {
  Box3()
      : min(glm::vec3(MAX_FLOAT, MAX_FLOAT, MAX_FLOAT)),
        max(glm::vec3(MIN_FLOAT, MIN_FLOAT, MIN_FLOAT)) {}
  Box3(const glm::vec3 &iMin, const glm::vec3 &iMax) : min(iMin), max(iMax) {}
  glm::vec3 getCenter() { return (max + min) / 2.0f; }
  glm::vec3 getSize() { return max - min; }
  void expandByPoint(const glm::vec3 &iPoint) {
    for (int i = 0; i < 3; ++i) {
      min[i] = std::min(min[i], iPoint[i]);
      max[i] = std::max(max[i], iPoint[i]);
    }
  }

  glm::vec3 min;
  glm::vec3 max;
};

Box3 computeBBox(const std::vector<float> &verticesWithNormals) {
  Box3 out;
  for (int i = 0; i < verticesWithNormals.size() / 6; ++i) {
    glm::vec3 vertex =
        glm::vec3(verticesWithNormals[6 * i], verticesWithNormals[6 * i + 1],
                  verticesWithNormals[6 * i + 2]);
    out.expandByPoint(vertex);
  }
  return out;
}

struct Mesh {
  std::vector<float> vertices;
  std::vector<unsigned int> indices;
};

Mesh importModel(const std::string &iFilename) {
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

  Mesh outMesh;
  outMesh.vertices.reserve(mesh->mNumVertices * 3 * 2);
  for (int i = 0; i < mesh->mNumVertices; ++i) {
    for (int j = 0; j < 3; ++j)
      outMesh.vertices.push_back(mesh->mVertices[i][j]);
    for (int j = 0; j < 3; ++j)
      outMesh.vertices.push_back(mesh->mNormals[i][j]);
  }
  Box3 box = computeBBox(outMesh.vertices);
  glm::vec3 bboxCenter = box.getCenter();
  for (int i = 0; i < mesh->mNumVertices; ++i)
    for (int j = 0; j < 3; ++j)
      outMesh.vertices[6 * i + j] -= bboxCenter[j];

  outMesh.indices.reserve(mesh->mNumFaces * 3);
  for (int i = 0; i < mesh->mNumFaces; ++i)
    for (int j = 0; j < 3; ++j)
      outMesh.indices.push_back(mesh->mFaces[i].mIndices[j]);
  return outMesh;
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

struct OpenGLObject {
  GLuint vbo, vao, ebo, instanceVBO;
  OpenGLObject(const Mesh &mesh, std::vector<glm::mat4> &models) {
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &ebo);
    glGenBuffers(1, &instanceVBO);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mesh.vertices.size(),
                 mesh.vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(mesh.indices[0]) * mesh.indices.size(),
                 mesh.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    bindMatrices(models);
  }

  void release() {
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &instanceVBO);
  }

  void bindMatrices(std::vector<glm::mat4> &models) {
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
  }
};

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

__global__ void raycast(size_t nIndices, float *vertices, unsigned int *indices,
                        unsigned char *output, glm::ivec3 dims,
                        glm::vec3 offset, float voxelSize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n = dims[0] * dims[1] * dims[2];

  for (int v = index; v < n; v += stride) {
    int i = v / dims[1] / dims[2];
    int j = (v / dims[2]) % dims[1];
    int k = v % dims[2];
    glm::vec3 pos = offset + (glm::vec3(i, j, k) + glm::vec3(0.5f)) * voxelSize;

    glm::vec3 vert[3];
    for (int idx = 0; idx < nIndices; idx += 3) {
      for (int verti = 0; verti < 3; ++verti)
        for (int vertj = 0; vertj < 3; ++vertj)
          vert[verti][vertj] = vertices[3 * indices[idx + verti] + vertj];
      float t;
      output[v] += rayTriangleIntersect(pos, glm::vec3(0, 0, 1), vert, t);
    }
  }
}

struct cuMesh {
  float *vertices;
  unsigned int *indices;
  cuMesh(const Mesh &iMesh) {
    cudaMallocManaged(&vertices, iMesh.vertices.size() / 2 * sizeof(float));
    cudaMallocManaged(&indices, iMesh.indices.size() * sizeof(unsigned int));
    for (int i = 0; i < iMesh.vertices.size() / 6; ++i)
      for (int j = 0; j < 3; ++j)
        vertices[i * 3 + j] = iMesh.vertices[i * 6 + j];
    std::memcpy(indices, iMesh.indices.data(),
                iMesh.indices.size() * sizeof(iMesh.indices[0]));
  }
};

struct RigidBody {
  glm::vec3 pos = glm::vec3(0, 0, 0); // center of mass
  glm::quat quat = glm::quat(1, 0, 0, 0);
  glm::vec3 V = glm::vec3(0, 0, 0);
  glm::vec3 W = glm::vec3(0, 0, 0);

  std::vector<glm::vec3> getRelativePositions(
      const std::vector<glm::vec3> &initialRelativePositions) const {
    std::vector<glm::vec3> relativePositions;
    std::transform(initialRelativePositions.cbegin(),
                   initialRelativePositions.cend(),
                   std::back_inserter(relativePositions),
                   [this](const glm::vec3 &initialRelativePos) {
                     return quat * initialRelativePos * glm::conjugate(quat);
                   });
    return relativePositions;
  }

  std::vector<glm::vec3>
  getPositions(const std::vector<glm::vec3> &relativePositions) const {
    std::vector<glm::vec3> positions;
    std::transform(
        relativePositions.cbegin(), relativePositions.cend(),
        std::back_inserter(positions),
        [this](const glm::vec3 &relativePos) { return pos + relativePos; });
    return positions;
  }

  glm::mat4 getWorldMatrix() {
    return glm::translate(glm::mat4(1.0), pos) * glm::toMat4(quat);
  }
};

std::vector<glm::mat4>
getParticleMatrices(const std::vector<glm::vec3> &positions, float ballScale) {
  std::vector<glm::mat4> matrices;
  std::transform(positions.cbegin(), positions.cend(),
                 std::back_inserter(matrices),
                 [ballScale](const glm::vec3 &position) {
                   return glm::scale(glm::translate(glm::mat4(1.0), position),
                                     glm::vec3(ballScale));
                 });
  return matrices;
}

std::vector<glm::vec3>
getVelocities(const RigidBody &body,
              const std::vector<glm::vec3> &relativePositions) {
  std::vector<glm::vec3> velocities;
  std::transform(relativePositions.cbegin(), relativePositions.cend(),
                 std::back_inserter(velocities),
                 [&body](const glm::vec3 &relativePos) {
                   return body.V + glm::cross(body.W, relativePos);
                 });
  return velocities;
}

int main(int argc, char *argp[]) {
  assert(argc == 3);
  Mesh bear = importModel(std::string(argp[1]));
  Mesh ball = importModel(std::string(argp[2]));

  GLFWwindow *window = initGLFW();
  assert(window);

  Box3 bearBBox = computeBBox(bear.vertices);
  Box3 ballBBox = computeBBox(ball.vertices);

  float voxelSize = min3(bearBBox.getSize()) / (float)g::discretization;
  float ballScale = voxelSize / max3(ballBBox.getSize());

  cuMesh cuBear(bear);

  glm::ivec3 dims = glm::ivec3(bearBBox.getSize() / voxelSize);

  int N = dims[0] * dims[1] * dims[2];
  unsigned char *voxels;
  cudaMallocManaged(&voxels, N * sizeof(unsigned char));
  memset(voxels, 0, N * sizeof(unsigned char));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  raycast<<<numBlocks, blockSize>>>(bear.indices.size(), cuBear.vertices,
                                    cuBear.indices, voxels, dims, bearBBox.min,
                                    voxelSize);
  cudaDeviceSynchronize();

  std::vector<glm::vec3> particleInitialRelativePositions;
  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int k = 0; k < dims[2]; ++k) {
        if (voxels[i * dims[1] * dims[2] + j * dims[2] + k] % 2 == 0)
          continue;
        glm::vec3 pos = bearBBox.min +
                        (glm::vec3(i, j, k) + glm::vec3(0.5f)) * voxelSize -
                        bearBBox.getCenter();
        particleInitialRelativePositions.push_back(pos);
      }
    }
  }

  std::vector<glm::vec3> particleRelativePositions;
  std::vector<glm::vec3> particlePositions;

  std::vector<glm::mat4> bearMatrices;
  std::vector<glm::mat4> mergedParticleMatrices;

  std::vector<RigidBody> bears = {{.pos = glm::vec3(0, 0, 0),
                                   .quat = glm::quat(1, 0, 0, 0),
                                   .V = glm::vec3(0, 0, 0),
                                   .W = glm::vec3(0, 0, 0)},
                                  {.pos = glm::vec3(50, 0, 0),
                                   .quat = glm::quat(1, 0, 0, 0),
                                   .V = glm::vec3(0, 0, 0),
                                   .W = glm::vec3(0, 0, 0)},
                                  {.pos = glm::vec3(-50, 0, 0),
                                   .quat = glm::quat(1, 0, 0, 0),
                                   .V = glm::vec3(0, 0, 0),
                                   .W = glm::vec3(0, 0, 0)}};
  for (auto freddy : bears) {
    particleRelativePositions =
        freddy.getRelativePositions(particleInitialRelativePositions);
    particlePositions = freddy.getPositions(particleRelativePositions);
    std::vector<glm::mat4> particleMatrices =
        getParticleMatrices(particlePositions, ballScale);

    mergedParticleMatrices.insert(mergedParticleMatrices.end(),
                                  particleMatrices.begin(),
                                  particleMatrices.end());
    bearMatrices.push_back(freddy.getWorldMatrix());
  }

  GLuint shader = compileShader(vertexShaderSrc, fragmentShaderSrc);

  OpenGLObject bearGL(bear, bearMatrices);
  OpenGLObject ballGL(ball, mergedParticleMatrices);

  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view = getCameraView();

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                       glm::value_ptr(g::projection));
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                       glm::value_ptr(view));
    glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1,
                 glm::value_ptr(g::cameraPos));

    ballGL.bindMatrices(mergedParticleMatrices);

    if (showBearBalls & 0b01) {
      glBindVertexArray(bearGL.vao);
      glDrawElementsInstanced(GL_TRIANGLES, bear.indices.size(),
                              GL_UNSIGNED_INT, 0, bearMatrices.size());
      glBindVertexArray(0);
    }

    if (showBearBalls & 0b10) {
      glBindVertexArray(ballGL.vao);
      glDrawElementsInstanced(GL_TRIANGLES, ball.indices.size(),
                              GL_UNSIGNED_INT, 0,
                              mergedParticleMatrices.size());
      glBindVertexArray(0);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  bearGL.release();
  ballGL.release();
  glfwTerminate();
  return 0;
}