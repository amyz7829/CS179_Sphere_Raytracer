#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "../Eigen/Dense"
#include <fstream>
#include <float.h>
#include <algorithm>
#include <math.h>
#include <sstream>
#include <cuda_runtime.h>
#include "raytracer.cuh"
#include "ta_utilities.hpp"


using namespace std;
using namespace Eigen;

struct color{
  float r;
  float g;
  float b;
};

struct material{
	color ambient;
	color diffuse;
	color specular;
	float shine;
};

struct sphere{
  float position[3];
  float radius;
  material m;
};

struct ray{
  float origin[3];
  float direction[3];
};

struct light{
  float position[3];
  color c;
  float k;
};

struct camera{
  float position[3];
  float orientation[4];
  float near;
	float far;
	float left;
	float right;
	float top;
	float bottom;
};

__host__ void parse_camera_data(ifstream *parser, camera *c);
__host__ void parse_sphere_data(ifstream *parser, sphere *s);
__host__ void parse_light_data(ifstream *parser, light *l);

__device__ Matrix4d makeTranslationMatrix(camera *c);
__device__ Matrix4d makeRotationMatrix(camera *c);
__device__ void normalize(ray *r);
__device__ void normalize(float *v1);
__device__ void cameraToWorld(camera *c, ray *r);

__device__ float dotProduct(const float *v1, const float *v2, int size);
__device__ float distance(const float *p1, const float *p2);
__device__ bool intersect(const ray r, const sphere s, float *pointHit, float *normalHit, float *minDistance);
__device__ color lighting(const light *lights, const int *lightsToUse, const sphere s, const float *pointHit, const float *normalHit, int num_lights, const camera *c);
__device__ color raytrace(const camera *c, const ray r, const sphere *spheres, const light *lights, int sphere_size, int light_size, int depth, int max_depth);

void cudaCallRaytraceKernel(const unsigned int block_size, const unsigned int threadsPerBlock, const camera *c, color *pixels, const sphere *spheres, const light *lights,
  int xres, int yres, int sphere_size, int lights_size, int depth, int max_depth);
