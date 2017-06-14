#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "../Eigen/Dense"
#include <fstream>
#include <float.h>
#include <algorithm>
#include <math.h>
#include <sstream>

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

void parse_camera_data(ifstream *parser, camera *c);
void parse_sphere_data(ifstream *parser, sphere *s);
void parse_light_data(ifstream *parser, light *l);

Matrix4d makeTranslationMatrix(camera *c);
Matrix4d makeRotationMatrix(camera *c);
void normalize(ray *r);
void normalize(float v1[]);
void cameraToWorld(camera *c, ray *r);

float dotProduct(const float v1[], const float v2[], int size);
float distance(const float p1[], const float p2[]);
bool intersect(const ray r, const sphere s, float pointHit[], float normalHit[], float *minDistance);
color lighting(const light lights[], const int lightsToUse[], const sphere s, const float pointHit[], const float normalHit[], int num_lights, const camera *c);
color raytrace(const camera *c, const ray r, const sphere spheres[], const light lights[], int sphere_size, int light_size, int depth, int max_depth);
