#include "raytracer.cuh"

__device__ Matrix4d makeTranslationMatrix(camera *c){
  //Create the translation matrix
  Matrix4d m(4, 4);
  m << 1, 0, 0, c->position[0], //row 1
     0, 1, 0, c->position[1], //row 2
     0, 0, 1, c->position[2], //row 3
     0, 0, 0, 1; //row 4

  return m;
}

__device__ Matrix4d makeRotationMatrix(camera *c){
  double mag = sqrt(pow(c->position[0], 2) + pow(c->position[1], 2) + pow(c->position[2], 2));
  double ux = c->position[0] / mag;
  double uy = c->position[1] / mag;
  double uz = c->position[2] / mag;

  //Create the rotation matrix
  Matrix4d m(4, 4);
  m << pow(ux, 2) + (1 - pow(ux, 2)) * cos(c->orientation[3]), ux * uy * (1 - cos(c->orientation[3])) - uz * sin(c->orientation[3]), ux * uz * (1 - cos(c->orientation[3])) + uy * sin(c->orientation[3]), 0, //row 1
     uy * ux * (1 - cos(c->orientation[3])) + uz * sin(c->orientation[3]), pow(uy, 2) + (1 - pow(uy, 2)) * cos(c->orientation[3]), uy * uz * (1 - cos(c->orientation[3])) - ux * sin(c->orientation[3]), 0, //row 2
     ux * uz * (1 - cos(c->orientation[3])) - uy * sin(c->orientation[3]), uy * uz * (1 - cos(c->orientation[3])) + ux * sin(c->orientation[3]), pow(uz, 2) + (1 - pow(uz, 2)) * cos(c->orientation[3]), 0, //row 3
     0, 								0,									  0,									    1; //row 4
  return m;
}


// Normalize a ray direction
__device__ void normalize(ray *r){
  float mag = pow(pow(r->direction[0], 2) + pow(r->direction[1], 2) + pow(r->direction[2], 2), .5);
  r->direction[0] = r->direction[0] / mag;
  r->direction[1] = r->direction[1] / mag;
  r->direction[2] = r->direction[2] / mag;

}

// Normalize a vector
__device__ void normalize(float v[]){
  float mag = pow(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2), .5);
  v[0] = v[0] / mag;
  v[1] = v[1] / mag;
  v[2] = v[2] / mag;
}

// Tarnsform a ray in camera space to world space
__device__ void cameraToWorld(camera *c, ray *r){
  Matrix4d rotation = makeRotationMatrix(c);
  Matrix4d translation = makeTranslationMatrix(c);
  Matrix4d transformation = rotation * translation;

  Vector4d rayOrigin(r->origin[0], r->origin[1], r->origin[2], 1);
  Vector4d rayDirection(r->direction[0], r->direction[1], r->direction[2], 1);

  Vector4d newRayOrigin = transformation * rayOrigin;
  Vector4d newRayDirection = transformation * rayDirection;

  r->origin[0] = newRayOrigin(0);
  r->origin[1] = newRayOrigin(1);
  r->origin[2] = newRayOrigin(2);

  r->direction[0] = newRayDirection(0) - r->origin[0];
  r->direction[1] = newRayDirection(1) - r->origin[1];
  r->direction[2] = newRayDirection(2) - r->origin[2];

  normalize(r);
}

// Calculate the dot product of two points
__device__ float dotProduct(const float v1[], const float v2[], int size){
  float sum = 0;
  for(int i = 0; i < size; i++){
    sum += v1[i] * v2[i];
  }
  return sum;
}

// Calculate how far two points are from one another
__device__ float distance(const float p1[], const float p2[]){
  return pow(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2), 1/2);
}

// Calculate if a ray intersects a sphere, and if so, the pointHit and the normal
__device__ bool intersect(const ray r, const sphere s, float pointHit[], float normalHit[], float *minDistance){
  float t0, t1;

  float L[3] = {s.position[0] - r.origin[0], s.position[1] - r.origin[1], s.position[2] - r.origin[2]};
 float tca = dotProduct(L, r.direction, 3);

  float d2 = dotProduct(L, L, 3) - pow(tca, 2);

  if(d2 > pow(s.radius, 2)){
    return false;
  }
  else{

    float thc = pow(s.radius - d2, .5);

    t0 = tca - thc;
    t1 = tca + thc;
    if(t1 < t0){
      swap(t0, t1);
    }
    if(t0 < 0){
      t0 = t1;
    }

    float hit[3];
    hit[0] = r.origin[0] + t0 * r.direction[0];
    hit[1] = r.origin[1] + t0 * r.direction[1];
    hit[2] = r.origin[2] + t0 * r.direction[2];
    if(distance(r.origin, hit) < *minDistance){
      *minDistance = distance(r.origin, hit);
      pointHit[0] = hit[0];
      pointHit[1] = hit[1];
      pointHit[2] = hit[2];

      normalHit[0] = hit[0] - s.position[0];
      normalHit[1] = hit[1] - s.position[1];
      normalHit[2] = hit[2] - s.position[2];
      normalize(normalHit);
    }
    if(t0 >= 0){
      return true;
    }
    else{
      return false;
    }
  }

}

// Calculates the lighting based on ambient, diffuse, and specular.
__device__ color lighting(const light lights[], const int lightsToUse[], const sphere s, const float pointHit[], const float normalHit[], int num_lights, const camera *c){
	float direction[3] = {c->position[0] - pointHit[0], c->position[1] - pointHit[1], c->position[2] - pointHit[2]};
	normalize(direction);
  bool light = false;
  float diffuse_sum[3] = {0, 0, 0};
  float specular_sum[3] = {0, 0, 0};
	for(int i = 0; i < num_lights; i++){
    // Only use light if it was calculated to reach the object earlier
    if(lightsToUse[i] == 1){
      light = true;
  		float light_direction[3] = {lights[i].position[0] - pointHit[0], lights[i].position[1] - pointHit[1], lights[i].position[2] - pointHit[2]};
  		float distance = sqrt(light_direction[0] * light_direction[0] +
  			light_direction[1] * light_direction[1] +
  			light_direction[2] * light_direction[2]);
  		float attenuation = 1 / (1 + lights[i].k * pow(distance, 2));
      float light_color[3] = {lights[i].c.r * attenuation, lights[i].c.g * attenuation, lights[i].c.b * attenuation};
  		if(dotProduct(normalHit, light_direction, 3) >= 0){
  			diffuse_sum[0] += dotProduct(normalHit, light_direction, 3) * light_color[0];
        diffuse_sum[1] += dotProduct(normalHit, light_direction, 3) * light_color[1];
        diffuse_sum[2] += dotProduct(normalHit, light_direction, 3) * light_color[2];
  		}

      float normalized_direction[3] = {direction[0] + light_direction[0], direction[1] + light_direction[1], direction[2] + light_direction[2]};
  		normalize(normalized_direction);

  		if(dotProduct(normalHit, normalized_direction, 3) >= 0{
        specular_sum[0] += pow(dotProduct(normalHit, normalized_direction, 3), s.m.shine) * light_color[0];
        specular_sum[1] += pow(dotProduct(normalHit, normalized_direction, 3), s.m.shine) * light_color[1];
        specular_sum[2] += pow(dotProduct(normalHit, normalized_direction, 3), s.m.shine) * light_color[2];
      }
    }
	}
    color col;

  if(light){
    col.r = min((float)1, s.m.ambient.r + s.m.diffuse.r * diffuse_sum[0] + s.m.specular.r * specular_sum[0]);
    col.g = min((float)1, s.m.ambient.g + s.m.diffuse.g * diffuse_sum[1] + s.m.specular.g * specular_sum[1]);
    col.b = min((float)1, s.m.ambient.b + s.m.diffuse.b * diffuse_sum[2] + s.m.specular.b * specular_sum[2]);
  }
  else{
    col.r = 0;
    col.g = 0;
    col.b = 0;
  }
  // cerr<<"sphere ambient: "<<s.m.ambient.r<<", "<<s.m.ambient.g<<", "<<s.m.ambient.b<<endl;
  // cerr<<"diffuse sum: "<<diffuse_sum[0]<<","<<diffuse_sum[1]<<", "<<diffuse_sum[2]<<endl;
  // cerr<<"specular sum: "<<specular_sum[0]<<","<<specular_sum[1]<<", "<<specular_sum[2]<<endl;

  // cerr<<col.r<<", "<<col.g<<", "<<col.b<<endl;
	return col;
}

// Computes the ray of reflection given a normal and the original ray
__device__ ray computeReflectionRay(const ray r, const float normalHit[]){
  ray reflection;
  reflection.direction[0] = r.direction[0] - 2 * normalHit[0] * dotProduct(r.direction, normalHit, 3);
  reflection.direction[1] = r.direction[1] - 2 * normalHit[1] * dotProduct(r.direction, normalHit, 3);
  reflection.direction[2] = r.direction[2] - 2 * normalHit[2] * dotProduct(r.direction, normalHit, 3);

  reflection.origin[0] = normalHit[0] + reflection.direction[0] * .01;
  reflection.origin[1] = normalHit[1] + reflection.direction[1] * .01;
  reflection.origin[2] = normalHit[2] + reflection.direction[2] * .01;

  return reflection;
}

// Given a ray and a scene, calculates the color of a pixel
__device__ color raytrace(const camera *c, const ray r, const sphere spheres[], const light lights[], int sphere_size, int lights_size, int depth, int max_depth){
  float pointHit[3];
  float normalHit[3];
  float minDistance = FLT_MAX;
  sphere closestSphere;
  bool intersection = false;
  color col;
  col.r = 0;
  col.g = 0;
  col.b = 0;
  // Iterate through every sphere in the scene to find the clsest intersection
  for(int i = 0; i < sphere_size; i++){
    if(intersect(r, spheres[i], pointHit, normalHit, &minDistance)){
      intersection = true;
      closestSphere = spheres[i];
    };
  }
  // If there's an intersection, we can calculate shadows and reflections
  if(intersection){
    // This keeps track of which lights can reach the point hit
    int lightsToUse[lights_size];
    for(int i = 0; i < lights_size; i++){
      lightsToUse[i] = 1;
    }
    bool shadow = false;

    // If we have not hit the max depth of our reflection, recursively calculate reflection
    if(depth < max_depth){
      ray reflection = computeReflectionRay(r, normalHit);
      color reflect_col;
      reflect_col =  raytrace(c, reflection, spheres, lights, sphere_size, lights_size, depth + 1, max_depth);
      col.r += .3 * reflect_col.r;
      col.g += .3 * reflect_col.g;
      col.b += .3 * reflect_col.b;
    }
    // For every light in the scene, see if it can reach the point hit. If it can't, then don't use that light
    // in calculating the lighting for that point
    for(int i = 0; i < lights_size; i++){
      ray shadowRay;
      shadowRay.direction[0] = lights[i].position[0] - pointHit[0];
      shadowRay.direction[1] = lights[i].position[1] - pointHit[1];
      shadowRay.direction[2] = lights[i].position[2] - pointHit[2];
      normalize(&shadowRay);

      // Slightly move the shadow to the side of the point to prevent false self-intersection
      shadowRay.origin[0] = pointHit[0] + .01 * shadowRay.direction[0];
      shadowRay.origin[1] = pointHit[1] + .01 * shadowRay.direction[1];
      shadowRay.origin[2] = pointHit[2] + .01 * shadowRay.direction[2];

      float shadowHit[3];
      float normalShadowHit[3];
      for(int j = 0; j < sphere_size; j++){
        if(intersect(shadowRay, spheres[j], shadowHit, normalShadowHit, &minDistance)){
          shadow = true;
          lightsToUse[i] = 0;
          break;
        }
      }
    }
    color new_col = lighting(lights, lightsToUse, closestSphere, pointHit, normalHit, lights_size, c);
    col.r += new_col.r;
    col.g += new_col.g;
    col.b += new_col.b;
    //Prevent color overflow
    if(col.r > 1){
      col.r = 1;
    }
    if(col.g > 1){
      col.g = 1;
    }
    if(col.b > 1){
      col.b = 1;
    }
  }
  return col;
}

// Creates the rays per pixel, and then sends them to the raytrace function to find the color of the pixel
__global__
void cudaRaytraceKernel(const camera *c, color *pixels, const sphere *spheres, const light *lights,
  int xres, int yres, int sphere_size, int lights_size, int depth, int max_depth){
    extern __shared__ sphere dev_spheres[];
    extern __shared__ light lights[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    while(idx < xres * yres){
      float aspectRatio = c->right / c->top;
      int i = idx % xres;
      int j = idx / xres;
      float pixel_x = (2 * ((i + .5) / xres) - 1) * c->top / c->near * aspectRatio;
      float pixel_y = (1 - 2 * ((j + .5) / yres)) * c->top / c->near;
      ray r;
      r.origin[0] = 0;
      r.origin[1] = 0;
      r.origin[2] = 0;

      r.direction[0] = pixel_x;
      r.direction[1] = pixel_y;
      r.direction[2] = -1;
      normalize(&r);
      cameraToWorld(c, &r);
      pixels[idx] = raytrace(c, r, spheres, lights, num_spheres, num_lights, 0, 3);
    }
}

// Calls the raytracing kernel
void cudaCallRaytraceKernel(const unsigned int blocks, const unsigned int threadsPerBlock, const camera *c, color *pixels, const sphere *spheres, const light *lights,
  int xres, int yres, int sphere_size, int lights_size, int depth, int max_depth){
    cudaRaytraceKernel<<<blocks, threadsPerBlock, sizeof(sphere) * num_spheres, sizeof(light) * lights_size>>>(c, pixels, spheres, lights, xres, yres, sphere_size, lights_size, depth, max_depth);
}
