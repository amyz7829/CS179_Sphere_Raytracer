#include "raytracer.h"

int main(){
  sphere s;
  s.position[0] = 0;
  s.position[1] = 0;
  s.position[2] = 0;

  s.radius = 1;

  ray r;
  r.origin[0] = 0;
  r.origin[1] = 0;
  r.origin[2] = -2;

  r.direction[0] = 0;
  r.direction[1] = 0;
  r.direction[2] = 3;

  normalize(&r);
  cout<<"direction"<<r.direction[0]<<", "<<r.direction[1]<<", "<<r.direction[2]<<endl;

  float v1[3] = {0, 0, 3};
  normalize(v1);
  cout<<"v1: "<<v1[0]<<", "<<v1[1]<<", "<<v1[2]<<endl;
  light l;
  l.position[0] = 0;
  l.position[1] = 2;
  l.position[2] = 0;

  float pointHit[3];
  float normalHit[3];

  sphere closestSphere;
  float minDistance = FLT_MAX;
  intersect(r, s, pointHit, normalHit, &closestSphere, &minDistance);
  cout<<"pointHit:("<<pointHit[0]<<", "<<pointHit[1]<<", "<<pointHit[2]<<")"<<endl;
  cout<<"normalHit:("<<normalHit[0]<<", "<<normalHit[1]<<", "<<normalHit[2]<<")"<<endl;
  cout<<"minDistance:"<<minDistance<<endl;

  ray r2;
  r2.origin[0] = 0;
  r2.origin[1] = 0;
  r2.origin[2] = 0;

  r2.direction[0] = 0;
  r2.direction[1] = 0;
  r2.direction[2] = -2;

  minDistance = FLT_MAX;
  if(intersect(r2, s, pointHit, normalHit, &closestSphere, &minDistance)){
  cout<<"pointHit:("<<pointHit[0]<<", "<<pointHit[1]<<", "<<pointHit[2]<<")"<<endl;
  cout<<"normalHit:("<<normalHit[0]<<", "<<normalHit[1]<<", "<<normalHit[2]<<")"<<endl;
  cout<<"minDistance:"<<minDistance<<endl;
}
}
