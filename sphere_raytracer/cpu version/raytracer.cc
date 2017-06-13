#include "raytracer.h"

int main(int argc, char* argv[]){
  if(argc != 4){
    cout<<"Wrong number of arguments, please provide a scene file and xres and yres"<<endl;
  }
  else{
    int xres = atoi(argv[2]);
    int yres = atoi(argv[3]);

    ifstream parser;
    parser.open(argv[1]);

    if(!parser.good()){
			cerr << "Something has gone wrong when opening your file. Please try again." << endl;
			return 1;
		}
    else{
      camera *c = new camera;
      parse_camera_data(&parser, c);
      string buf;

      //should be spheres n line
      getline(parser, buf);
      stringstream ss;
      ss.clear();
      ss.str("");
      ss<<buf;

      string type;
      int num_spheres;
      ss>>type>>num_spheres;
      sphere spheres[num_spheres];
      for(int i = 0; i < num_spheres; i++){
        parse_sphere_data(&parser, &spheres[i]);
      }
      //should be lights n line
      getline(parser, buf);
      int num_lights;
      ss.clear();
      ss.str("");
      ss<<buf;
      ss>>type>>num_lights;

      light lights[num_lights];
      for(int i = 0; i < num_spheres; i++){
        parse_light_data(&parser, &lights[i]);
      }

      color pixels[xres][yres];
      for(int i = 0; i < xres; i++){
        for(int j = 0; j < yres; j++){
          float aspectRatio = c->right / c->top;
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
          // Changes the ray from world coordinates to
          cameraToWorld(c, &r);
          // cerr<<"on pixel: ("<<i<<", "<<j<<")"<<endl;
          // cerr<<"light color: "<<lights[0].c.r<<", "<<lights[0].c.g<<", "<<lights[0].c.b<<endl;
          pixels[i][j] = raytrace(c, r, spheres, lights, num_spheres, num_lights, 0, 3);
        }
      }
      cout<<"P3"<<endl;
      cout<<xres<<" "<<yres<<endl;
      cout<<255<<endl;
      for(int i = 0; i < yres; i++){
        for(int j = 0; j < xres; j++){
          cout<<(int)(pixels[j][i].r * 255)<<" "<<(int)(pixels[j][i].g * 255)<<" "<<(int)(pixels[j][i].b * 255)<<endl;
        }
      }
    }
  }
}
