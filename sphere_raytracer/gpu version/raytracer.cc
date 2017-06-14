#include "raytracer.h"

int main(int argc, char* argv[]){
  TA_Utilities::select_least_utilized_GPU();
  int max_time_allowed_in_seconds = 40;
  TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

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

      color pixels[xres * yres];
      for(int i = 0; i < xres * yres){
          pixels[i].r = 0;
          pixels[i].g = 0;
          pixels[i].b = 0;
      }

      color *dev_pixels;
      sphere *dev_spheres;
      light *dev_lights;
      camera *dev_camera;

      cudaMalloc((void**)&dev_pixels, sizeof(color) * xres * yres);
      cudaMalloc((void **)&dev_spheres, sizeof(sphere) * num_spheres);
      cudaMalloc((void **)&dev_lights, sizeof(light) * num_lights);
      cudaMalloc((void **)&dev_camera, sizeof(camera));

      cudaMemcpy(dev_pixels, pixels, sizeof(color) * xres * yres, cudaMemcpyHostToDevice);
      cudaMemcpy(dev_spheres, spheres, sizeof(sphere) * num_spheres, cudaMemcpyHostToDevice);
      cudaMemcpy(dev_lights, lights, sizeof(light) * num_lights, cudaMemcpyHostToDevice);
      cudaMemcpy(dev_camera, c, sizeof(camera));

      cudaCallRaytraceKernel(1024, 1024, dev_pixels, dev_spheres, dev_lights, dev_camera, xres, yres, num_spheres, num_lights, 0, 3);

      cudaMemcpy(pixels, dev_pixels, sizeof(color) * xres * yres, cudaMemcpyDeviceToHost);

      cudaFree(dev_pixels);
      cudaFree(dev_spheres);
      cudaFree(dev_lights);
      cudaFree(dev_camera);

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
