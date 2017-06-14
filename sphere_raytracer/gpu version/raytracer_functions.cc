#include "raytracer.h"

void parse_camera_data(ifstream *parser, camera *c){
  string buf;
  getline(*parser, buf);

	//Create a stringstream
	stringstream ss;
	ss.clear();
	ss.str("");
  ss<<buf;
  if(buf.find("camera") != string::npos){
  	//Until a new line is hit, keep parsing
  	while((int)buf[0] != 0){
  		//Always clear the string stream first
  		ss.clear();
  		ss.str("");
  		ss<<buf;
  		string type;
  		if(buf.find("position") != string::npos){
  			//Type = position, x, y, z are the vertex coords
  			string type;

  			ss>>type>>c->position[0]>>c->position[1]>>c->position[2];
  		}
  		if(buf.find("direction") != string::npos){
  			ss>>type>>c->orientation[0]>>c->orientation[1]>>c->orientation[2]>>c->orientation[3];
  		}
  		if(buf.find("near") != string::npos){
  			float near;
  			ss>>type>>near;

  			c->near = near;
  		}
  		if(buf.find("far") != string::npos){
  			float far;
  			ss>>type>>far;

  			c->far = far;
  		}
  		if(buf.find("left") != string::npos){
  			float left;
  			ss>>type>>left;

  			c->left = left;
  		}
  		if(buf.find("right") != string::npos){
  			float right;
  			ss>>type>>right;

  			c->right = right;
  		}
  		if(buf.find("top") != string::npos){
  			float top;
  			ss>>type>>top;

  			c->top = top;
  		}
  		if(buf.find("bottom") != string::npos){
  			float bottom;
  			ss>>type>>bottom;

  			c->bottom = bottom;
  		}

  		//Move forward another line
  		getline(*parser, buf);
  	}
  }
  else{
    cerr<<"error while parsing"<<endl;
  }
}

void parse_sphere_data(ifstream *parser, sphere *s){
  string buf;

  getline(*parser, buf);
	//Create a stringstream
	stringstream ss;
	ss.clear();
	ss.str("");
  ss<<buf;

	//Until a new line is hit, keep parsing
	while((int)buf[0] != 0){
		//Always clear the string stream first
		ss.clear();
		ss.str("");
		ss<<buf;
    // cerr<<"buf: "<<buf<<endl;
		string type;
		if(buf.find("position") != string::npos){
			ss>>type>>s->position[0]>>s->position[1]>>s->position[2];
		}
		if(buf.find("radius") != string::npos){
			ss>>type>>s->radius;
		}
		if(buf.find("ambient") != string::npos){
			ss>>type>>s->m.ambient.r>>s->m.ambient.g>>s->m.ambient.b;
      cerr<<"ambient"<<s->m.ambient.r<<", "<<s->m.ambient.g<<", "<<s->m.ambient.b<<endl;
		}
		if(buf.find("diffuse") != string::npos){
			ss>>type>>s->m.diffuse.r>>s->m.diffuse.g>>s->m.diffuse.b;
      cerr<<"diffuse"<<s->m.diffuse.r<<", "<<s->m.diffuse.g<<", "<<s->m.diffuse.b<<endl;
		}
		if(buf.find("specular") != string::npos){
			ss>>type>>s->m.specular.r>>s->m.specular.g>>s->m.specular.b;
      cerr<<"specular"<<s->m.specular.r<<", "<<s->m.specular.g<<", "<<s->m.specular.b<<endl;
		}
		if(buf.find("shininess") != string::npos){
			ss>>type>>s->m.shine;
		}
    //Move forward another line
		getline(*parser, buf);
	}
}

void parse_light_data(ifstream *parser, light *l){
  string buf;

  getline(*parser, buf);

	//Create a stringstream
	stringstream ss;
	ss.clear();
	ss.str("");
  ss<<buf;

	//Until a new line is hit, keep parsing
	while((int)buf[0] != 0 && !parser.eof()){
		//Always clear the string stream first
		ss.clear();
		ss.str("");
		ss<<buf;
		string type;
    cerr<<"buf: "<<buf<<endl;
		if(buf.find("position") != string::npos){
			ss>>type>>l->position[0]>>l->position[1]>>l->position[2];
		}
		if(buf.find("color") != string::npos){

			ss>>type>>l->c.r>>l->c.g>>l->c.b;
		}
		if(buf.find("k") != string::npos){
			ss>>type>>l->k;
		}
    //Move forward another line
		getline(*parser, buf);
	}
}
