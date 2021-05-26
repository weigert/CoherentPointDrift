#include <TinyEngine/TinyEngine>
#include <TinyEngine/image>
#include <TinyEngine/color>
#include <TinyEngine/camera>

#include "include/shape.h"
#include "../../../cpd.h"

int main( int argc, char* args[] ) {

	Tiny::window("Particle System", 800, 800);
	cam::near = -10.0f;
	cam::far = 10.0f;
	cam::zoomrate = 1.0f;
	cam::init(200);

	Tiny::view.interface = [&](){ /* ... */ }; //No Interface

	Square3D model;									//Model we want to instance render!

	//Generate Point Clouds
	std::vector<glm::vec3> pointsA = shape::cube(1000, glm::vec3(0.25, 0.5, 1.0f));
	std::vector<glm::vec3> pointsB = shape::cube(1000, glm::vec3(0.25, 0.5, 1.0f));

	glm::mat4 rtrans = shape::randomtransform();

	//Apply Transformation and Noise
	for(auto& point: pointsB)
		point = glm::vec3(rtrans*glm::vec4(point, 1.0f));
	shape::noise(pointsA, 2.0f); //Add Gaussian Noise
	shape::noise(pointsB, 2.0f); //Add Gaussian Noise

	//Compute the Theoretical Rigid Transformation
	Eigen::MatrixXf X = cpd::makeset(pointsA);
	Eigen::MatrixXf Y = cpd::makeset(pointsB);

	std::cout<<"Initializing CPD"<<std::endl;
	cpd::initialize(X, Y);

	int niter = 500;

	bool paused = true;

	Tiny::event.handler = [&](){

		if(!Tiny::event.press.empty()){

		  if(Tiny::event.press.back() == SDLK_SPACE){

				std::cout<<"Resetting and Starting CPD"<<std::endl;
				niter = 500;
				cpd::oldvar = 0.0f;

				glm::mat4 transform = cpd::rigid();

				for(auto& point: pointsB)
					point = glm::vec3(transform*glm::vec4(point, 1.0f));

				Y = cpd::makeset(pointsB);
				cpd::initialize(X, Y);

			}

			if(Tiny::event.press.back() == SDLK_p){
				paused = !paused;
			}

		}

		cam::handler();

	};

	//Construct the Models for the PointCloud!
	std::vector<glm::mat4> models;
	std::vector<glm::vec3> colors;

	for(auto& point: pointsA){
		glm::mat4 model = glm::mat4(1.0);
		model = glm::scale(glm::mat4(1.0), glm::vec3(0.5))*model;
		model = glm::rotate(glm::mat4(1.0), glm::radians(90.0f-cam::rot), glm::vec3(0,1,0))*model;
		model = glm::translate(glm::mat4(1.0), point)*model;
		models.push_back(model);
		colors.push_back(glm::vec3(0,0,1));
	}

	for(auto& point: pointsB){
		glm::mat4 model = glm::mat4(1.0);
		model = glm::scale(glm::mat4(1.0), glm::vec3(0.5))*model;
		model = glm::rotate(glm::mat4(1.0), glm::radians(90.0f-cam::rot), glm::vec3(0,1,0))*model;
		model = glm::translate(glm::mat4(1.0), point)*model;
		models.push_back(model);
		colors.push_back(glm::vec3(1,0,0));
	}

	Instance particle(&model);			//Particle system based on this model
	particle.addBuffer(colors);			//Update particle system
	particle.addBuffer(models);			//Update particle system

	Shader particleShader({"source/shader/particle.vs", "source/shader/particle.fs"}, {"in_Quad", "in_Tex", "in_Color", "in_Model"});

	Tiny::view.pipeline = [&](){

		Tiny::view.target(color::black);

		particleShader.use();
		particleShader.uniform("vp", cam::vp);
		particle.render();

	};

	Tiny::loop([&](){ //Autorotate Camera

		if(paused) return;

		cam::pan(0.1f);

		cpd::itersolve(X, Y, niter, 0.0001f);
		glm::mat4 transform = cpd::rigid();

		models.clear();

		for(auto& point: pointsA){
			glm::mat4 model = glm::mat4(1.0);
			model = glm::scale(glm::mat4(1.0), glm::vec3(0.5))*model;
			model = glm::rotate(glm::mat4(1.0), glm::radians(90.0f-cam::rot), glm::vec3(0,1,0))*model;
			model = glm::translate(glm::mat4(1.0), point)*model;
			models.push_back(model);
		}

		for(auto& point: pointsB){
			glm::mat4 model = glm::mat4(1.0);
			model = glm::scale(glm::mat4(1.0), glm::vec3(0.5))*model;
			model = glm::rotate(glm::mat4(1.0), glm::radians(90.0f-cam::rot), glm::vec3(0,1,0))*model;
			model = glm::translate(glm::mat4(1.0), glm::vec3(transform*glm::vec4(point, 1.0f)))*model;
			models.push_back(model);
		}

		particle.updateBuffer(models, 1);

	});

	Tiny::quit();

	return 0;
}
