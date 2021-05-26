#include "../../../cpd.h"
#include "include/shape.h"
#include <glm/gtx/string_cast.hpp>

int main( int argc, char* args[] ) {

	const int Npoints = 1000;		//Number of Points per Set
	const float var = 2.0f;			//Noise Variance
	const int maxiter = 500;
	const float tol = 0.01f;

	std::cout<<"Generating Pointset X, Y..."<<std::endl;
	std::vector<glm::vec3> pointsA = shape::cube(Npoints, glm::vec3(0.25, 0.5, 1.0f));
	std::vector<glm::vec3> pointsB = shape::cube(Npoints, glm::vec3(0.25, 0.5, 1.0f));

	std::cout<<"Generating Random Transformation Matrix..."<<std::endl;
	glm::mat4 rtrans = shape::randomtransform();

	std::cout<<"Applying Transformation to Pointset Y..."<<std::endl;
	for(auto& point: pointsB)
		point = glm::vec3(rtrans*glm::vec4(point, 1.0f));

	std::cout<<"Adding Gaussian Noise, Isotropic Variance = "<<var<<std::endl;
	shape::noise(pointsA, var); //Add Gaussian Noise
	shape::noise(pointsB, var); //Add Gaussian Noise

	std::cout<<"Constructing Pointset Matrices..."<<std::endl;
	Eigen::MatrixXf X = cpd::makeset(pointsA);
	Eigen::MatrixXf Y = cpd::makeset(pointsB);

	std::cout<<"Initializing CPD..."<<std::endl;
	cpd::initialize(X, Y);

	std::cout<<"Solving Transformation, maxiter = "<<maxiter<<", tol = "<<tol<<" ..."<<std::endl;
	cpd::solve(X, Y, maxiter, 0.01f);

	std::cout<<"Done..."<<std::endl<<std::endl;

	std::cout<<"Result:"<<std::endl<<std::endl;;
	std::cout<<"Initial Pose:"<<std::endl;
	std::cout<<glm::to_string(glm::inverse(rtrans))<<std::endl;
	std::cout<<std::endl;

	std::cout<<"Computed Pose:"<<std::endl;
	std::cout<<glm::to_string(cpd::rigid())<<std::endl;

	return 0;
}
