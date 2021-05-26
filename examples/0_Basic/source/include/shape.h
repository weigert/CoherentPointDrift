/*
Function to Generate a Random Point Cloud with Noise!

Let us see what we can do... some kind of shape?

Then generate a random pose matrix? Basically by sampling some random rotation matrix...

Then add a random translation vector sampled.

We should sample an asymmetric shape.
*/

#include "distribution.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <random>

namespace shape {
using namespace std;
using namespace glm;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 generator (seed);
std::uniform_real_distribution<double> uniform01(-1.0, 1.0);

vector<vec3> sphere(const size_t N){
  vector<vec3> pointset;
  for(size_t i = 0; i < N; i++)
    pointset.push_back(50.0f*normalize(vec3(uniform01(generator), uniform01(generator), uniform01(generator))));
  return pointset;
}

vector<vec3> cube(const size_t N, const vec3 scale){
  vector<vec3> pointset;
  for(size_t i = 0; i < N; i++){
    if(i%6 == 0) pointset.push_back(50.0f*scale*vec3(1.0f, uniform01(generator), uniform01(generator)));
    if(i%6 == 1) pointset.push_back(50.0f*scale*vec3(-1.0f, uniform01(generator), uniform01(generator)));
    if(i%6 == 2) pointset.push_back(50.0f*scale*vec3(uniform01(generator), 1.0f, uniform01(generator)));
    if(i%6 == 3) pointset.push_back(50.0f*scale*vec3(uniform01(generator), -1.0f, uniform01(generator)));
    if(i%6 == 4) pointset.push_back(50.0f*scale*vec3(uniform01(generator), uniform01(generator), 1.0f));
    if(i%6 == 5) pointset.push_back(50.0f*scale*vec3(uniform01(generator), uniform01(generator), -1.0f));
  }
  return pointset;
}

void noise(vector<vec3>& pointset, float std){
  for(auto& point: pointset)
    point = dist::normal(point, glm::vec3(std));
}

//Random 3D Rotation Matrix!
glm::mat4 randomtransform(){

  quat q;
  q.x = dist::normal(0, 1);
  q.y = dist::normal(0, 1);
  q.z = dist::normal(0, 1);
  q.w = dist::normal(0, 1);

  glm::mat4 transform = toMat4(normalize(q));

  //Add Random Translation
  transform[3][0] = 50.0f*uniform01(generator);
  transform[3][1] = 50.0f*uniform01(generator);
  transform[3][2] = 50.0f*uniform01(generator);

  return transform;
  
}

}
