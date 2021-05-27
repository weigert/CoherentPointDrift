/*

Coherent Point Drift Algorithm!
Author: Nicholas McDonald

Utilizes nanoflann for kdtree lookup

*/

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <glm/glm.hpp>
#include <functional>
#include <iostream>

#include "nanoflann.hpp"

#define PI 3.14159265f

namespace cpd {

using namespace Eigen;
using namespace std;
using namespace glm;

//All the properties needed for the algorithm below!

float D = 0.0f;   //Dimensionality of Pointsets
float N = 0.0f;   //Size of Sampled Pointset
float M = 0.0f;   //Size of Centroid Pointset

float s;          //Scale
Matrix3d R;       //Rotation
Vector3d t;       //Translation
float var;        //Variance

float w = 0.0f;  //Noise Estimate (Free Parameter)
Matrix<double,-1,-1,RowMajor> P;       //Posterior Probability Matrix (Assignments)
VectorXd PX;      //Sum over X
VectorXd PY;      //Sum over Y

//Options
bool enforcescale = true; //Force Scale to be s = 1

double scenescale = 0.0;

/*
================================================================================
              Initialization, Estimation and Maximization Steps
================================================================================
*/

JacobiSVD<Matrix<double,-1,-1,RowMajor>> svd;
nanoflann::KDTreeEigenMatrixAdaptor<Matrix<double,-1,-1,RowMajor>>* kdtree;

void initialize(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y){

  s = 1.0f;                     //Initialize Scale to 1
  R = Matrix3d::Identity();     //Initialize Rotation Matrix to Identity
  t = Vector3d::Zero();         //Initialize Translation Vector to Zero

  N = X.rows();                 //Get the Number of Points in "Sampled Set"
  M = Y.rows();                 //Get the Number of Points in "Centroid Set"

  var = 0.0f;                   //Compute the Worst-Case Variance
  for(size_t n = 0; n < N; n++)
  for(size_t m = 0; m < M; m++){
    Vector3d d = (X.block<1,3>(n,0)-Y.block<1,3>(m,0)); //Distance Between Two Points x_n, y_m
    float dd = d.dot(d);
    if(dd > scenescale) scenescale = dd;
    var += dd/(float)(D*N*M);                     //Normalize and Add
  }

  P = Matrix<double,-1,-1,RowMajor>::Zero(M,N);      //Allocate Memory for Probability Matrix

  //Sort into Kd-Tree
  std::cout<<"Building KDTree"<<std::endl;
  kdtree = new nanoflann::KDTreeEigenMatrixAdaptor<Matrix<double,-1,-1,RowMajor>>(3, std::cref(X), 10);
  kdtree->index->buildIndex();
  std::cout<<"Done"<<std::endl;

}

void pose(mat4 guess){  //Set an Initial Pose!

  R(0,0) = guess[0][0];
  R(0,1) = guess[1][0];
  R(0,2) = guess[2][0];
  R(1,0) = guess[0][1];
  R(1,1) = guess[1][1];
  R(1,2) = guess[2][1];
  R(2,0) = guess[0][2];
  R(2,1) = guess[1][2];
  R(2,2) = guess[2][2];
  t(0) = guess[3][0];
  t(1) = guess[3][1];
  t(2) = guess[3][2];

}

/*
void estimate(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y){

  //Individual Assignment Probability for x_n, y_m!
  const function<float(Vector3d, Vector3d)> Pmn = [](Vector3d x, Vector3d y){
    Vector3d d = (x-s*R*y-t);       //Rigid Transform
    return exp(-0.5f/var*d.dot(d)); //Gaussian Probability
  };

  //Constant Normalization Bias
  const float bias = pow(2.0f*PI*var, D/2.0f)*w/(1.0f-w)*M/N;

  PX = VectorXd::Zero(M); //Zero-Out Cumulative Probabilities
  PY = VectorXd::Zero(N);

  for(size_t m = 0; m < M; m++){

    float Z = bias;

    for(size_t n = 0; n < N; n++){
      P(m,n) = Pmn(X.block<1,3>(n,0), Y.block<1,3>(m,0)); //Compute Probability
      Z += P(m,n);                                        //Accumulate Partition Function
    }

    for(size_t n = 0; n < N; n++){
      P(m,n) /= Z;                                        //Normalize Density
      PX(m) += P(m,n);                                    //Accumulate Probability along N
      PY(n) += P(m,n);                                    //Accumulate Probability along M
    }

  }

}
*/

void estimate(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y){

  //Individual Assignment Probability for x_n, y_m!
  const function<float(Vector3d, Vector3d)> Pmn = [](Vector3d x, Vector3d y){
    Vector3d d = (x-y);             //Rigid Transform ALREADY INCLUDED!
    return exp(-0.5f/var*d.dot(d)); //Gaussian Probability
  };

  //Constant Normalization Bias
  const float bias = pow(2.0f*PI*var, D/2.0f)*w/(1.0f-w)*M/N;

  //Zero the Probability Matrix!
  P = Matrix<double,-1,-1,RowMajor>::Zero(M,N);
  PX = VectorXd::Zero(M); //Zero-Out Cumulative Probabilities
  PY = VectorXd::Zero(N);

  //Approximate Z differently!
  vector<pair<long int,double> > matches;
  nanoflann::SearchParams params;

  for(size_t m = 0; m < M; m++){

    float Z = bias;

    Vector3d YV = Y.block<1,3>(m,0);
    YV = s*R*YV+t;  //Rigid Transform Here!

    const size_t nmatches = kdtree->index->radiusSearch(&YV(0), sqrt(scenescale*var), matches, params);

    for(auto& match: matches){
      P(m,match.first) = Pmn(X.block<1,3>(match.first,0), YV);
      Z += P(m,match.first);
    }

    for(auto& match: matches){
      P(m,match.first) /= Z;
      PX(m) += P(m,match.first);                    //Accumulate Probability along N
      PY(match.first) += P(m,match.first);          //Accumulate Probability along M
    }

  }

}

void maximize(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y){

  float Np = 1.0f/P.sum();                                //Normalization Constant
  Vector3d uX = X.transpose()*PX*Np;                      //Average Position, X-Set
  Vector3d uY = Y.transpose()*PY*Np;                      //Average Position, Y-Set

  Matrix<double,-1,-1,RowMajor> XC = X.rowwise() - uX.transpose();             //Centered X-Set
  Matrix<double,-1,-1,RowMajor> YC = Y.rowwise() - uY.transpose();             //Centered Y-Set

  Matrix<double,-1,-1,RowMajor> A = XC.transpose()*P.transpose()*YC;           //Singular Value Decomp. Matrix

  svd.compute(A, ComputeFullU|ComputeFullV);              //Compute the SVD of A
  Matrix<double,-1,-1,RowMajor> U = svd.matrixU();                             //Get the SVD Matrix U
  Matrix<double,-1,-1,RowMajor> V = svd.matrixV();                             //Get the SVD Matrix V

  Matrix<double,-1,-1,RowMajor> C = Matrix<double,-1,-1,RowMajor>::Identity(D, D);                  //Construct the SVD-Derived Matrix C
  C(D-1, D-1) = (U*V.transpose()).determinant();

  //Compute the Rigid Transformation Parameters
  R = U*C*V.transpose();
  s = (A.transpose()*R).trace()/(YC.transpose()*PY.asDiagonal()*YC).trace();

  //Recompute Standard Deviation
  var = Np/D*((XC.transpose()*PX.asDiagonal()*XC).trace() - s*(A.transpose()*R).trace());

  if(enforcescale) s = 1.0f;  //Enforce Constant Scale
  t = uX - s*R*uY;

}

/*
================================================================================
                          Iteration and Utilization
================================================================================
*/

void output(){
  cout<<"Rigid Transform (Var = "<<var<<"): "<<endl;
  cout<<"R "<<R<<endl;
  cout<<"t "<<t<<endl;
  cout<<"s "<<s<<endl;
}


//Single Iteration Step
float oldvar = 0.0f;
bool itersolve(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y, int& N, const float tol = 0.01f){

  if(N-- <= 0 || abs(oldvar - var) <= tol) return false;

  oldvar = var;
//  cout<<"Iteration: "<<N<<endl;
//  cout<<"Estimating..."<<endl;
  timer::benchmark<std::chrono::milliseconds>([&](){

    cpd::estimate(X, Y);

  });

//  cout<<"Maximizing..."<<endl;

//  timer::benchmark<std::chrono::milliseconds>([&](){

  cpd::maximize(X, Y);

//  });

//  cpd::output();
  return true;

}

void solve(Matrix<double,-1,-1,RowMajor>& X, Matrix<double,-1,-1,RowMajor>& Y, int& maxN, const float tol = 0.01f){
  while(itersolve(X, Y, maxN, tol));
}

mat4 rigid(){                       //Extract a glm-based 4x4 transformation matrix from Y->X
  mat4 transform = {s*R(0,0), s*R(0,1), s*R(0,2), t(0),
                    s*R(1,0), s*R(1,1), s*R(1,2), t(1),
                    s*R(2,0), s*R(2,1), s*R(2,2), t(2),
                    0,        0,        0,        1.0f};
  return transpose(transform); //Tranpose because of Column Major Ordering
}

/*
================================================================================
                Pointset Representation Translation (Helpers)
================================================================================
*/

Matrix<double,-1,-1,RowMajor> makeset(vector<vec2>& pointset){
  Matrix<double,-1,-1,RowMajor> S(pointset.size(), 2);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
  }
  D = 2;
  return S;
}

Matrix<double,-1,-1,RowMajor> makeset(vector<vec3>& pointset){
  Matrix<double,-1,-1,RowMajor> S(pointset.size(), 3);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
    S(i, 2) = pointset[i].z;
  }
  D = 3;
  return S;
}

}
