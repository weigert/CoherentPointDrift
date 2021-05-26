/*

Coherent Point Drift Algorithm!
Author: Nicholas McDonald

*/

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <functional>
#include <iostream>

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
Matrix3f R;       //Rotation
Vector3f t;       //Translation
float var;        //Variance

float w = 0.0f;   //Noise Estimate (Free Parameter)
MatrixXf P;       //Posterior Probability Matrix (Assignments)

//Options
bool enforcescale = true; //Force Scale to be s = 1

/*
================================================================================
              Initialization, Estimation and Maximization Steps
================================================================================
*/

void initialize(MatrixXf& X, MatrixXf& Y){

  s = 1.0f;                     //Initialize Scale to 1
  R = Matrix3f::Identity();     //Initialize Rotation Matrix to Identity
  t = Vector3f::Zero();         //Initialize Translation Vector to Zero

  N = X.rows();                 //Get the Number of Points in "Sampled Set"
  M = Y.rows();                 //Get the Number of Points in "Centroid Set"

  var = 0.0f;                   //Compute the Worst-Case Variance
  for(size_t n = 0; n < N; n++)
  for(size_t m = 0; m < M; m++){
    Vector3f d = (X.block<1,3>(n,0)-Y.block<1,3>(m,0)); //Distance Between Two Points x_n, y_m
    var += d.dot(d)/(float)(D*N*M);                     //Normalize and Add
  }

  P = MatrixXf::Zero(M,N);      //Allocate Memory for Probability Matrix

}

void estimate(MatrixXf& X, MatrixXf& Y){

  //Individual Assignment Probability for x_n, y_m!
  const function<float(Vector3f, Vector3f)> Pmn = [](Vector3f x, Vector3f y){
    Vector3f d = (x-s*R*y-t);       //Rigid Transform
    return exp(-0.5f/var*d.dot(d)); //Gaussian Probability
  };

  //Constant Normalization Bias
  const float bias = pow(2.0f*PI*var, D/2.0f)*w/(1.0f-w)*M/N;

  for(size_t n = 0; n < N; n++)     //Full Probability Matrix
  for(size_t m = 0; m < M; m++)
    P(m,n) = Pmn(X.block<1,3>(n,0), Y.block<1,3>(m,0));

  for(size_t n = 0; n < N; n++){    //Normalize
    float Z = bias;                 //Add Bias
    for(size_t m = 0; m < M; m++)   //Accumulate
      Z += P(m,n);
    for(size_t m = 0; m < M; m++)   //Divide
      P(m,n) /= Z;
  }

}

void maximize(MatrixXf& X, MatrixXf& Y){

  float Np = P.sum();                                     //Normalization Constant
  VectorXf P1 = P*MatrixXf::Ones(N,1);                    //Sum over all n
  VectorXf PT1 = P.transpose()*MatrixXf::Ones(M,1);       //Sum over all m
  Vector3f uX = X.transpose()*PT1/Np;                     //Average Position, X-Set
  Vector3f uY = Y.transpose()*P1/Np;                      //Average Position, Y-Set

  MatrixXf XC = X - MatrixXf::Ones(N,1)*uX.transpose();   //Centered X-Set
  MatrixXf YC = Y - MatrixXf::Ones(M,1)*uY.transpose();   //Centered Y-Set

  MatrixXf A = XC.transpose()*P.transpose()*YC;           //Singular Value Decomp. Matrix

  JacobiSVD<MatrixXf> svd(A, ComputeFullU|ComputeFullV);  //Compute the SVD of A
  MatrixXf U = svd.matrixU();                             //Get the SVD Matrix U
  MatrixXf V = svd.matrixV();                             //Get the SVD Matrix V

  MatrixXf C = MatrixXf::Zero(D, D);                      //Construct the SVD-Derived Matrix C
  for(int i = 0; i < D-1; i++)
    C(i, i) = 1;
  C(D-1, D-1) = (U*V.transpose()).determinant();

  //Compute the Rigid Transformation Parameters
  R = U*C*V.transpose();
  s = (A.transpose()*R).trace()/(YC.transpose()*P1.asDiagonal()*YC).trace();
  t = uX - s*R*uY;

  //Recompute Standard Deviation
  var = 1.0f/(Np*D)*((XC.transpose()*PT1.asDiagonal()*XC).trace() - s*(A.transpose()*R).trace());

  if(enforcescale) s = 1.0f;  //Enforce Constant Scale

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
bool itersolve(MatrixXf& X, MatrixXf& Y, int& N, const float tol = 0.01f){

  if(N-- <= 0 || abs(oldvar - var) <= tol) return false;

  oldvar = var;
  cout<<"Iteration: "<<N<<endl;
  cout<<"Estimating..."<<endl;
  cpd::estimate(X, Y);
  cout<<"Maximizing..."<<endl;
  cpd::maximize(X, Y);
  cpd::output();
  return true;

}

void solve(MatrixXf& X, MatrixXf& Y, const int maxN, const float tol = 0.01f){
  int niter = maxN;
  while(itersolve(X, Y, niter, tol));
}


mat4 rigid(){                       //Extract a glm-based 4x4 transformation matrix from Y->X
  mat4 transform = {s*R(0,0), s*R(0,1), s*R(0,2), t(0),
                    s*R(1,0), s*R(1,1), s*R(1,2), t(1),
                    s*R(2,0), s*R(2,1), s*R(2,2), t(2),
                    0,        0,        0,        1.0f};
  return glm::transpose(transform); //Tranpose because of Column Major Ordering
}

/*
================================================================================
                Pointset Representation Translation (Helpers)
================================================================================
*/

MatrixXf makeset(vector<vec2>& pointset){
  MatrixXf S(pointset.size(), 2);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
  }
  D = 2;
  return S;
}

MatrixXf makeset(vector<vec3>& pointset){
  MatrixXf S(pointset.size(), 3);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
    S(i, 2) = pointset[i].z;
  }
  D = 3;
  return S;
}

}
