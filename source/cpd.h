/*

Coherent Point Drift Algorithm!

*/

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <functional>

#define PI 3.14159265f

namespace cpd {
using namespace Eigen;
using namespace std;
using namespace glm;

/*
================================================================================
                    Pointset Representation Translation
================================================================================
*/

float D = 0.0f;  //Dimensionality of Pointsets
float N = 0.0f;  //Size of Sampled Pointset
float M = 0.0f;  //Size of Centroid Pointset

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

/*
================================================================================
                      Rigid Transformation Formulation
================================================================================
*/

float s;        //Scale
Matrix3f R;     //Rotation
Vector3f t;     //Translation
float var;      //Variance

float w = 0.0f; //Noise Estimate (Free Parameter)
MatrixXf P;     //Posterior Probability Matrix (Assignments)

void initialize(MatrixXf& X, MatrixXf& Y){

  //Rigid Transformation Parameters
  s = 1.0f;
  R = Matrix3f::Identity();
  t = Vector3f::Zero();

  //Get Size Parameters
  N = X.rows();
  M = Y.rows();

  //Variance (Worst-Case)
  var = 0.0f;
  for(size_t n = 0; n < N; n++)
    for(size_t m = 0; m < M; m++){
      Vector3f d = (X.block<1,3>(n,0)-Y.block<1,3>(m,0));
      var += d.dot(d)/(float)(D*N*M);
    }

  //Allocate Memory for the Probability Matrix
  P = MatrixXf::Zero(M,N);

}

void estimate(MatrixXf& X, MatrixXf& Y){

  std::cout<<"Estimating..."<<std::endl;

  //Individual Assignment Probability!
  const function<float(Vector3f, Vector3f)> Pmn = [](Vector3f x, Vector3f y){
    Vector3f d = (x-s*R*y-t);       //Rigid Transform
    return exp(-0.5f/var*d.dot(d)); //Gaussian Probability
  };

  //Constant Normalization Bias
  const float bias = pow(2.0f*PI*var, D/2.0f)*w/(1.0f-w)*M/N;

  //Compute the Probability Matrix!
  for(size_t n = 0; n < N; n++)
    for(size_t m = 0; m < M; m++)
      P(m,n) = Pmn(X.block<1,3>(n,0), Y.block<1,3>(m,0));

  //Normalize
  for(size_t n = 0; n < N; n++){

    //Accumulate
    float Z = bias;
    for(size_t m = 0; m < M; m++)
      Z += P(m,n);

    //Divide for all m belonging to n
    for(size_t m = 0; m < M; m++)
      P(m,n) /= Z;

  }

}

void maximize(MatrixXf& X, MatrixXf& Y){

  std::cout<<"Maximizing..."<<std::endl;

  //Compute Helper Quantities
  float Np = P.sum(); //Normalization Constant
  VectorXf P1 = P*MatrixXf::Ones(N,1);
  VectorXf PT1 = P.transpose()*MatrixXf::Ones(M,1);
  Vector3f uX = X.transpose()*PT1/Np; //Average Position X
  Vector3f uY = Y.transpose()*P1/Np; //Average Position X

  //Centralize X, Y
  MatrixXf XC = X - MatrixXf::Ones(N,1)*uX.transpose();
  MatrixXf YC = Y - MatrixXf::Ones(M,1)*uY.transpose();

  //Singular Value Decomposition
  MatrixXf A = XC.transpose()*P.transpose()*YC;

  JacobiSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
  MatrixXf U = svd.matrixU();
  MatrixXf V = svd.matrixV();

  MatrixXf C = MatrixXf::Zero(U.cols(), V.rows());
  for(int i = 0; i < C.cols() && i < C.rows(); i++)
    C(i, i) = 1;

  if(C.cols() < C.cols()) C(C.cols()-1, C.cols()-1) = (U*V.transpose()).determinant();
  else C(C.rows()-1, C.rows()-1) = (U*V.transpose()).determinant();

  //Compute the Derived Quantities
  R = U*C*V.transpose();
  s = (A.transpose()*R).trace()/(YC.transpose()*P1.asDiagonal()*YC).trace();
  t = uX - s*R*uY;

  //Recompute the Standard Deviation
  var = 1.0f/(Np*D)*((XC.transpose()*PT1.asDiagonal()*XC).trace() - s*(A.transpose()*R).trace());

  //Reset Scale?
  s = 1.0f;

}

void output(){

  std::cout<<"Rigid Transform (Err = "<<var<<"): "<<std::endl;
  std::cout<<"R "<<R<<std::endl;
  std::cout<<"t "<<t<<std::endl;
  std::cout<<"s "<<s<<std::endl;

}

float oldvar = 0.0f;
void solve(MatrixXf& X, MatrixXf& Y, const int maxN, const float tol = 0.01f){

  for(int i = 0; i < maxN && abs(oldvar - var) > tol; i++){
    oldvar = var;
    std::cout<<"Iteration: "<<i<<std::endl;
    cpd::estimate(X, Y);
    cpd::maximize(X, Y);
    cpd::output();
  }

}

//Single Step!
void itersolve(MatrixXf& X, MatrixXf& Y, int& N, const float tol = 0.01f){

  if(N-- <= 0 || abs(oldvar - var) <= tol) return;

  oldvar = var;
  std::cout<<"Iteration: "<<N<<std::endl;
  cpd::estimate(X, Y);
  cpd::maximize(X, Y);
  cpd::output();

}

mat4 rigid(){

  mat4 transform = mat4(1);
  transform[0][0] = s*R(0,0);
  transform[0][1] = s*R(0,1);
  transform[0][2] = s*R(0,2);
  transform[1][0] = s*R(1,0);
  transform[1][1] = s*R(1,1);
  transform[1][2] = s*R(1,2);
  transform[2][0] = s*R(2,0);
  transform[2][1] = s*R(2,1);
  transform[2][2] = s*R(2,2);
  transform[0][3] = t(0);
  transform[1][3] = t(1);
  transform[2][3] = t(2);
  return transform;

}

}
