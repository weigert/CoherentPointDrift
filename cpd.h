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

//For convenience:
//  Rowmajor makes sense because we want points to be ordered consecutively in an NxD Matrix!
typedef Matrix<double,-1,-1,RowMajor> RowMatrix;

//All the properties needed for the algorithm below!

float D = 0.0f;   //Dimensionality of Pointsets
float N = 0.0f;   //Size of Sampled Pointset
float M = 0.0f;   //Size of Centroid Pointset

float s;          //Scale
Matrix3d R;       //Rotation
Vector3d t;       //Translation
float var;        //Variance

float w = 0.0f;   //Noise Estimate (Free Parameter)
RowMatrix P;      //Posterior Probability Matrix (Assignments)
VectorXd PX;      //Sum over X
VectorXd PY;      //Sum over Y

//Options
bool enforcescale = true; //Force Scale to be s = 1

//Parameters Relevant for Acceleration!

//KD-Tree
bool usetree = false;  //Whether to use the tree acceleration

nanoflann::KDTreeEigenMatrixAdaptor<RowMatrix>* kdtree = NULL;
const int nleaf = 25;       //Number of Points in Leaf-Node of Kd-Tree

double scenescale = 0.0;    //Search radius for "near" centroids is scaled by the diagonal of pointset bounding box!
                            //Note: Computed automatically

/*
================================================================================
              Initialization, Estimation and Maximization Steps
================================================================================
*/

JacobiSVD<RowMatrix> svd;

void initialize(RowMatrix& X, RowMatrix& Y){

  s = 1.0f;                     //Initialize Scale to 1
  R = Matrix3d::Identity();     //Initialize Rotation Matrix to Identity
  t = Vector3d::Zero();         //Initialize Translation Vector to Zero

  N = X.rows();                 //Get the Number of Points in "Sampled Set"
  M = Y.rows();                 //Get the Number of Points in "Centroid Set"

  scenescale = 0.0;            //Reset Scenescale
  var = 0.0f;                   //Compute the Worst-Case Variance
  for(size_t n = 0; n < N; n++)
  for(size_t m = 0; m < M; m++){
    Vector3d d = (X.block<1,3>(n,0)-Y.block<1,3>(m,0)); //Distance Between Two Points x_n, y_m
    float dd = d.dot(d);
    if(usetree && dd > scenescale) scenescale = dd;     //scenescale = max. distance squared
    var += dd/(float)(D*N*M);                           //Normalize and Add
  }

  P = RowMatrix::Zero(M,N);      //Allocate Memory for Probability Matrix

  //Sort the static pointset (i.e. samples) into a kd-tree!
  if(usetree){
    if(kdtree != NULL) delete kdtree; //handle "reinitialization"
    kdtree = new nanoflann::KDTreeEigenMatrixAdaptor<RowMatrix>(D, cref(X), nleaf);
    kdtree->index->buildIndex();
  }

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
================================================================================
            Gauss Transform Methods / Probability Computation
================================================================================
*/

//Individual Assignment Probability for x_n, y_m!
float Pmn(Vector3d x, Vector3d& y){
  return exp(-0.5f/var*(x-y).dot(x-y)); //Gaussian Probability
};

float bias = 0.0f;

void direct(RowMatrix& X, RowMatrix& Y){

  for(size_t m = 0; m < M; m++){

    float Z = bias;

    Vector3d YV = Y.block<1,3>(m,0);
    YV = s*R*YV+t;  //Rigid Transform Here!

    for(size_t n = 0; n < N; n++){
      P(m,n) = Pmn(X.block<1,3>(n,0), YV); //Compute Probability
      Z += P(m,n);                                        //Accumulate Partition Function
    }

    for(size_t n = 0; n < N; n++){
      P(m,n) /= Z;                                        //Normalize Density
      PX(n) += P(m,n);                                    //Accumulate Probability along N
      PY(m) += P(m,n);                                    //Accumulate Probability along M
    }

  }

}

void singletree(RowMatrix& X, RowMatrix& Y){

  vector<pair<long int,double> > matches;
  nanoflann::SearchParams params;

  for(size_t m = 0; m < M; m++){

    float Z = bias;

    Vector3d YV = Y.block<1,3>(m,0);
    YV = s*R*YV+t;  //Rigid Transform Here!

    const size_t nmatches = kdtree->index->radiusSearch(&YV(0), pow(scenescale,1.0f/D)*sqrt(var), matches, params);

    for(auto& match: matches){
      P(m,match.first) = Pmn(X.block<1,3>(match.first,0), YV);
      Z += P(m,match.first);
    }

    for(auto& match: matches){
      P(m,match.first) /= Z;
      PX(match.first) += P(m,match.first);          //Accumulate Probability along M
      PY(m) += P(m,match.first);                    //Accumulate Probability along N
    }

  }

}

/*
================================================================================
                  Expectation Maximization Algorithm
================================================================================
*/

void estimate(RowMatrix& X, RowMatrix& Y){

  //Zero-Out Probabilities
  P = RowMatrix::Zero(M,N);
  PX = VectorXd::Zero(N);
  PY = VectorXd::Zero(M);

  //Compute Bias
  bias = pow(2.0f*PI*var, D/2.0f)*w/(1.0f-w)*M/N;

  //Compute P, PX, PY (~ Gauss Transform)
  if(usetree) singletree(X, Y);
  else direct(X, Y);

}

void maximize(RowMatrix& X, RowMatrix& Y){

  float Np = 1.0f/P.sum();                                //Normalization Constant
  Vector3d uX = X.transpose()*PX*Np;                      //Average Position, X-Set
  Vector3d uY = Y.transpose()*PY*Np;                      //Average Position, Y-Set

  RowMatrix XC = X.rowwise() - uX.transpose();            //Centered X-Set
  RowMatrix YC = Y.rowwise() - uY.transpose();            //Centered Y-Set

  RowMatrix A = XC.transpose()*P.transpose()*YC;          //Singular Value Decomp. Matrix

  svd.compute(A, ComputeFullU|ComputeFullV);              //Compute the SVD of A
  RowMatrix U = svd.matrixU();                            //Get the SVD Matrix U
  RowMatrix V = svd.matrixV();                            //Get the SVD Matrix V

  RowMatrix C = RowMatrix::Identity(D, D);                //Construct the SVD-Derived Matrix C
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
bool itersolve(RowMatrix& X, RowMatrix& Y, int& N, const float tol = 0.01f){

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

void solve(RowMatrix& X, RowMatrix& Y, int& maxN, const float tol = 0.01f){
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

RowMatrix makeset(vector<vec2>& pointset){
  RowMatrix S(pointset.size(), 2);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
  }
  D = 2;
  return S;
}

RowMatrix makeset(vector<vec3>& pointset){
  RowMatrix S(pointset.size(), 3);
  for(size_t i = 0; i < pointset.size(); i++){
    S(i, 0) = pointset[i].x;
    S(i, 1) = pointset[i].y;
    S(i, 2) = pointset[i].z;
  }
  D = 3;
  return S;
}

}
