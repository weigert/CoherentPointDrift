# Coherent Point Drift

Single-Header Reference Implementation of Rigid Coherent Point Drift in C++ / Eigen3  ( < 80 LOC )

Example programs include a very basic useage and a version with visualization. The algorithm is performant but not optimized with fast gaussian transform (yet).

The algorithm is robust against noise and occlusion.

Reference: Point Set Registration: Coherent Point Drift, Andriy Myronenko and Xubo Song (2010)

![Coherent Point Drift Example on Cuboid with 3 Different Side Lengths, Constant Scale](https://github.com/weigert/CoherentPointDrift/blob/master/screenshots/cpd.gif)

Coherent Point Drift Example on Cuboid with 3 Different Side Lengths, Constant Scale. Example Program 1

## Algorithm

This is an expectation maximization algorithm which estimates the rigid transformation parameters of between two point clouds. It assumes that one point cloud is a gaussian mixture model from which the other is generated. We can compute a likelihood of points being generated from this mixture model (assignments) and use this to find optimal transform parameters using expectation maximization of the posterior distribution (i.e. parameters given evidence of assignments).

A rigid transformation includes rotation, translation and scaling.

## Usage

The actual CPD portion is separated out cleanly in a header so it can be reutilized without the visualization.

The relevant portion of the code is only about 80 lines of code, but makes use of existing linear algebra routines provided by Eigen.

### Building / Running Examples

Enter the relevant example folders and use the make files. Binaries are built in place.

    cd examples/0_Basic
    make all
    ./main

### Dependencies

`cpd.h`

    - Eigen3
    - glm

`/examples/1_Visual`

    - TinyEngine

### Visual Program

    P: Pause / Unpause
    Space: Apply Rigid Transformation, Restart Iteration
