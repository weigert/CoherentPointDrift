# Coherent Point Drift

A simple reference C++ / Eigen3 implementation of the classic coherent point drift algorithm for a rigid transform (scale, translate, rotate) from the publication ... with a simple visualization.

The actual CPD portion is separated out cleanly in a header so it can be reutilized without the visualization.

## Algorithm

This is an expectation maximization algorithm which estimates the rigid transformation parameters of between two point clouds. It assumes that one point cloud is a gaussian mixture model from which the other is generated, and then the expectation is maximized in this framework.

## Usage

## References
