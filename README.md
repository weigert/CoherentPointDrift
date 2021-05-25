# Coherent Point Drift

Single-Header Reference Implementation of Rigid Coherent Point Drift in C++ / Eigen3 with Visualization

Reference: Point Set Registration: Coherent Point Drift, Andriy Myronenko and Xubo Song (2010)

**Note: This repo will be cleaned soon**

## Algorithm

This is an expectation maximization algorithm which estimates the rigid transformation parameters of between two point clouds. It assumes that one point cloud is a gaussian mixture model from which the other is generated, and then the expectation is maximized in this framework.

A rigid transformation includes rotation, translation and scaling.

## Usage

The actual CPD portion is separated out cleanly in a header so it can be reutilized without the visualization.



## References
