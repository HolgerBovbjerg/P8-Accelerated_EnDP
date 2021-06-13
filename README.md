# P8-Accelerated_EnDP
This repository contains material developed as part of a signal processing master project at Aalborg University.

The purpose of the project is to investigate methods for making [Ensemble Density Propagation (EnDP)](https://ieeexplore.ieee.org/document/9231635) feasible for large DNNs, as this is not discussed in the original article. 

Through an analysis of the EnDP framework, it was found that distributed computing might be a good solution to make the framework a viable option in DNNs with large input and many layers.  

A distributed implementation of a convolution layer from the VGG-16 network is used to test how well the framwork distributes to a large network.  

## Repository structure
The repository has three main folders ```Function_files```, ```Tests``` and ```Strato```.
The ```Function_files``` folder contains various functions created as part of the project. 
The ```Tests``` folder contains test scripts used in the project. 
The ```Strato``` folder contains a guide to how the strato cloud is used with DASK and a ping script for testing proper connection to workers. 
