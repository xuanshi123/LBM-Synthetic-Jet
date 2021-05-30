# LBM-Synthetic-Jet

ModelDisplayUI.py gives visual display of the voxelized point grid of the 3D model STL file.

main.py execute commands of grid voxelization and and flow conditions, then run LBM simulation based on numba and GPUs.

Missing fixed Pressure and Reverse Flow Prevention on the right side external boundary!!!!!!!!!

The multiprocess version 
https://github.com/xuanshi123/LBM-multiProcess
assigns one process to one GPU to circumvent Global Interpreter Lock and speed up the calculation for multiplt GPUs.
