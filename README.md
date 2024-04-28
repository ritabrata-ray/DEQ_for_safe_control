# DEQ_for_safe_control
10-707: Advanced Deep Learning Course Project

The trajectory_data_sampling.py script performs monte carlo sampling of the long term safe probability of the 1-D system for both interior and boundary points and generates the 1_d_interior_data.npy and the 1_d_boundary_data.npy files.  This data is then used to train the pinn model and the deq model. The PINN is a fully connected 4 layer NN with 5150 parameters whereas the DEQ model has only 2650 parameters and performs better than the PINN model.
