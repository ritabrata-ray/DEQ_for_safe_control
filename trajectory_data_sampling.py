import numpy as np

import numpy.random as npr

# def wiener1(dt=0.1,X0=X(t=0),Nt=1000):
#     """ Input variables:
#     dt    time step
#     X0    intial value, X(t=0) = X0
#     Nt    number of time steps
#     """
#     # initialize start value
#     res = X0
#     # calculate and store time series
#     for ii in range(1,Nt):
#         #         X(t+dt)=X(t)+sqrt(dt)*npr.randn(Nt)
#         res += sqrt(dt)*npr.randn(Nt)
#
#     # return final value after t = N*dt
#     return res



dx = 0.4
dT = 0.5
T_max = 10
T_min = 0

def dynamics_runner(x_0, f, sigma, T_max):
    dim = np.shape(x_0)[0]
    noise_dim = np.shape(sigma(x_0))[1]
    horizon = int(np.ceil(T_max/dT))
    traj = np.zeros((horizon+1,dim))
    x_h = x_0
    recovered = False
    for h in range(horizon+1):
        traj[h,:] = x_h
        dw_h = np.sqrt(dT)*npr.randn(noise_dim)
        x_h = f(x_h) * dT + np.dot(sigma(x_h), dw_h)
        if (x_h >= 2.0):
            recovered = True
    return traj, recovered

def sample_from_box(low, high, size=(1,)):
    """
    Generate random vectors uniformly from a box defined by low and high values.

    Parameters:
        low: array_like
            Lower bounds of the box.
        high: array_like
            Upper bounds of the box.
        size: int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k),
            then m * n * k samples are drawn. Default is (1,).

    Returns:
        ndarray
            Random vectors sampled from the box.
    """
    low = np.asarray(low)
    high = np.asarray(high)
    return np.random.uniform(low=low, high=high, size=size)


def MC_sampler(N_P=100,N_T =100): #number of trajectories required is N_P x N_T
    MC_samples_dict = {}# dictionary to store the map F(x,T) at samples
    # Set of interest is Omega x tau = [-10,2] x [0,10] and safety for x \geq 2 i.e., \phi(x) = x-2
    # sampling dx=0.4, dT =0.5. We want to estimate the recovery probability
    low = np.array([-10])  # Lower bounds of the box
    high = np.array([2])  # Upper bounds of the box
    sample_size = (N_P, 1)  # Sample size (5 vectors of dimension 1)
    random_vectors = sample_from_box(low, high, size=sample_size)
    random_T = npr.uniform(T_min,T_max,N_T)
    for v in random_vectors:
        for T in random_T:
            recovery_count = 0
            for s in range(10): # 10 MC samples per trajectory input (x,T)
                traj, recovery = dynamics_runner(v,f,sigma,T) #run trajectory upto time T
                if(recovery):
                    recovery_count += 1
                # MC_samples_data.append(traj)
            F_v_T = recovery_count/10
            MC_samples_dict[tuple(v),T] = F_v_T
    sample_data = np.array(list(MC_samples_dict.items()), dtype=object)
    return sample_data

f = lambda x: 0.7
sigma = lambda x: [[1.0]]

data = MC_sampler(100,100)
np.save('1_d_interior_data.npy',data)
loaded_data = np.load('1_d_interior_data.npy', allow_pickle=True)
N_b = 10000 #boundary or initial data samples
boundary_edges = ['A', 'B', 'C', 'D']
boundary_points_dict={}
# the boundary is the rectangle between T=0 to T=10 and x=-10 to x=2.
# Each edge has a different weight: Edge A the x=-10 -> x=2 line at T = 0 has weight 12/44 = 3/11 (edge length by perimeter)
# Edge C the x=-10 -> x=2 line at T = 10 has weight 12/44 = 3/11 (edge length by perimeter)
# Edge B the T=10 -> T=10 at x = 2 has weight 10/44 = 5/22 = 2.5/11 (edge length by perimeter)
# Edge D the T=10 -> T=10 at x = -10 has weight 10/44 = 5/22 = 2.5/11 (edge length by perimeter)
boundary_sample_edges = npr.choice(boundary_edges,N_b,p=[3/11,5/22,3/11,5/22])
for e in boundary_sample_edges:
    if (e == 'A'):
        v = sample_from_box([-10],[2],1)
        T = 0
        if (v >= 1.99): # there will be no samples if we make v>= 2.0
            boundary_points_dict[tuple(v),T] = 1
        else:
            boundary_points_dict[tuple(v), T] = 0
    elif (e == 'B'):
        T = npr.uniform(T_min, T_max)
        v = [2]
        boundary_points_dict[tuple(v), T] = 1
    elif (e == 'C'):
        v = sample_from_box([-10], [2], 1)
        T = 10
        if (v >= 1.99):  # there will be no samples if we make v>= 2.0
            boundary_points_dict[tuple(v), T] = 1
        else:
            recovery_count = 0
            for s in range(10):  # 10 MC samples per trajectory input (x,T)
                traj, recovery = dynamics_runner(v, f, sigma, T)  # run trajectory upto time T
                if (recovery):
                    recovery_count += 1
                # MC_samples_data.append(traj)
            F_v_T = recovery_count / 10
            boundary_points_dict[tuple(v), T] = F_v_T
    elif (e == 'D'):
        T = npr.uniform(T_min, T_max)
        v = [-10]
        recovery_count = 0
        for s in range(10):  # 10 MC samples per trajectory input (x,T)
            traj, recovery = dynamics_runner(v, f, sigma, T)  # run trajectory upto time T
            if (recovery):
                recovery_count += 1
            # MC_samples_data.append(traj)
        F_v_T = recovery_count / 10
        boundary_points_dict[tuple(v), T] = F_v_T
boundary_data = np.array(list(boundary_points_dict.items()), dtype=object)
np.save('1_d_boundary_data.npy',boundary_data)
loaded_boundary_data = np.load('1_d_boundary_data.npy', allow_pickle= True)
#print(loaded_boundary_data)