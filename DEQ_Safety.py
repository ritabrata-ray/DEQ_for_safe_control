import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#Dynamics functions
f = lambda x: 0.7
sigma = lambda x: 1.0

class TanhFixedPointLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # initialize output z to be zero
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate until convergence
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            if self.err < self.tol:
                break

        return z


class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :,
                                                        None] * self.linear.weight[None, :, :]
                z = z - torch.linalg.solve(J, g[:, :, None])[:, :, 0]
                self.iterations += 1

        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad: torch.linalg.solve(J.transpose(1, 2), grad[:, :, None])[:, :, 0])
        return z

def physics_loss(model, x, T):
    # Define your physics-informed loss function
    x.requires_grad_(True)
    T.requires_grad_(True)
    u = model(x, T)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_T = torch.autograd.grad(u, T, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    loss_pde = 0.5 * torch.mean((u_T - 0.5*((sigma(x))**2)*u_xx - f(x)*u_x)**2) # PDE loss
    return  10000 * loss_pde # multiplied by 10000 so that physics and boundary losses are comparable

def boundary_loss(model, x, T, F):
    # x.requires_grad_(True)
    # T.requires_grad_(True)
    u = model(x, T)
    loss_boundary = 0.5 * torch.mean((u - F) ** 2)  # Boundary loss may need some more weight since there is less boundary data
    return loss_boundary

# Optimizer and model instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class DEQ_model(nn.Module):
    def __init__(self):
        super(DEQ_model, self).__init__()
        # Define your neural network architecture
        self.fc1 = nn.Linear(2, 50)  # Input: (x, t)
        self.implicit_layer = TanhNewtonImplicitLayer(50, max_iter=40) #50 x 50 weight matrix weight tied
        self.fc2 = nn.Linear(50, 1)   #2650 params total

    def forward(self, x, T):
        # Concatenate space and time
        input_data = torch.cat((x.unsqueeze(1), T.unsqueeze(1)), dim=1)
        # Forward pass through the network
        out = self.fc1(input_data)
        out = self.implicit_layer(out)
        out = self.fc2(out)
        return out

model = DEQ_model().to(device)
optimizer_p = optim.Adam(model.parameters(), lr=0.001)
optimizer_b = optim.Adam(model.parameters(), lr= 0.01)

# Data loading and pre-processing

loaded_boundary_data = np.load('1_d_boundary_data.npy', allow_pickle= True)
x_T_tr_b = loaded_boundary_data[:,0]
x_T_tr_b_array = [list(i) for i in zip(*x_T_tr_b)]
x_tr_b = np.array([]) #empty array
T_tr_b = np.array([]) #empty array
F_tr_b = loaded_boundary_data[:,1]
F_tr_b = np.array(F_tr_b, dtype= float)
for state in x_T_tr_b_array[0]:
    x_tr_b = np.append(x_tr_b, state[0])
for time in x_T_tr_b_array[1]:
    T_tr_b = np.append(T_tr_b, time)

loaded_physics_data = np.load('1_d_interior_data.npy', allow_pickle=True)
x_T_tr_p = loaded_physics_data[:,0]
x_T_tr_p_array = [list(i) for i in zip(*x_T_tr_p)]
x_tr_p = np.array([]) #empty array
T_tr_p = np.array([]) #empty array
F_tr_p = loaded_physics_data[:,1]
F_tr_p = np.array(F_tr_p, dtype= float)
for state in x_T_tr_p_array[0]:
    x_tr_p = np.append(x_tr_p, state[0])
for time in x_T_tr_p_array[1]:
    T_tr_p = np.append(T_tr_p, time)

N_b = np.shape(T_tr_b)[0] # boundary data size
N_p = np.shape(T_tr_p)[0] # physics data size
# an 80-20% split of training and test data on both boundary and physics data
N_b_tr = int(N_b * 0.8)
N_p_tr = int(N_p * 0.8)
N_b_te = N_b - N_b_tr
N_p_te = N_p - N_p_tr
# Now we shuffle all data
boundary_indices = np.random.permutation(N_b)
physics_indices = np.random.permutation(N_p)
x_tr_b = x_tr_b[boundary_indices]
T_tr_b = T_tr_b[boundary_indices]
F_tr_b = F_tr_b[boundary_indices]
x_tr_p = x_tr_p[physics_indices]
T_tr_p = T_tr_p[physics_indices]
F_tr_p = F_tr_p[physics_indices]

x_te_b = x_tr_b[N_b_tr:]
T_te_b = T_tr_b[N_b_tr:]
F_te_b = F_tr_b[N_b_tr:]
x_tr_b = x_tr_b[0:N_b_tr]
T_tr_b = T_tr_b[0:N_b_tr]
F_tr_b = F_tr_b[0:N_b_tr]

x_te_p = x_tr_p[N_p_tr:]
T_te_p = T_tr_p[N_p_tr:]
F_te_p = F_tr_p[N_p_tr:]
x_tr_p = x_tr_p[0:N_p_tr]
T_tr_p = T_tr_p[0:N_p_tr]
F_tr_p = F_tr_p[0:N_p_tr]


#Evaluating on untrained model

figure_of_merit_naive = torch.sqrt(2.0 * boundary_loss(model, torch.tensor(x_te_p, dtype= torch.float), torch.tensor(T_te_p, dtype= torch.float),
                                torch.tensor(F_te_p, dtype = torch.float)))

print(figure_of_merit_naive)


# Training loop
num_epochs = 10
physics_batch_size = 100
boundary_batch_size = 100
for epoch in range(num_epochs):

    l_boundary_epoch_average = 0
    l_physics_epoch_average = 0
    for i in range(0,N_b_tr,boundary_batch_size): # runs same as "for i in range(0, N_p_tr, physics_batch_size):"
        batch_x_tr_b = torch.tensor(x_tr_b[i:i+boundary_batch_size],dtype=torch.float)
        batch_T_tr_b = torch.tensor(T_tr_b[i:i+boundary_batch_size],dtype=torch.float)
        batch_F_tr_b = torch.tensor(F_tr_b[i:i+boundary_batch_size], dtype=torch.float)
        batch_x_tr_p = torch.tensor(x_tr_p[i:i + physics_batch_size], dtype=torch.float)
        batch_T_tr_p = torch.tensor(T_tr_p[i:i + physics_batch_size], dtype=torch.float)
        # Compute boundary loss
        l_b = boundary_loss(model,batch_x_tr_b,batch_T_tr_b,batch_F_tr_b)
        # Optimization step
        optimizer_b.zero_grad()
        l_b.backward()
        optimizer_b.step()
        # Compute physics loss
        l_p = physics_loss(model, batch_x_tr_p, batch_T_tr_p)
        # Optimization step
        optimizer_p.zero_grad()
        l_p.backward()
        optimizer_p.step()
        # Accumulate average loss
        l_boundary_epoch_average += l_b.item()
        l_physics_epoch_average += l_p.item()
    l_boundary_epoch_average = l_boundary_epoch_average *(1/(N_b_tr/boundary_batch_size))
    l_physics_epoch_average = l_physics_epoch_average * (1 / (N_p_tr / physics_batch_size))
    # Print loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Boundary Loss: {l_boundary_epoch_average:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Physics Loss: {l_physics_epoch_average:.4f}")

   # Now we evaluate performance on test data
"""
test_physics_loss = physics_loss(model, torch.tensor(x_te_p, dtype=torch.float), torch.tensor(T_te_p, dtype=torch.float))
print(test_physics_loss)


test_boundary_loss = boundary_loss(model, torch.tensor(x_te_b, dtype=torch.float), torch.tensor(T_te_b, dtype= torch.float),
                                                        torch.tensor(F_te_b, dtype=torch.float))
print(test_boundary_loss)
"""
figure_of_merit = torch.sqrt(2.0 * boundary_loss(model, torch.tensor(x_te_p, dtype= torch.float), torch.tensor(T_te_p, dtype= torch.float),
                                torch.tensor(F_te_p, dtype = torch.float)))

print(figure_of_merit)
