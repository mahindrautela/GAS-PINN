import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn_string import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import math
import time

# number of training samples
num_train_samples = 25000
    
# number of test samples
num_test_samples = 2500
    
# Other variables
c = 0.5
L = 10
n = L
T = 1

# Initial conditions
def u0(t):
    z = -np.sin(1*math.pi*t)
    return z

def du0_dt(tx):
    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

# Analytical solution
xx = np.linspace(0,L,num_test_samples)
tt = np.linspace(0,T,num_test_samples)
usol = np.zeros((num_test_samples,num_test_samples))
for i,xi in enumerate(xx):
    for j,tj in enumerate(tt):
        usol[i,j] = -np.sin(math.pi*xi)*np.cos(n*math.pi*c*tj/L)

plt.plot(xx,usol[:,251])

########################################################################
######################## collocation points ############################
########################################################################

# create training input
tx_eqn = np.random.rand(num_train_samples, 2)
tx_eqn[..., 0] = T*tx_eqn[..., 0]                      # t =  0 ~ +1
tx_eqn[..., 1] = L*tx_eqn[..., 1]                      # x = 0 ~ +10
print('\nShape of t_eqn ==>',tx_eqn.shape)

tx_ini = np.random.rand(num_train_samples, 2)
tx_ini[..., 0] = 0                                     # t = 0
tx_ini[..., 1] = L*tx_ini[..., 1]                      # x = 0 ~ +10
print('\nShape of tx_ini ==>',tx_ini.shape)

tx_bnd = np.random.rand(num_train_samples, 2)
tx_bnd[..., 0] = T*tx_bnd[..., 0]                      # t =  0 ~ +1
tx_bnd[..., 1] = L*np.round(tx_bnd[..., 1])            # x =  0 or +10
print('\nShape of tx_bnd ==>',tx_bnd.shape)

u_zero = np.zeros((num_train_samples, 1))
u_ini = u0(tx_ini[:,1,None])
du_dt_ini = np.zeros((num_train_samples, 1))

x_ini_sort = np.sort(tx_ini[:,1])
u_ini0 = u0(x_ini_sort)
fig = plt.figure(figsize=(7,4))
plt.plot(x_ini_sort,u_ini0)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Initial condition on u')

#########################################################################
########################### TRAINING PINNs ##############################
#########################################################################

# build a core network model
network = Network.build()
network.summary()

# build a PINN model
pinn = PINN(network,c).build()

# train the model using L-BFGS-B algorithm
begin = time.time()
x_train = [tx_eqn, tx_ini, tx_bnd]
y_train = [u_zero, u_ini, du_dt_ini, u_zero]
lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
lbfgs.fit()
end = time.time()
totaltime = end-begin
print("\n Total runtime of the program is",totaltime)

#########################################################################
######################## PREDICTION #####################################
#########################################################################

# predict u(t,x) distribution
t_flat = np.linspace(0, T, num_test_samples)
x_flat = np.linspace(0, L, num_test_samples)
t, x = np.meshgrid(t_flat, x_flat)
tx = np.stack([t.flatten(), x.flatten()], axis=-1)
u = network.predict(tx, batch_size=num_test_samples)
u = u.reshape(t.shape)

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 3) # A grid layout to place subplots within a figure.
plt.subplot(gs[0, :])
vmin, vmax = -1.0, +1.0
plt.pcolormesh(t, x, u, cmap='rainbow', shading = 'auto', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)

# plot u(t=const, x) cross-sections
tfrac = np.array([0.25,0.5,0.75])
t_cross_sections = (T*tfrac).tolist()
idx = [int(x) for x in (num_test_samples*tfrac)]

for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    full = np.full(t_flat.shape, t_cs)
    tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    #print(u.shape)
    plt.plot(x_flat, u, 'b-', linewidth = 2)
    plt.plot(x_flat, usol[:,idx[i]], 'r--', linewidth = 2)
    plt.title('t={}'.format(t_cs))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.ylim(-1,1)
    plt.legend(['Exact','Prediction'], loc = 'upper right',fontsize=4)
plt.tight_layout()
plt.savefig('result_img_dirichlet.png', transparent=True)
plt.savefig('result_img_dirichlet.eps', transparent=True, format='eps',
            dpi = 1200)
plt.show()

# comparison plots
idx = np.array(t_cross_sections) * num_test_samples
mm = 0
tx = np.stack([np.full(t_flat.shape, t_cross_sections[mm]), x_flat], axis=-1)
u_snap = network.predict(tx, batch_size=num_test_samples)

fig2 = plt.figure(figsize=(7,4))
plt.plot(x_flat,usol[:,int(idx[mm])], 'b-', linewidth = 2)       
plt.plot(x_flat,u_snap, 'r--', linewidth = 2)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')    
plt.title('$t = 0.25s$', fontsize = 10)
#plt.set_xlim([-1.1,1.1])
plt.legend(['Exact','Prediction'],loc = 'best')
plt.ylim(-1.1,1.1)

# Save data for inversion = time snapshots  
from scipy.io import savemat
mdic1 = {"a1": u_snap, "label": "experiment"}
savemat("u_observe_snap_tquat.mat", mdic1)
