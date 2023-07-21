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

# number of training samples
num_train_samples = 25000
    
# number of test samples
num_test_samples = 2500
    
# Other variables
c = np.linspace(0.1,1,10)
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

def RMS(S):
    rms = np.sqrt(np.mean(S**2))
    return rms
    
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

#########################################################################
########################### TRAINING PINNs ##############################
#########################################################################
xx = np.linspace(0,L,num_test_samples)
tt = np.linspace(0,T,num_test_samples)

# upload the data
import scipy.io
u_tquat = scipy.io.loadmat('u_observe_snap_tquat.mat')
u_tquat = u_tquat['a1']
Asignal = RMS(u_tquat)

# add gaussian noise to the data
mu = 0
sigma = 1
beta_range = [0.01, 0.025, 0.05, 0.1]
beta = beta_range[0]
noise = beta*(sigma*np.random.randn(2500,1) + mu)
Anoise = RMS(noise)

# noisy data
u_tquat_n = u_tquat + noise

# signal to noise ratio
snr = 20*np.log10(Asignal/Anoise)
print('\n Signal to noise ratio, SNR = ' + str(np.round(snr,2)) + ' dB')

# plot the data
fig = plt.figure(figsize=(7,4))
plt.plot(xx,u_tquat_n, '-', linewidth = 2)   
plt.xlabel('$x$', fontsize = 15)
plt.ylabel('Normalized u(x,t)', fontsize = 15)
plt.xticks(fontsize = 12) 
plt.yticks(fontsize = 12)    
#plt.savefig('exp_pinns.png', transparent=True)

U_diff = []
U_diff_mean = []
U_pred = []

for ic in c:
    
    ic = np.round(ic,2)
    print('\n ###### ------>>>> PINNs simulation at speed = ' + str(ic))
    
    if ic > 0.1:
        del usol, network, pinn, lbfgs, u_pred
    
    # Analytical solution
    usol = np.zeros((num_test_samples,num_test_samples))
    for i,xi in enumerate(xx):
        for j,tj in enumerate(tt):
            usol[i,j] = -np.sin(math.pi*xi)*np.cos(n*math.pi*ic*tj/L)
            
    network = Network.build()
    network.summary()
    
    # build a PINN model
    pinn = PINN(network,ic).build()
    
    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_zero, u_ini, du_dt_ini, u_zero]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()
    
    # prediction from models
    tfrac = np.array([0.25,0.5,0.75,1.0])
    t_cross_sections = (T*tfrac).tolist()
    idx = np.array(t_cross_sections) * num_test_samples
    mm = 0
    tx = np.stack([np.full(tt.shape, t_cross_sections[mm]), xx], axis=-1)
    u_pred = network.predict(tx, batch_size=num_test_samples)
    
    U_pred.append(u_pred)
    udiff = u_pred - u_tquat_n
    U_diff.append(udiff)
    U_diff_mean.append(np.mean(udiff**2))
    
    fig2 = plt.figure(figsize=(7,4))
    plt.plot(xx,u_tquat_n, 'b-', linewidth = 2)       
    plt.plot(xx,u_pred, 'r--', linewidth = 2)
    plt.xlabel('location, x', fontsize = 15)
    plt.ylabel('displacement field, u(x,t)', fontsize = 15)    
    #plt.title('PINNs prediction at speed, c =' + str(ic*10) + ' km/s', fontsize = 15)
    plt.legend(['Observation','PINNs prediction'], loc = 'best')
    plt.ylim(-1.1,1.1)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('results_c_'+str(int(ic*10))+'.png', transparent=True, dpi=300)

# Save data for inversion = time snapshots  
from scipy.io import savemat
mdic1 = {"a1": U_diff_mean, "label": "experiment"}
savemat("mse_speed_estimation.mat", mdic1)

# plot the mse
fig3 = plt.figure(figsize=(7,4))
plt.plot(c*10,np.array(U_diff_mean), 'b-', linewidth = 2)       
plt.xlabel('speed, c in km/s', fontsize = 15)
plt.ylabel('mean squared error (MSE)', fontsize = 15)    
#plt.title('PINNs prediction at speed, c =' + str(ic*10) + ' km/s', fontsize = 15)
plt.ylim(-1.1,1.1)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)