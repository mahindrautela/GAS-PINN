# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:55:10 2023

@author: mahin
"""
import numpy as np
import math

class Forcing:
    
    def F_tone(delT,tlim,f,amp,T):
        Nc = 5
        t1 = np.arange(delT,2*tlim,delT)
        t2 = np.arange(2*tlim+delT,3*tlim,delT)
        t3 = np.arange(3*tlim+delT,T+delT,delT)
        t = np.concatenate((t1,t2,t3))
    
        f1 = np.zeros((len(t1),))
        f2 = amp * np.sin(2*3.142*f*t2) * (1-np.cos(2*3.142*f*t2/Nc))
        f3 = np.zeros((len(t3),))
        F = np.concatenate((f1,f2,f3))

        return (t,F)
        
    def F_tri(delT, T):
        t0 = np.arange(delT,99e-6,delT)
        t1 = np.arange(100e-6,150e-6,delT)
        t2 = np.arange(151e-6,200e-6,delT)
        t3 = np.arange(201e-6,T,delT)
        T = np.concatenate((t1,t2,t3))

        f0 = np.zeros((len(t0),))
        f1 = (1/50e-6)*(t1-100e-6)
        f2 = (-1/50e-6)*(t2-200e-6)
        f3 = np.zeros((len(t3),))
        Ft = np.concatenate((f0,f1,f2,f3))
        
        return (T,Ft)
        
    def F_sin(f,tx):
        t = tx[...,0,None]
        F = -np.sin(2*math.pi*f*t)
        return F