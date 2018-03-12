#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:30:05 2017

@author: huy
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.stats import cauchy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from copy import copy, deepcopy
import os

def read_graph(name, k, Gtype = 'link'):
    G = pd.read_csv(path + '/' + name, header = None, index_col = None)
    G = np.array(G)
    d = len(G)
    Q0 = G[(d-8):,:]; Q0 = np.delete(Q0, range(8,d), axis = 1)
    G = np.delete(G, range(d-8,d), axis = 0)
    
    not0 = np.where(G[:,0] != 0)[0][0]
    G = G/(G[not0,0])

    return G,Q0


def createNetwork(N):
    G = 1.*np.random.binomial(1, 0.025, (N,N))
    for i in range(N):
        G[i,i] = 0
        for j in range(N):
            if G[i,j] == 1:
                G[j, i] = 1
    return G


def Kuramoto(G, phase, frequencies, T, K, N):
    r = []
    K = deepcopy(G) * (K*1./N)
    dt = 1./T
    for i in range(T+1):
        tiled = np.tile(phase, (N, 1))
        sin = np.sin(phase - tiled.T)
        temp = []
        for i in range(len(phase)):
            temp.append(np.dot(sin[i,:], K[i,:].T))
        phase += dt*(frequencies + np.array(temp))
        r.append(np.abs(np.mean(np.exp(phase*np.complex(0,1)))))
    
    return r


def simulation(G, k, N, T):
    mean_r = []
    num = np.int(T/2 + 1)
    for i in range(100):
        phase = np.random.uniform(1, 2*np.pi, N)
        frequencies = np.random.standard_cauchy(N)
        r = Kuramoto(G, phase, frequencies, T, k, N)
        r1 = r[num:]
        mean_r.append(np.mean(r1))
    
    return np.mean(mean_r)

def test():
    N = 36
    G = createNetwork(N)
    T = 500#timestep
    dt = 1./T
    Klow = 1
    Khigh = 500
    dK = np.linspace(Klow, Khigh, 500)
    R = []
    cur = time.time()
    for k in dK:
        print(k)
        R.append(simulation(G,k,N,T))
    print(time.time() - cur)
    plt.plot(dK,R)
    plt.ylim(0,1)
#    plt.xlim(0,1.5)
 #   name = '{}_{}.png'.format(N, T)
 #   plt.savefig('/home/huy/Desktop/ResearchProject/' + name)   
    return R

def main():
    K = np.linspace(20.2,25,25)
    r = []
    Gtype = ['link','node']
    os.chdir('/home/huy/Desktop/ResearchProject/graph/new/prior')
    for g in Gtype:
        path = os.getcwd() +'/' + g
        temp = []
        print(g)
        for k in K:
            cur = time.time()
            temp.append(run(k, g))
            print(time.time() - cur)
        r.append(temp)
        
    plt.plot(K,r[0])
    plt.plot(K,r[1])
#    plt.ylim(0.15,.3)
    plt.legend(['Glink', 'Gnode'])    

def run(k, Gtype = 'link'):
    filename = os.listdir(path)
    R = []
    for f in filename:
        print('At k = {}, considering {}'.format(k,f))        
        G, Q = read_graph(f, k, Gtype)
        N = len(G)
        T = 1000
        R.append(simulation(G, k, N, T))
    return np.mean(np.array(R))
 
if __name__ == '__main__':    
    test()

    