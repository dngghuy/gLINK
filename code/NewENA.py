#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:26:51 2017

@author: huy
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import time

def Phase2fromGraph(name, Gtype = 'link'):

    os.chdir('/home/huy/Desktop/ResearchProject/graph/new/prior')
    path = os.getcwd()
    G = pd.read_csv(path + '/' + name, header = None, index_col = None)
    G = np.array(G)
    d = len(G)
    Q0 = G[(d-8):,:]; Q0 = np.delete(Q0, range(8,d), axis = 1)
    G = np.delete(G, range(d-8,d), axis = 0)
    
    not0 = np.where(G[:, 0] != 0)[0][0]
    G = G/(G[not0,0])
    
    RunPhase2(G, type = Gtype)
    
    return 'Done'

def step_function(a,b):
    if ((a-b) < 0):
        return 0
    else:
        return 1

def RunPhase2(G, type='link'):
    if len(G) == 36:
        nin = 8
        nout = 8
        m = 20
    else:
        nin = 8
        nout = 8
        m = len(G) - nin - nout

    Q = Output(nin, nout, m, G)
    err = Error(nin, nout, m, Q, Q0)
    robust_link = robustness(nin, nout, m, G, Q0, damage_type='link', h = 0.007)
    robust_node = robustness(nin, nout, m, G, Q0, damage_type = 'node', h = 0.007)
    
    if type == 'link':
        G_link, m_link = Phase2_LinkRemoval(nin, nout, m, G, Q0, curr_err = err, curr_robust = robust_link, sigma= 10**(-4), h = 0.007, iterations = 300000)
        robust_link = robustness(nin, nout, m_link, G_link, Q0, damage_type='link', h = 0.007)

        Q_temp = np.zeros((8,len(G_link)))
        Q_temp[0:8,0:8] = Q0
        G_link = np.concatenate((G_link,Q_temp), axis = 0)
        G_link = pd.DataFrame(G_link)
        fname = path + '/link_new' + '/G_linkNEW_{}.csv'.format(time.ctime())
        G_link.to_csv(fname, index = False, header= False)
#    m = len(G_link) - nin - nout
#    Q_link = Output(nin, nout, m, G_link)
#    err_link = Error(nin, nout, m, Q_link, Q0)
#    robust_link = robustness(nin, nout, m, G_link, Q0, damage_type='link', h = 0.007)
    
    else:
        G_node, m_node = Phase2_NodeRemoval(nin, nout, m, G, Q0, curr_err = err, curr_robust = robust_node, sigma= 10**(-4), h = 0.007, iterations = 300000) 
        robust_node = robustness(nin, nout, m_node, G_node, Q0, damage_type = 'node', h = 0.007)
        Q_temp = np.zeros((8,len(G_node)))
        Q_temp[0:8,0:8] = Q0
        G_node = np.concatenate((G_node,Q_temp), axis = 0)
        G_node = pd.DataFrame(G_node)
        name = path + '/node_new' + '/G_nodeNEW_{}.csv'.format(time.ctime())
        G_node.to_csv(name, index = False, header= False)
#        m = len(G_node) - nin - nout
#        Q_node = Output(nin, nout, m, G_node)
#        err_node = Error(nin, nout, m, Q_node, Q0)
#        robust_node = robustness(nin, nout, m, G_node, Q0, damage_type='node', h = 0.007)
    print('Done')
      
def randomAdjacencyMatrix(nin, m, nout):
    
    size = nin + m + nout
    G = np.zeros((size,size))
    binom_1 = np.zeros((m, nin+m))
    for i in range(m):
        binom_1[i,:] = np.random.binomial(1,0.7,(nin+m))
    binom_2 = np.random.binomial(1,0.7, (nout,m))
    G[nin:(nin+m),:(nin+m)] = binom_1
    G[(nin+m):,nin:(nin+m)] = binom_2
    for i in range(nin, nin+m):
        G[i,i] =0

    #Checking
    for i in range(size):
        for j in range(size):
            if (G[i,j] == 1) & (G[j,i] == 1):

                G[j,i] = 0
    for i in range(m):
        if sum(G[(nin+i),:]) == 0:
            G[:,(nin+i)] = 0

    return G
    
    
def Output(nin, nout, m, G):
    size = nin + nout + m
    adja_temp = deepcopy(G)
    for i in range(size):
        if sum(adja_temp[:,i] != 0):
            adja_temp[:,i] = -1*adja_temp[:,i]/sum(adja_temp[:,i])
    adja_temp += np.diag(np.ones(m+nin+nout))
    b = np.zeros(size); b[:nin,] = adja_temp.diagonal()[:nin]

##########################################
#Making output pattern
##########################################
    Q = np.zeros((nout, nin))
    for i in range(nin):
        temp = deepcopy(b); temp[:nin] = 0
        temp[i] = 1
        try:
            Q[:,i] = np.linalg.solve(adja_temp, temp)[-nout:]
    
        except np.linalg.linalg.LinAlgError as err:        
            if 'Singular matrix' in err.message:
                Q[:,i] = (np.linalg.lstsq(adja_temp, temp)[0])[-nout:]
            else:
                raise
        
    return Q
    
def Error(nin, nout, m, Q, Q0):
    E = np.zeros((nout,nin))
    for i in range(nin):
        for j in range(nout):
            E[j,i] = (Q[j,i] - Q0[j,i])**2
    err = (1./(2*nin)) * sum(np.sum(E, axis = 0))
    
    return err
    
    
def path_mutation(nin, nout, m, G, add=True):
    size = nin+nout+m
    nmid = np.random.randint(1,m+1,1)[0]
    input_node = np.random.randint(0,nin,1)[0]
    output_node = np.random.randint(nin+m, size, 1)[0]
    middle_node = np.random.choice(range(nin,nin+m), size = nmid, replace=False)
    G_temp = deepcopy(G)
    if add:
        G_temp[middle_node[0], input_node] = 1
        for i in range(len(middle_node)-1):
            G_temp[middle_node[i+1], middle_node[i]] = 1
            if G_temp[middle_node[i], middle_node[i+1]] == 1:
                G_temp[middle_node[i], middle_node[i+1]] = 0
        G_temp[output_node, middle_node[-1]] = 1
    else:
        G_temp[middle_node[0], input_node] = 0
        for i in range(len(middle_node) -1 ):
            G_temp[middle_node[i+1], middle_node[i]] = 0
        G_temp[output_node, middle_node[-1]] = 0

    return G_temp
    
def robustness(nin, nout, m, G, Q0, damage_type, h=0.007):
    #make graphs using path mutation
    temp = 0
    if damage_type == 'link':
        #Calculate number of links
        link_tuple = np.where(G != 0)
        link_num = len(link_tuple[0])
        for i in range(link_num):
            G_temp = deepcopy(G)
            G_temp[link_tuple[0][i], link_tuple[1][i]] = 0
#            if sum(G_temp[link_tuple[0][i], 0:nin]) == 0:
#                G_temp[:,link_tuple[0][i]] = 0
            Q_temp = Output(nin, nout, m, G_temp)
            err = Error(nin, nout, m, Q_temp, Q0)
            temp = temp + step_function(h, err)
            
        result = 1.*temp/link_num
            
    if damage_type == 'node':
#        size = nin + nout + m
        for i in range(m):
            G_temp = deepcopy(G)
            G_temp = np.delete(G_temp, (nin+i), 0)
            G_temp = np.delete(G_temp, (nin+i), 1)         
            m_temp = m - 1
            Q_temp = Output(nin, nout, m_temp, G_temp)
            err_temp = Error(nin, nout, m_temp, Q_temp, Q0)
            temp = temp + step_function(h, err_temp)
            
        result = 1.*temp/m
            
    return result
    
def Phase2_LinkRemoval(nin, nout, m, G, Q0, curr_err, curr_robust, sigma = 10**(-4), h = 0.007, iterations = 100000):
#Re-calculate the error and robustness of G

    for i in range(iterations):
        #Checking
        count = 0
        for q in range(m):
            if sum(G[(nin+q),:]) == 0:
                print('THERE IS A NODE WITHOUT INPUT CONNECTION!!! At iteration {} and at middle node {}'.format(i, nin + q))
                G = np.delete(G, (nin+q), axis = 0)
                G = np.delete(G, (nin+q), axis = 1)
                count += 1
                m1 = len(G) - nin - nout
                Q = Output(nin, nout, m1, G)
                curr_err = Error(nin, nout, m1, Q, Q0)
                curr_robust = robustness(nin, nout, m1, G, Q0, damage_type='node', h=0.007)
        m = m - count        
        
        
        if curr_robust == 1.0:
            break
        if (i+1) % 1000 ==0:
            print(i)
            print('Current error: {}'.format(curr_err))
            print('Current robustness: {}'.format(curr_robust))
            print('Current shape: {}'.format(G.shape))
        G_temp = deepcopy(G)
        add = np.random.choice([True, False], size = 1)[0]
        G_temp = path_mutation(nin, nout, m, G_temp, add = add)
        Q_temp = Output(nin, nout, m, G_temp)
        err_temp = Error(nin, nout, m, Q_temp, Q0)
        #Calculate robustness of link removal
        robust_temp = robustness(nin, nout, m, G_temp, Q0, damage_type='link', h = 0.007)

        derr = err_temp - curr_err
        drobust = curr_robust - robust_temp        
    
        if err_temp > h:
            #Decision will be based on the flow error
            if derr <= 0:
                G = G_temp
                curr_err = err_temp
                curr_robust = robust_temp
            if derr > 0:
                prob = np.e**(-derr/(curr_err*sigma))
                accept = np.random.binomial(1, prob, size = 1)
                if accept == 1:
                    G = G_temp
                    curr_err = err_temp
                    curr_robust = robust_temp
        

        if err_temp < h:
            #Decision will be based on robustness
            if drobust <= 0:
                G = G_temp
                curr_err = err_temp
                curr_robust = robust_temp

            if drobust > 0:
                prob = np.e**(-drobust/((1-curr_robust)*sigma))
                accept = np.random.binomial(1, prob, size = 1)
                if accept == 1:
                    G = G_temp
                    curr_err = err_temp
                    curr_robust = robust_temp         
                
    return G, m
    
def Phase2_NodeRemoval(nin, nout, m, G, Q0, curr_err, curr_robust, sigma = 10**(-4), h = 0.007, iterations = 100000):
    #Re-calculate the error and robustness of G
    for i in range(iterations):
        count = 0
        for q in range(m):
            if sum(G[(nin+q),:]) == 0:
                print('THERE IS A NODE WITHOUT INPUT CONNECTION!!! At iteration {} and at middle node {}'.format(i, nin + q))
                G = np.delete(G, (nin+q), axis = 0)
                G = np.delete(G, (nin+q), axis = 1)
                count += 1
                m1 = len(G) - nin - nout
                Q = Output(nin, nout, m1, G)
                curr_err = Error(nin, nout, m1, Q, Q0)
                curr_robust = robustness(nin, nout, m1, G, Q0, damage_type='node', h=0.007)
        m = m - count        
        
        if curr_robust == 1.0:
            break
        if (i+1) % 1000 == 0:
            print(i)
            print('Current error: {}'.format(curr_err))
            print('Current robustness: {}'.format(curr_robust))
            print('Current shape: {}'.format(G.shape))

        G_temp = deepcopy(G)
        add = np.random.choice([True, False], size = 1)[0]
        G_temp = path_mutation(nin, nout, m, G_temp, add = add)
        Q_temp = Output(nin, nout, m, G_temp)
        err_temp = Error(nin, nout, m, Q_temp, Q0)
        robust_temp = robustness(nin, nout, m, G_temp, Q0, damage_type='node', h = 0.007)

        derr = err_temp - curr_err
        drobust = curr_robust - robust_temp
    
        if err_temp > h:
            #Decision will be based on the flow error
            if derr <= 0:
                G = G_temp
                curr_err = err_temp
                curr_robust = robust_temp

            if derr > 0:
                prob = np.float64(np.e**(-derr/(curr_err*sigma)))
                accept = np.random.binomial(1, prob, size = 1)
                if accept == 1:
                    print('Accept')
                    G = G_temp
                    curr_err = err_temp
                    curr_robust = robust_temp
        if err_temp < h:
            # Decision will be based on robustness
            if drobust <= 0:
                G = G_temp
                curr_err = err_temp
                curr_robust = robust_temp
            if drobust > 0:
                prob = np.float64(np.e**(-drobust/((1-curr_robust)*sigma)))
                accept = np.random.binomial(1, prob, size = 1)
                if accept == 1:
                    print('Accept')
                    G = G_temp
                    curr_err = err_temp
                    curr_robust = robust_temp

    return G, m
    
if __name__ == '__main__':
    
    os.chdir('/home/huy/Desktop/ResearchProject/graph/new/prior')
    path = os.getcwd()
    
    for j in range(2,100):
        nin = 8
        nout = 8
        m = 20
        size = m + nin + nout
    
        G = randomAdjacencyMatrix(nin, m, nout)
        h=0.007
        sigma = 1*10**(-4)
        S = 0.02
        Q = Output(nin, nout, m, G)

        #Make ideal output pattern with K = 4 activated output - In this, we just randomize
        K = 4
        Q0 = np.random.uniform(size = (nout, nin))            
        for row in range(len(Q)):
            r = np.random.choice(7, K, replace = False)
            Q0[row][r] = 0
    
        for p in range(Q0.shape[1]):
            if sum(Q0[:,p] == 0):
                Q0[:, p] == 0
            Q0[:,p] = Q0[:,p]/sum(Q0[:,p])

        err = Error(nin, nout, m, Q, Q0)
    
        #Phase 1: Pattern Recognition
        for i in range(30000):
            add_temp = np.random.choice([True, False], size = 1)
            G_temp = deepcopy(G)
            G_temp = path_mutation(nin, nout, m, G_temp, add_temp)
            Q_temp = Output(nin, nout, m, G_temp)
            err_temp = Error(nin, nout, m, Q_temp, Q0)
            derr = err_temp - err
            if derr <= 0:
                G = G_temp
                err = err_temp
            else:
                prob = np.e**(-derr/(err*sigma))
                accept = np.random.binomial(1,prob,size = 1)
                if accept == 1:
                    G = G_temp
                    err = err_temp
        for i in range(m):
            if sum(G[(nin+m),:]) == 0:
                print('THERE IS A NODE WITHOUT INPUT CONNECTION!!!')
                G = np.delete(G, (nin+m), axis = 0)
                G = np.delete(G, (nin+m), axis = 1)
                print('Now the size is: {}'.format(G.shape))
        Q = Output(nin, nout, m, G)
        err = Error(nin, nout, m, Q, Q0)    
    
        robust_link = robustness(nin, nout, m, G, Q0, damage_type='link', h = 0.007)
        robust_node = robustness(nin, nout, m, G, Q0, damage_type = 'node', h = 0.007)
        print('Graph {}; robust link is: {}; robust node is: {}'.format(j, robust_link, robust_node))

        
        Q_temp = np.zeros((8,len(G)))
        Q_temp[0:8,0:8] = Q0
        G = np.concatenate((G,Q_temp), axis = 0)
        G = pd.DataFrame(G)
        name = path + '/G_{}_{}_{}_{}.csv'.format(j, err, robust_link, robust_node)
        G.to_csv(name, index = False, header= False)
        
        #Phase 2:
        RunPhase2(G, type='link')
        RunPhase2(G, type='node')
        
    
