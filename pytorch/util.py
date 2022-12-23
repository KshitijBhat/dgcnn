#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from topologylayer.functional.levelset_dionysus import Diagramlayer as DiagramlayerToplevel
from topologtlaye.functional.utils_dionysus import top_cost


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def Toplevel3Dcost(x):


    ''' Diagramlayer Toplevel Setup'''
    dtype=torch.float32
    fx,fy,fz = 32, 64, 1024
    axis_x = np.arange(0, fx)
    axis_y = np.arange(0, fy)
    axis_z = np.arange(0, fz)
    
    points = np.vstack(np.meshgrid(axis_x,axis_y,axis_z)).reshape(3,-1).T
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    faces = tri.simplices.copy()
    F = DiagramlayerToplevel().init_filtration(faces)
    diagramlayerToplevel = DiagramlayerToplevel.apply
    ''' '''
    return top_cost(x,diagramlayerToplevel,F)



def Toplevel2Dcost(x):


    ''' Diagramlayer Toplevel Setup'''
    dtype=torch.float32
    fx,fy = 32, 40
    axis_x = np.arange(0, fx)
    axis_y = np.arange(0, fy)   
    grid_axes = np.array(np.meshgrid(axis_x, axis_y))
    grid_axes = np.transpose(grid_axes, (1, 2, 0))
    from scipy.spatial import Delaunay
    tri = Delaunay(grid_axes.reshape([-1, 2]))
    faces = tri.simplices.copy()
    F = DiagramlayerToplevel().init_filtration(faces)
    diagramlayerToplevel = DiagramlayerToplevel.apply
    ''' '''

    return top_cost(x,diagramlayerToplevel,F)


def cal_top_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    loss = loss + Toplevel2Dcost(pred)
    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
