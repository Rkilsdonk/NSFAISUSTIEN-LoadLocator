import matplotlib.pyplot as plt
import time
import helics as h
import logging
import pandas as pd
import numpy as np
import argparse
import torch as torch
import json as json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cmath as cmath
#from transformer tutorial pytorch
from transformer_tutorial_accompaniment import MultiHeadAttention
from transformer_tutorial_accompaniment import gen_batch, jagged_to_padded, benchmark
from transformer_tutorial_accompaniment import TransformerEncoderLayer
from transformer_tutorial_accompaniment import TransformerDecoderLayer
from transformer_tutorial_accompaniment import Transformer, TransformerDecoder, TransformerEncoder,TsTransformer,LSTMGS, ActiveTransformer,ResidualMLP
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
if __name__ == "__main__":
    logging.basicConfig(filename="validationresults2.txt")
    VoltageTimeSeries=torch.load("RandomTimeSeries9.pt")
    T, total_dims = VoltageTimeSeries.shape
    # You'll need to specify N and E
    E = 6 
    N = total_dims-6  # number of response variables
     # number of control variables
    
    # Split VoltageTimeSeries into responses and controls
    responses = VoltageTimeSeries[:, :N]  # [T, N]
    controls = VoltageTimeSeries[:, -6:]   # [T, E]
    # Get controls from previous timestep (t-1)
    prevcontrols = controls[:-1]  # [T-1, E]
    curr_responses = responses[1:]  # [T-1, N] (responses at time t)
    
    # Create control state indicators (zero vs nonzero)
    results={}
    for i in range(T-288):
        if tuple(prevcontrols[i].tolist()) in results:
            numocc=results[tuple(prevcontrols[i].tolist())][0]
            results[tuple(prevcontrols[i].tolist())][1]=(results[tuple(prevcontrols[i].tolist())][1]*numocc+curr_responses[i])/(numocc+1)
            results[tuple(prevcontrols[i].tolist())][0]=results[tuple(prevcontrols[i].tolist())][0]+1
        else:
            results[tuple(prevcontrols[i].tolist())]=[1,curr_responses[i]]
    logger.info(len(results.keys()))
    # Average responses for each control state
    
    # logger.info(torch.mean(VoltageTimeSeries,dim=0))
    # logger.info(torch.var(VoltageTimeSeries,dim=0))
    # logger.info(torch.max(VoltageTimeSeries,dim=0))
    # logger.info(torch.min(VoltageTimeSeries,dim=0))
    # logger.info(torch.min(VoltageTimeSeries,dim=0)[0]-torch.max(VoltageTimeSeries,dim=0)[0])
    # logger.info(torch.max(torch.abs(VoltageTimeSeries),dim=0))
    # logger.info(torch.min(torch.abs(VoltageTimeSeries),dim=0))
    loss=torch.nn.MSELoss()
    #logger.info(f"Persistance:{loss(VoltageTimeSeries[1:,:-6],VoltageTimeSeries[:-1,:-6])}")
    model15=torch.load("model15.pt",weights_only=False)
    logger.info(f"model15.pt:{loss(VoltageTimeSeries[1:,:-6],model15(VoltageTimeSeries[:-1,:]))}")
    logger.info(f"model15.pt extended:{loss(VoltageTimeSeries[-288:,:-6],model15(VoltageTimeSeries[:-1,:])[-288:])}")
    #logger.info(f"Peristance extended:{torch.mean(loss(VoltageTimeSeries[-200:,:-6],VoltageTimeSeries[-201:-1,:-6]),dim=0)}")
    rollingmeans=torch.zeros_like(curr_responses)
    for i in range(T-1):
        rollingmeans[i]=torch.mean(responses[:i],dim=0)
    logger.info(f"exactmeanfull:{loss(rollingmeans,VoltageTimeSeries[1:,:-6])}")
    logger.info(f"exactmeanperiod:{loss(rollingmeans[-288:],VoltageTimeSeries[-288:,:-6])}")
    modelloss=0
    count=0
    for i in range(T-288,T):
        if(tuple(VoltageTimeSeries[i-1,-6:].tolist()) in results):
            modelloss+=loss(VoltageTimeSeries[i,:-6],results[tuple(VoltageTimeSeries[(i-1),-6:].tolist())][1])
        else:
            count=count+1
    modelloss=modelloss/288
    logger.info(torch.mean(loss(VoltageTimeSeries[-288:,:-6],VoltageTimeSeries[-289:-1,:-6]),dim=0))
    logger.info(modelloss)
    #logger.info(f"averageseeking loss{modelloss}")
    logger.info(count)
    logger.info(VoltageTimeSeries.shape)
    for i in range(1,T):
        meanloss=loss(torch.mean(dim=0,input=VoltageTimeSeries[0:i-1,:-6]),VoltageTimeSeries[i,:-6]).item()
        logger.info(f"{meanloss},")