# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:08:26 2018

@author: monish.mukherjee
"""
import matplotlib.pyplot as plt
import time
import helics as h
import logging
import pandas as pd
import numpy as np
import argparse
import torch as pt
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
logger.setLevel(logging.DEBUG)
def parse_voltage(s: str) -> complex:
    # drop the trailing “ V” and parse Python complex
    s = s.strip()
    if s.endswith("V"):
        s = s[:-1].strip()
    complexvalue=(complex(s))
    return complexvalue #seeing if polar coordinates work better

def destroy_federate(fed):
    grantedtime = h.helicsFederateRequestTime(fed, h.HELICS_TIME_MAXTIME)
    status = h.helicsFederateDisconnect(fed)
    h.helicsFederateDestroy(fed)
    logger.info("Federate finalized")


if __name__ == "__main__":
    pt.set_default_dtype(pt.float64)
    device = pt.accelerator.current_accelerator().type if pt.accelerator.is_available() else "cpu"
    logging.basicConfig(filename="RLController{}.log".format(time.ctime()))
    logger.info(device)
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-c', '--case_num',
    #                     help='Case number, must be either "1b" or "1c"',
    #                     nargs=1)
    # args = parser.parse_args()

    # #################################  Registering  federate from json  ########################################
    # case_num = str(args.case_num[0])
    fed = h.helicsCreateCombinationFederateFromConfig(f"RLControl.json")
    # h.helicsFederateRegisterInterfaces(fed, "Control.json")
    federate_name = h.helicsFederateGetName(fed)
    logger.info("HELICS Version: {}".format(h.helicsGetVersion()))
    logger.info("{}: Federate {} has been registered".format(federate_name, federate_name))
    endpoint_count = h.helicsFederateGetEndpointCount(fed)
    subkeys_count = h.helicsFederateGetInputCount(fed)
    ######################   Reference to Publications and Subscription form index  #############################
    endid = {}
    subid = {}
    for i in range(0, endpoint_count):
        endid["m{}".format(i)] = h.helicsFederateGetEndpointByIndex(fed, i)
        end_name = h.helicsEndpointGetName(endid["m{}".format(i)])
        logger.info("{}: Registered Endpoint ---> {}".format(federate_name, end_name))

    for i in range(0, subkeys_count):
        subid["m{}".format(i)] = h.helicsFederateGetInputByIndex(fed, i)
        status = h.helicsInputSetDefaultComplex(subid["m{}".format(i)], 0, 0)
        sub_key = h.helicsInputGetTarget(subid["m{}".format(i)])
        logger.info("{}: Registered Subscription ---> {}".format(federate_name, sub_key))

    ######################   Entering Execution Mode  ##########################################################
    h.helicsFederateEnterExecutingMode(fed)

    plotting = False ## Adjust this flag to visulaize the control actions aas the simulation progresses
    hours = 96
    val_switchHour=72
    total_inteval = int(60 * 60 * hours)
    val_interval= int(60*60*val_switchHour)
    grantedtime = -1
    update_interval = 60*5 ## Adjust this to change EV update interval
    feeder_limit_upper = 4 * (1000 * 1000) ## Adjust this to change upper limit to trigger EVs
    feeder_limit_lower = 2.7 * (1000 * 1000) ## Adjust this to change lower limit to trigger EVs
    k = 0
    EV_data = {}
    time_sim = []
    feeder_real_power = []
    LoadInfo=True
    UncSeek=False
    Persistant=False
    olooplen=5
    evaluate=True
    #uncseek using Q-Learning, where Q is the error of the predictive model
    LossExpectations=None
    if plotting:
        ax ={}
        fig = plt.figure()
        #fig.subplots_adjust(hspace=0.4, wspace=0.4)
        #ax['Feeder'] = plt.subplot(313)
        #ax['EV1'] = plt.subplot(331)
       # ax['EV2'] = plt.subplot(332)
       # ax['EV3'] = plt.subplot(333)
       # ax['EV4'] = plt.subplot(334)
       # ax['EV5'] = plt.subplot(335)
       # ax['EV6'] = plt.subplot(336)
        ax['Loss']=plt.subplot(111)

    voltageTimeSeries=None
    commandTimeSeries=pt.ones([1,endpoint_count])
    commandCurrentVals=pt.ones([endpoint_count],dtype=pt.int32)
    learning_rate = .0001
    if(UncSeek==False):
        loss=nn.MSELoss()
    else:
        loss=nn.MSELoss(reduction='none')
    epoch=-1
    model=None
    optimizer=None
    #to select a random subset of meters
    metercount=None
    meter_keys=None
    phase_keys=None
    predictions=None
    meancorrection=None
    variancecorrection=1
    d_latent=2048
    numblocks=8
    nheads=8
    meanmodel={}
    for t in range(0, total_inteval, update_interval):
        epoch+=1
        while grantedtime < t:
            grantedtime = h.helicsFederateRequestTime(fed, t)

        time_sim.append(t / 3600)
        ############################### Subscribing to streamed data from to GridLAB-D ###################################
        for i in range(0, subkeys_count):
            sub = subid["m{}".format(i)]
            inputvolts = h.helicsInputGetString(sub)
            for i in range(0, endpoint_count):
                end_point = endid["m{}".format(i)]
            ####################### Clearing all pending messages and stroing the most recent one ######################
            """ Note: In case GridLAB-D and EV Controller are running in different intervals 
                        there might be pending messages which gets stored in the endpoint buffer  """
            while h.helicsEndpointHasMessage(end_point):
                end_point_msg_obj = h.helicsEndpointGetMessage(end_point)

            #logger.info("inputvolts\n")
            #logger.info(inputvolts)
        
        if(t==0):
            #initialize ML model
            # 1) parse JSON → dict
            data = json.loads(inputvolts)
        

 # 2) fix ordering of meters & phases
            meter_keys = sorted(data.keys())
            if (metercount!=None):
                pt.randperm(len(meter_keys))
                meter_keys=meter_keys[0:metercount]
                meter_keys=sorted(meter_keys)
            phase_keys = [
            "measured_voltage_1",
            "measured_voltage_2",
            "measured_voltage_N",
            ]

# 3) build a list: [ [ real, imag, … ], … ]
            vals = []
            for m in meter_keys:
                for ph in (data[m]):
                    c = parse_voltage(data[m][ph])
                    vals.append(c.real)
                    vals.append(c.imag)
            if(LoadInfo==True):
                for i in range(endpoint_count):
                    vals.append(commandCurrentVals[i]) #commands are floats, not complex, but this makes tensor regular
            t_real_imag = pt.tensor(vals, dtype=pt.float64)
            voltageTimeSeries=t_real_imag.unsqueeze(0)
            meancorrection=pt.mean(voltageTimeSeries)
            logger.info(f"Mean:{meancorrection}")
            helvariancecorrection=pt.std(voltageTimeSeries).item()
            logger.info(F"Variance:{variancecorrection}")
            voltageTimeSeries=(voltageTimeSeries-meancorrection)/variancecorrection
            width=voltageTimeSeries.size(1)
            #logger.info(voltageTimeSeries.size())
            if(LoadInfo==False):
                model=TsTransformer(d_input=width,d_output=width,d_latent=2048,numblocks=6,nheads=8,persistance=Persistant)
            #     #model=LSTMGS(d_input=width,d_output=width,d_latent=2048,numblocks=10)
            elif(UncSeek==True):
                model=TsTransformer(d_input=width,d_output=(width-endpoint_count),d_latent=1,numblocks=1,nheads=1,persistance=Persistant)
                LossExpectations=pt.ones(2**endpoint_count)
                logger.info(LossExpectations.shape)
                LossExpectations=LossExpectations*100#higher than plausible
            else:
                #model=ResidualMLP(d_input=width,d_output=(width-endpoint_count),d_latent=2048,numblocks=8,persistance=Persistant)
                model=TsTransformer(d_input=width,d_output=(width-endpoint_count),d_latent=d_latent,numblocks=numblocks,nheads=nheads,persistance=Persistant)
            optimizer = pt.optim.SGD(model.parameters(True), lr=learning_rate)
            model.train()
    
        if(t>0):
            optimizer.zero_grad()
            data = json.loads(inputvolts)# 3) build a nested list: [ [ [real, imag], … ], … ]
            vals = []
            for m in meter_keys:
                for ph in data[m]:
                    c = parse_voltage(data[m][ph])
                    vals.append(c.real)
                    vals.append(c.imag)
            commandCurrentVals=pt.randint(low=0, high=2,size=commandCurrentVals.shape)
            for i in range(1,endpoint_count+1):
                if(commandCurrentVals[-i]!=commandTimeSeries[-1][-i]):
                    end = endid["m{}".format(i-1)]
                    source_end_name = str(h.helicsEndpointGetName(end))
                    dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
                    msg = h.helicsFederateCreateMessage(fed)
                    end = endid["m{}".format(k)]
                    source_end_name = str(h.helicsEndpointGetName(end))
                    if(commandCurrentVals[-i]==0):
                        h.helicsMessageSetString(msg, '0+0j')
                        status = h.helicsEndpointSendMessage(end, msg)
                    else:
                        status = h.helicsEndpointSendBytes(end, '200000+0.0j')
                    
            commandTimeSeries=pt.cat((commandTimeSeries,commandCurrentVals.unsqueeze(0)))
            for i in range(endpoint_count):
                vals.append(commandCurrentVals[i])
            t_real_imag = (pt.tensor(vals, dtype=pt.float32)-meancorrection)/variancecorrection
            #logger.info(t_real_imag)
            voltageTimeSeries=pt.cat((voltageTimeSeries,t_real_imag.unsqueeze(0)))
            #logger.info(voltageTimeSeries)
            # if(LoadInfo==True):
            #     logger.info(voltageTimeSeries.shape)
            #     logger.info(loss(voltageTimeSeries[-1:],voltageTimeSeries[1:]))
            #     logger.info(loss(voltageTimeSeries[0],voltageTimeSeries[-1]))
            #     error=None
            #     voltage_changes = None
            if(LoadInfo):
                voltage_changes=pt.abs(voltageTimeSeries[1:,:-endpoint_count] - voltageTimeSeries[:-1,:-endpoint_count])
            else:
                voltage_changes=pt.abs(voltageTimeSeries[1:,:-endpoint_count] - voltageTimeSeries[:-1,:-endpoint_count])
            responses = voltageTimeSeries[-1, :-6]  # [T, N]
            controls = voltageTimeSeries[-2, -6:]   # [T, E]
            # Get controls from previous timestep (t-1)
            
            if tuple(controls.tolist()) in meanmodel:
                logger.info(f"next state conditional mean error:{loss(meanmodel[tuple(controls.tolist())][1],responses)}")
                numocc=meanmodel[tuple(controls.tolist())][0]
                meanmodel[tuple(controls.tolist())][1]=(meanmodel[tuple(controls.tolist())][1]*numocc+responses)/(numocc+1)
                meanmodel[tuple(controls.tolist())][0]=meanmodel[tuple(controls.tolist())][0]+1
            else:
                #conditional mean if unobserved is just regular mean
                logger.info(f"next state conditional mean error:{loss(pt.mean(voltageTimeSeries[:-1,:-6],dim=0),responses)}")
                logger.info("New Command")
                meanmodel[tuple(controls.tolist())]=[1,responses]
            
            logger.info(f"Mean voltage change: {voltage_changes.mean()}")
            logger.info(f"Std voltage change: {voltage_changes.std()}")
            logger.info(f"Max voltage change: {voltage_changes.max()}")
            logger.info(f"MS Voltage change: {pt.mean(loss(voltageTimeSeries[1:,:-endpoint_count], voltageTimeSeries[:-1,:-endpoint_count]))}")
            logger.info(f"Next step persistance loss:{pt.mean(loss(voltageTimeSeries[-1,:-endpoint_count],voltageTimeSeries[-2,:-endpoint_count]))}")
            
            # if(pt.mean(loss(voltageTimeSeries[-1,:-endpoint_count],voltageTimeSeries[-2,:-endpoint_count]))>.05):
            #     logger.info(voltageTimeSeries[-1,:-endpoint_count]-voltageTimeSeries[-2,:-endpoint_count])
            logger.info(f"Next step mean change{pt.mean(pt.abs(voltageTimeSeries[-1,:-endpoint_count]-voltageTimeSeries[-2:,:-endpoint_count]))}")
            logger.info(f"Next step max change{pt.max(pt.abs(voltageTimeSeries[-1,:-endpoint_count]-voltageTimeSeries[-2,:-endpoint_count]))}")
        #     # 2) fix ordering of meters & phases
            if(LoadInfo==False and t<val_interval):#valdiationsplit
                predictions=model.forward(voltageTimeSeries[:-1])
                nextrounderror=loss(voltageTimeSeries[-1],predictions[-1])
                logger.info(f"Next Round Error{nextrounderror}")
                if(nextrounderror>100):
                    model=TsTransformer(d_input=width,d_output=(width-endpoint_count),d_latent=d_latent,numblocks=numblocks,nheads=nheads,persistance=Persistant)

                for i in range(olooplen):
                    predictions=model.forward(voltageTimeSeries[:-1])#normalize model to prevent gradient blowup
                    error=loss(predictions,voltageTimeSeries[1:])
                    logger.info(f"Oloop error {i}:{error}")
                    logger.info(predictions)
                    error.backward()
                    optimizer.step()
                    optimizer.zero_grad() 

        #         #infer                #
        #         #logger.info(voltageTimeSeries[1:]-predictions)
        #         #logger.info(pt.argmax(voltageTimeSeries[1:]-predictions))
        #         #logger.info(pt.argmin(voltageTimeSeries[1:]-predictions))

            if(LoadInfo==True and UncSeek==False and t<val_interval):#valdiationsplit
                predictions=model.forward(voltageTimeSeries[:-1])
                logger.info(f"Next Round Error{loss(voltageTimeSeries[-1,:-endpoint_count],predictions[-1])}")
                for i in range(olooplen):
                    predictions=model.forward(voltageTimeSeries[:-1])#normalize model to prevent gradient blowup
        #             #logger.info(voltageTimeSeries[1:,:-endpoint_count].shape)
                    error=loss(predictions,voltageTimeSeries[1:,:-endpoint_count])
                    logger.info(f"Oloop error {i}:{error}")
        #             #logger.info(predictions)
                    error.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        #             commandCurrentVals=pt.randint(high=2,size=commandCurrentVals.shape)
        #             for i in range(1,endpoint_count):h.helicsEndpointGetName(end))
        #                     dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
        #                     msg = h.helicsFederateCreateMessage(fed)

        #                     h.helicsMessageSetString(msg, str(complex(commandCurrentVals[-i])*200000))
        #                     status = h.helicsEndpointSendMessage(end, msg)
        #             #logger.info(commandCurrentVals)
        #             voltageTimeSeries[:-1,-i:]=0
        #         # offeffect=model.forward(voltageTimeSeries)[-1]
        #         # voltageTimeSeries[:-1,-i:]=1
        #         # oneffect=model.forward(voltageTimeSeries)[-1]
        #         # logger.info(offeffect-oneffect)
        
        #         #infer                #
        #         #logger.info(voltageTimeSeries[1:]-predictions)
        #         #logger.info(pt.argmax(voltageTimeSeries[1:]-predictions))
        #         #logger.info(pt.argmin(voltageTimeSeries[1:]-predictions))
            
        #     if(UncSeek==True and t<val_interval):
        #         predictions=model.forward(voltageTimeSeries[:-1])
        #         error=pt.mean(input=(loss(predictions,voltageTimeSeries[1:,:-endpoint_count])),dim=1)
        #         logger.info(f"Next Round Error:{error[-1]}")
        #         prevcommand=0
        #         for i in range(endpoint_count):
        #             prevcommand=prevcommand+commandCurrentVals[i]*(2**i)
        #         LossExpectations[prevcommand.item()]=(error[-1])
        #         for i in range(olooplen):
        #             predictions=model.forward(voltageTimeSeries[:-1])#normalize model to prevent gradient blowup
        #             error=pt.mean(input=(loss(predictions[:,:],voltageTimeSeries[1:,:-endpoint_count])))
        #             logger.info(f"Oloop error {i}:{error}")
        #             error.backward()
        #             #logger.info(error[-1])
        #             #logger.info(predictions)
        #             optimizer.step()
        #             optimizer.zero_grad()
        #         command=int(pt.argmax(LossExpectations))
        #         for i in range(endpoint_count):
        #             if(command>>i)&1==0:
        #                 commandCurrentVals[i]=0
        #                 if(voltageTimeSeries[-1][width-endpoint_count+i]!=0):
        #                     voltageTimeSeries[-1][width-endpoint_count+i]=0
        #                     end = endid["m{}".format(i)]
        #                     source_end_name = str(h.helicsEndpointGetName(end))
        #                     dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
        #                     msg = h.helicsFederateCreateMessage(fed)
        #                     h.helicsMessageSetString(msg, '0+0j')
        #                     status = h.helicsEndpointSendMessage(end, msg)
        #             else:
        #                 commandCurrentVals[i]=1
        #                 if(voltageTimeSeries[-1][width-endpoint_count+i]!=1):
        #                     voltageTimeSeries[-1][width-endpoint_count+i]=1
        #                     end = endid["m{}".format(i)]
        #                     source_end_name = str(h.helicsEndpointGetName(end))
        #                     dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
        #                     msg = h.helicsFederateCreateMessage(fed)
        #                     h.helicsMessageSetString(msg, '200000+0j')
        #                     status = h.helicsEndpointSendMessage(end, msg)
        #         commandTimeSeries=pt.cat((commandTimeSeries,commandCurrentVals.unsqueeze(0)))
        #         #logger.info(commandCurrentVals)
        #         #infer
        #         # if(LoadInfo==True):
        #         #     voltagetemp=voltageTimeSeries[:-1,-i:].clone()
        #         #     voltageTimeSeries[:-1,-i:]=0
        #         #     offeffect=model.forward(voltageTimeSeries)[-1]
        #         #     voltageTimeSeries[:-1,-i:]=1
        #         #     oneffect=model.forward(voltageTimeSeries)[-1]
        #         #     logger.info(offeffect-oneffect)
        #         #     voltageTimeSeries[:-1,-i:]=voltagetemp
        #     validation_error=0
            if(t>val_interval and evaluate==False):
                if(LoadInfo and UncSeek):
                    predictions=model.forward(voltageTimeSeries[:-1])
                    error=pt.mean(input=(loss(predictions,voltageTimeSeries[1:,:-endpoint_count])))
                    validation_error=validation_error+error
                elif(LoadInfo and not UncSeek):
                    predictions=model.forward(voltageTimeSeries[:-1])
                    error=pt.mean(input=(loss(predictions,voltageTimeSeries[1:,:-endpoint_count])))
                    validation_error=validation_error+error
                else:
                    predictions=model.forward(voltageTimeSeries[:-1])
                    error=loss(predictions,voltageTimeSeries[1:])
                    validation_error=validation_error+error
                logger.info("Validation error:{}".format(validation_error))
            
            logger.info(commandCurrentVals)

        #     if plotting:
        #     #  ax['Feeder'].clear()
        #     #  ax['Feeder'].plot(time_sim, feeder_real_power)
        #     #  ax['Feeder'].plot(np.linspace(0,24,25), feeder_limit_upper*np.ones(25), 'r--')
        #     #  ax['Feeder'].plot(np.linspace(0,24,25), feeder_limit_lower*np.ones(25), 'g--')
        #     #  ax['Feeder'].set_ylabel("Feeder Load (kW)")
        #     #  ax['Feeder'].set_xlabel("Time (Hrs)")
        #     #  ax['Feeder'].set_xlim([0, 24])
        #     #  ax['Feeder'].grid()
        #     #  for keys in EV_data:
        #     #      ax[keys].clear()
        #     #      ax[keys].plot(time_sim, EV_data[keys])
        #     #      ax[keys].set_ylabel("EV Output (kW)")
        #     #      ax[keys].set_xlabel("Time (Hrs)")
        #     #      ax[keys].set_title(keys)
        #     #      ax[keys].set_xlim([0, 24])
        #     #      ax[keys].grid()
        #     #  plt.show(block=False)
        #     #  plt.pause(0.01)
        #         ax['Loss'].clear()
        #         ax['Loss'].plot(epoch,error.item())
        #         ax['Loss'].set_ylabel("Loss")
        #         ax['Loss'].set_xlabel("epoch")
        #         plt.show(block=False)
        #         plt.pause(0.01)
        #         if t == (total_inteval - update_interval):
        #             plt.tight_layout()
        #             plt.savefig(f"./output/{case_num}_EV_plot.png", dpi=200)
        # if(t>600 and t<3000):
        #     voltageTimeSeries=voltageTimeSeries[1:]
        #     meancorrection=(voltageTimeSeries[-1]+meancorrection)*variancecorrection

        #for i in range(0, endpoint_count):
         #   end_point = endid["m{}".format(i)]
         #   ####################### Clearing all pending messages and stroing the most recent one ######################
         #   """ Note: In case GridLAB-D and EV Controller are running in different intervals 
         #               there might be pending messages which gets stored in the endpoint buffer  """
         #   while h.helicsEndpointHasMessage(end_point):
         #       end_point_msg_obj = h.helicsEndpointGetMessage(end_point)
         #       # logger.info("removing pending messages")

##           EV_name = end_point.name.split('/')[-1]
 #           if EV_name not in EV_data:
  #                  EV_data[EV_name] = []
   #         EV_data[EV_name].append(EV_now.real / 1000)

        logger.info("{}: Federate Granted Time = {}".format(federate_name, grantedtime))

        # if feeder_real_power[-1] > feeder_limit_upper:
        #     logger.info("{}: Warning !!!! Feeder OverLimit ---> Total Feeder Load is over the Feeder Upper Limit".format(federate_name))

        #     if k < endpoint_count:
        #         end = endid["m{}".format(k)]
        #         logger.info("endid: {}".format(endid))
        #         source_end_name = str(h.helicsEndpointGetName(end))
        #         dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
        #         logger.info("{}: source endpoint {} and destination endpoint {}".format(federate_name, source_end_name, dest_end_name))
        #         msg = h.helicsFederateCreateMessage(fed)
        #         h.helicsMessageSetString(msg, '0+0j')
        #         status = h.helicsEndpointSendMessage(end, msg)
        #         logger.info("{}: Turning off {}".format(federate_name, source_end_name))
        #         k = k + 1
        #     else:
        #         logger.info("{}: All EVs are turned off")

        # if feeder_real_power[-1] < feeder_limit_lower:
        #     logger.info("{}: Safe !!!! Feeder Can Support EVs --> Total Feeder Load is under the Feeder Lower Limit".format(federate_name))
        #     if k > 0:
        #         k = k - 1
        #         end = endid["m{}".format(k)]
        #         source_end_name = str(h.helicsEndpointGetName(end))
        #         dest_end_name   = str(h.helicsEndpointGetDefaultDestination(end))
        #         logger.info("{}: source endpoint {} and destination endpoint {}".format(federate_name, source_end_name, dest_end_name))
        #         status = h.helicsEndpointSendBytes(end, '200000+0.0j')
        #         logger.info("{}: Turning on {}".format(federate_name, source_end_name))
        #     else:
        #         logger.info("{}: All EVs are turned on".format(federate_name))

        

    # EV_data["time"] = time_sim
    # EV_data["feeder_load"] = feeder_real_power
    # pd.DataFrame.from_dict(data=EV_data).to_csv(f"{case_num}_EV_Outputs.csv", header=True)
    # def jacobianLimit(x):
    #     # Function that returns only the output slice we care about
    #                 full_output = model(x)
    #                 return full_output[-1]
    pt.save(model,"model.pt")
    pt.save(voltageTimeSeries, "RandomTimeSeries.pt")
    if(LoadInfo):
        jacobian=pt.func.jacrev(jacobianLimit)(voltageTimeSeries)
        pt.save(model,"jacobian.pt")
    t = 60 * 60 * 24
    # while grantedtime < t:
    #     grantedtime = h.helicsFederateRequestTime(fed, t)
    
    logger.info("{}: Destroying federate".format(federate_name))
    destroy_federate(fed)
