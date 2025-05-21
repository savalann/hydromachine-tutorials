# This file created on 01/13/2024 by savalan

# Import packages ==============================
# Base Packages
import numpy as np

# main packages
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he
# Functions ==============================

#class hem(obs, pred, metric):  # evaluation metrics 

def RMSE(DF, predictions, observation):
    R = []
    for pred in np.arange(0, len(predictions),1):
        rmse = mean_squared_error(DF[observation], DF[predictions[pred]], squared=False)
        R.append(rmse)
        #print('RMSE for ', predictions[pred], ' is ', rmse, ' cfs')
    return R

def MAPE(DF, predictions, observation):
    P =[]
    for pred in np.arange(0, len(predictions),1):
        mape = round(mean_absolute_percentage_error(DF[observation], DF[predictions[pred]])*100, 2)
        P.append(mape)
        #print('Mean Absolute Percentage Error for ', predictions[pred], ' is ', mape, '%')
    return P

def PBias(DF, predictions, observation):
    PB = []
    for pred in np.arange(0, len(predictions),1):
        pbias = he.evaluator(he.pbias,  DF[predictions[pred]], DF[observation])
        pbias = round(pbias[0],2)
        PB.append(pbias)
        #print('Percentage Bias for ', predictions[pred], ' is ', pbias, '%')
    return PB    
  
def KGE(DF, predictions, observation):
    KG = []
    for pred in np.arange(0, len(predictions),1):
        kge, r, alpha, beta = he.evaluator(he.kge,  DF[predictions[pred]], DF[observation])
        kge = round(kge[0],2)
        KG.append(kge)
        #print('Kling-Glutz Efficiency for ', predictions[pred], ' is ', kge)
    return KG

# if 'r2' in metric:
#     r2 = pow(np.corrcoef(obs, pred)[0, 1], 2) 
#     all_columns.append('r2')
#     all_result.append(r2)
# if 'mae' in metric:
#     mae = np.mean(np.abs(res))
#     all_columns.append('mae')
#     all_result.append(mae)
# if 'mse' in metric:
#     mse = np.mean(np.square(res))
#     all_columns.append('mse')
#     all_result.append(mse)
# if 'rmse' in metric:
#     rmse = np.sqrt(mse)
#     all_columns.append('rmse')
#     all_result.append(rmse)
# if 'nse' in metric:
#     nse = 1 - ((pow((obs - pred), 2)).sum()) / ((pow((obs - obs.mean()), 2)).sum())
#     all_columns.append('nse')
#     all_result.append(nse)

