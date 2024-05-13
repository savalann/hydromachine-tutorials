# This file created on 01/14/2024 by savalan

# Import packages ==============================
# My Packages
from g_evaluation_metric import MAPE, RMSE, KGE, PBias
import pandas as pd
import numpy as np

# Functions ==============================

def evtab(Eval_DF_mine, prediction_columns, nhdreach, observation_column, mod):

    #get annual supply diffs
    cfsday_AFday = 1.983
    
    #Get RMSE from the model
    rmse = RMSE(Eval_DF_mine, prediction_columns, observation_column)

    #Get Mean Absolute Percentage Error from the model
    mape = MAPE(Eval_DF_mine, prediction_columns, observation_column)

    #Get Percent Bias from the model
    pbias = PBias(Eval_DF_mine, prediction_columns, observation_column)

    #Get Kling-Gutz Efficiency from the model
    kge = KGE(Eval_DF_mine, prediction_columns, observation_column)
    
    #Get Volumetric values
    Eval_DF_mine.set_index('datetime', inplace = True, drop =True)
    flowcols = [f"{mod}_flow", 'flow_cfs', 'NWM_flow']
    SupplyEval = Eval_DF_mine[flowcols].copy()
    SupplyEval = SupplyEval*cfsday_AFday
    #set up cumulative monthly values
    SupplyEval['Year'] = SupplyEval.index.year

    for col_name in flowcols:
        SupplyEval[col_name] = SupplyEval.groupby(['Year'])[col_name].cumsum()  

    EOY_mod_vol_af = SupplyEval[f"{mod}_flow"].iloc[-1]
    EOY_obs_vol_af = SupplyEval["flow_cfs"].iloc[-1]
    EOY_nwm_vol_af = SupplyEval[f"NWM_flow"].iloc[-1]
    NWM_vol_diff_af = EOY_nwm_vol_af - EOY_obs_vol_af
    Mod_vol_diff_af = EOY_mod_vol_af - EOY_obs_vol_af
    NWM_Perc_diff = (NWM_vol_diff_af/EOY_obs_vol_af)*100
    Mod_Perc_diff = (Mod_vol_diff_af/EOY_obs_vol_af)*100
    
     #Get Performance Metrics from the model
    Srmse = RMSE(SupplyEval, prediction_columns, observation_column)
    Smape = MAPE(SupplyEval, prediction_columns, observation_column)
    Spbias = PBias(SupplyEval, prediction_columns, observation_column)
    Skge = KGE(SupplyEval, prediction_columns, observation_column)
    
    
    # #save model performance
    # sitestats = [Eval_DF_mine.iloc[0, 1], nhdreach, rmse[0], rmse[1],  pbias[0], pbias[1], kge[0], kge[1], mape[0],mape[1]]

    
    # Supplystats = [Eval_DF_mine.iloc[0, 1], nhdreach, Srmse[0], Srmse[1],  Spbias[0], Spbias[1], Skge[0], Skge[1], Smape[0],  
    #              Smape[1],EOY_obs_vol_af, EOY_nwm_vol_af,EOY_mod_vol_af,NWM_vol_diff_af,Mod_vol_diff_af, NWM_Perc_diff, Mod_Perc_diff ]

    sitestats = [Eval_DF_mine['station_id'].drop_duplicates()[0], nhdreach, rmse[0], rmse[1],  pbias[0], pbias[1], kge[0], kge[1], mape[0],mape[1]]
                
    
    Supplystats = [Eval_DF_mine['station_id'].drop_duplicates()[0], nhdreach, Srmse[0], Srmse[1],  Spbias[0], Spbias[1], Skge[0], Skge[1], Smape[0], Smape[1],EOY_obs_vol_af, EOY_nwm_vol_af,EOY_mod_vol_af,NWM_vol_diff_af,Mod_vol_diff_af, NWM_Perc_diff, Mod_Perc_diff ]

    return sitestats, Supplystats


def EvalTable(yhat_test, data_test, lookback):

    df_eval = data_test.iloc[lookback:, :]
    df_eval['lstm_flow'] = yhat_test
    prediction_columns = ['NWM_flow', f"lstm_flow"]
    observation_column = 'flow_cfs'
    result_daily, result_cumulative = evtab(df_eval, prediction_columns, '10375648', observation_column, 'lstm')
    
    model_name = 'LSTM'
    #Evaluation columns for prediction time series
    cols = ['USGSid', 'NHDPlusid', 'NWM_RMSE', f"{model_name}_RMSE", 'NWM_PBias', f"{model_name}_PBias", 
            'NWM_KGE', f"{model_name}__KGE", 'NWM_MAPE',  f"{model_name}_MAPE"]
    
    #Evaluation columns for accumulated supply time series
    supcols = ['USGSid', 'NHDPlusid', 'NWM_RMSE', f"{model_name}_RMSE", 'NWM_PBias', f"{model_name}_PBias", 
            'NWM_KGE', f"{model_name}__KGE", 'NWM_MAPE',  f"{model_name}_MAPE", 'Obs_vol', 'NWM_vol', f"{model_name}_vol",
            'NWM_vol_err', f"{model_name}_vol_err", 'NWM_vol_Perc_diff', f"{model_name}_vol_Perc_diff"]
        
    #save model results
    EvalDF_all = pd.DataFrame(np.array(result_daily).reshape(1, -1), columns=cols)
    SupplyEvalDF_all = pd.DataFrame(np.array(result_cumulative).reshape(1, -1), columns=supcols)
    EvalDF_all.iloc[:, 2:] = EvalDF_all.iloc[:, 2:].astype(float).round(2)
    SupplyEvalDF_all.iloc[:, 2:] = SupplyEvalDF_all.iloc[:, 2:].astype(float).round(2)
    # print("Model Performance for Daily cfs")
    # display(EvalDF_all)   
    # print("Model Performance for Daily Accumulated Supply (Acre-Feet)")
    # display(SupplyEvalDF_all)

    return EvalDF_all, SupplyEvalDF_all, df_eval