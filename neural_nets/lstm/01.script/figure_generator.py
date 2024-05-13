import matplotlib.pyplot as plt
# %% Function

def EvalPlot(dictionary, supply=False):
    model = 'lstm'
    cfsday_AFday = 1.983
    cols = ['flow_cfs', 'NWM_flow', f"lstm_flow"]

    # make figure
    fig, ax = plt.subplots(dpi=300)

    RegionDF = dictionary.copy()
    RegionDF = RegionDF[cols]

    if supply == True:
        RegionDF = RegionDF * cfsday_AFday / 1000
        units = 'kAF'
        # set up cumulative monthly values
        RegionDF['Year'] = RegionDF.index.year

        for site in cols:
            RegionDF[site] = RegionDF.groupby(['Year'])[site].cumsum()
        # Plotting the data with labels for the legend
        ax.plot(RegionDF.index, RegionDF['flow_cfs'], color='green', label='Obs Flow ')
        ax.plot(RegionDF.index, RegionDF[f"{model}_flow"], color='orange', label=f"{model} flow")
        ax.plot(RegionDF.index, RegionDF['NWM_flow'], color='blue', label=f"NWM flow")
        ax.set_xlabel('Time (day)')
        ax.set_ylabel('Streamflow (kAF)')


    else: 


        max_value = max(RegionDF['flow_cfs'].max(), max(RegionDF[f"{model}_flow"].max(), RegionDF['NWM_flow'].max()))
        min_value = min(RegionDF['flow_cfs'].min(), min(RegionDF[f"{model}_flow"].min(), RegionDF['NWM_flow'].min()))
        ax.plot([min_value, max_value], [min_value, max_value], color='green', linestyle='--')
        
        ax.scatter(RegionDF['flow_cfs'], RegionDF[f"{model}_flow"], color='orange', label=f"{model} flow", s=10)
        ax.scatter(RegionDF['flow_cfs'], RegionDF['NWM_flow'], color='blue', label=f"NWM flow", s=10)
        ax.set_xlabel('Observation (cfs)')
        ax.set_ylabel('Model (cfs)')
    


    # Adding a title to each subplot
    ax.set_title('NWM vs LSTM')

    ax.tick_params(axis='x', rotation=45)
    plt.legend()

    plt.tight_layout()
    # plt.savefig(f'final.png', dpi=600)
    plt.show()

