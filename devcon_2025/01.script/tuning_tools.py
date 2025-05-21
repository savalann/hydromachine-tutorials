import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from model import CustomBiLSTM


def tune_model(input_size, device, train_dataset, epochs, params, selected_station):

    
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    train_loader = {selected_station[0]: DataLoader(train_dataset, batch_size=batch_size, shuffle=False)}
    
    # Create the Model
    bilstm_model = CustomBiLSTM(input_size, hidden_size, num_layers, 1, device, embedding=False, station_list=selected_station)
    
    # Create the Optimizer
    bilstm_optimizer = optim.Adam(bilstm_model.parameters(), lr=learning_rate, weight_decay=0)
    
    # Run the training function
    model_parameters = bilstm_model.train_model(train_loader, epochs, bilstm_optimizer, early_stopping_patience=0, val_loader=None, tune='True')
    
    _, val_loss = bilstm_model.evaluate_model(train_loader[selected_station[0]])

    return val_loss, bilstm_model


def tuning_game(input_size, device, train_dataset, epochs, params, selected_station):
    
    # Evaluate the model with initial parameters and calculate the mean of absolute scores.
    current_score, bilstm_model = tune_model(input_size, device, train_dataset, epochs, params, [selected_station])
    print(f"Initial score: {current_score} with params: {params}")
    
    # Initialize the interactive tuning loop.
    continue_tuning = True
    while continue_tuning:
        # Prompt the user if they want to continue tuning.
        change = input("Do you want to change any variable? (y/n): ")
        if change.lower() == 'y':
            # Ask which parameter to change.
            variable = input("Which variable number? (batch_size(1)/learning_rate(2)/hidden_size(3)/num_layers(4)):")
            # Map user input to the corresponding parameter.
            if variable == '1':
                variable = 'batch_size'
            elif variable == '2':
                variable = 'learning_rate'
            elif variable == '3':
                variable = 'hidden_size'
            elif variable == '4':
                variable = 'num_layers'
            else:
                print('Error: Wrong Number')
                break
    
            # Prompt for the new value and validate the type.
            value = input(f"Enter the new value for {variable} (previous value {params[variable]}): ")
            if variable == 'batch_size' or variable == 'num_layers' or variable == 'hidden_size' :
                value = int(value)
            else:
                value = float(value)
    
            # Update parameter and re-evaluate the model.
            old_param = params[variable]
            params[variable] = value
            new_score, bilstm_model = tune_model(input_size, device, train_dataset, epochs, params, [selected_station])
            print('Previous Mean Score: %.3f' % (current_score))
            print('New Mean Score: %.3f ' % (new_score))
            current_score = new_score
    
            # Prompt if the new parameter setting should be kept.
            keep_answer = input(f"Do you want to keep the new variable?(y/n): ")
            if keep_answer == 'n':
                params[variable] = old_param
        else:
            # Exit tuning loop.
            continue_tuning = False
            bilstm_model.save_model("bilstm_weights.pth")
            print("Finished tuning.")
            print(f"Final parameters: {params}.")
