# This file created on 01/13/2024 by savalan

# Import packages ==============================
# main packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Functions ==============================

class CustomBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=None, embedding=False, station_list=None):
        super(CustomBiLSTM, self).__init__()

        self.station_list = station_list
        self.embedding = embedding
        self.embedding_size = 20

        if self.embedding == True:
            input_size = 12 + self.embedding_size

        # --- Static embedding layer ---
        self.static_embedding = nn.Sequential(
            nn.Linear(6, self.embedding_size),
            nn.Tanh(),  # You can expand this to include sigmoid if needed
            nn.Dropout(0.0)
        )


        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Output size is doubled for bidirectional LSTM

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.loss_function = nn.L1Loss()
        self.device = device
        self.to(self.device)
        self.validation_indicator = 0

    def forward(self, x):

        if self.embedding == True:
            x_static = x[:, :, :6]
            x_dynamic = x[:, :, 6:]
            x_static = x_static[:, 0, :]

            # Embed static features
            x_static_emb = self.static_embedding(x_static)  # [batch, static_embedding_dim]
            x_static_emb = x_static_emb.unsqueeze(1).repeat(1, x_dynamic.size(1), 1)  # [batch, seq_len, static_embedding_dim]

            # Concatenate along feature dimension
            x = torch.cat([x_dynamic, x_static_emb], dim=-1)  # [batch, seq_len, input_size + static_embedding_dim]

        # Initialize hidden state and cell state for both directions
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection

        # Forward propagate LSTM
        # print(x.shape)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, train_loader, epochs, optimizer, early_stopping_patience=0, val_loader=None, tune='False'):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            shuffled_list = self.station_list.copy()
            random.shuffle(shuffled_list)
            for station_id in shuffled_list:
                train_loader_selected = train_loader[station_id]
                
                for inputs, targets in train_loader_selected:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
    
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = self.loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()
    
                
                val_loss = 0
                best_model_parameters = self.state_dict()
                if val_loader is not None:
                    self.validation_indicator = 1
                    val_loss = self.evaluate_model(val_loader)[1]
    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_parameters = self.state_dict()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
    
                    if epochs_no_improve == early_stopping_patience and early_stopping_patience > 0:
                        print('Early stopping triggered')
                        break
            if tune == 'False':
                print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}', f'Validation Loss: {val_loss}')
            self.validation_indicator = 0
        return best_model_parameters
        print('Training is done!')

    def evaluate_model(self, data_loader):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
        avg_loss = total_loss / total
        # if self.validation_indicator == 0:
        #     # print(f'Validation Loss: {avg_loss}')
        return outputs, avg_loss
        #outputs if self.validation_indicator == 0 else avg_loss

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location=self.device))
# early stop and checkpoint shoudl be different functions. 



