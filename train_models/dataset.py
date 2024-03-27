import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np



class PV_Dataset(Dataset):
    def __init__(self, data, input_steps=6, output_steps=1, step=1, feature_columns=['value']):
        """
        Args:
            data (DataFrame): DataFrame containing the time series data.
            input_steps (int): Number of past time steps used for forecasting.
            output_steps (int): Number of future time steps to forecast.
            step (int): Step interval between the last input step and the first output step.
            feature_columns (list of str): List of column names to be used as features. 
        """
        self.dataframe = data
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.step = step  # Add step as an attribute
        self.feature_columns = feature_columns if feature_columns is not None else self.dataframe.columns.drop(['date', 'time_of_day', 'time', 'daylight_level'])
        self.samples = []

        # Preprocess the data and create samples
        self._create_samples()
    
    def _create_samples(self):
        # Group by date to ensure all slices are from the same day
        grouped = self.dataframe.groupby('date')
        for _, group in grouped:
            # Convert the specified features to a numpy array
            feature_data = group[self.feature_columns].to_numpy()
            # Adjust the loop to account for the step interval
            for i in range(len(group) - self.input_steps - self.output_steps - self.step + 2):  # Adjusted for the step
                input_indices = range(i, i + self.input_steps)
                # Adjust the starting index for output_data based on step
                output_indices = range(i + self.input_steps + self.step - 1, i + self.input_steps + self.step + self.output_steps - 1)
                input_data = feature_data[input_indices]
                output_data = feature_data[output_indices, 0]  # Assuming the 'value' column is the target
                self.samples.append((input_data, output_data))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_data, output_data = self.samples[idx]
        return torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data, dtype=torch.float)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_data, output_data = self.samples[idx]
        return torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data, dtype=torch.float)

# This is just the definition of the class, for usage you need to instantiate it with actual parameters
# For example: pv_dataset = PV_Dataset('/path/to/your/data.csv')

def read_data(data_path, input_steps=6, output_steps=1, step=1,feature_columns=['value'], train_size=0.7, test_size=0.2, batch_size=32):
    # Read and preprocess the data
    data = pd.read_csv(data_path)
    if feature_columns is None:
        feature_columns = data.columns.drop(['date', 'time_of_day', 'time', 'daylight_level'])
    mean = data[feature_columns].mean().values
    std = data[feature_columns].std().values
    data[feature_columns] = (data[feature_columns] - mean) / std

    # Create the full dataset
    full_dataset = PV_Dataset(data, input_steps, output_steps,step, feature_columns)
    
    # Calculate the sizes of each dataset
    total_samples = len(full_dataset)
    train_end = int(total_samples * train_size)
    test_end = train_end + int(total_samples * test_size)
    
    # Split the dataset indices for train, test, and validation
    train_data = list(range(0, train_end))
    test_data = list(range(train_end, test_end))
    val_data = list(range(test_end, total_samples))
    
    # Create DataLoaders for each dataset
    train_loader = DataLoader([full_dataset[i] for i in train_data], batch_size=batch_size, shuffle=False)  # Changed to shuffle=False for sequential data
    test_loader = DataLoader([full_dataset[i] for i in test_data], batch_size=batch_size, shuffle=False)
    val_loader = DataLoader([full_dataset[i] for i in val_data], batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader, mean, std


if __name__ =='__main__':

    data_path = '../levels/high.csv'
    # Example usage
    train_loader,test_loader,val_loader,mean,std = read_data(data_path, input_steps=6, output_steps=2, feature_columns=['value','GlobalR'])
    # Showing the shape of first batch in dataset
    # first_input, first_output = dataset[0]
    # print(first_input.shape, first_output.shape)
    print('-----------dataset information------------')
    print('mean:',mean)
    print('std:',std)
    print('train_dataset:',len(train_loader))
    print('test_dataset:',len(test_loader))
    print('val_loader:',len(val_loader))
    
    
    
    
