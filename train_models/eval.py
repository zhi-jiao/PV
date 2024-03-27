import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan, epsilon=1e-8):
    """
    Calculate the Modified Mean Absolute Percentage Error (MAPE) while ignoring the specified null values and adding a small constant to avoid division by zero.
    
    Args:
        y_true (np.array): The ground truth values.
        y_pred (np.array): The predicted values.
        null_val (float, optional): The value to be ignored in the y_true. Defaults to np.nan.
        epsilon (float, optional): Small constant added to the denominator to avoid division by zero. Defaults to 1e-8.
    
    Returns:
        float: The modified masked MAPE value.
    """
    # Create a mask array that is True wherever y_true is not the null value
    mask = ~np.isclose(y_true, null_val)
    
    # Calculate the modified MAPE
    mape = np.abs((y_pred - y_true) / (y_true + epsilon))
    
    # Apply mask to MAPE: Ignore the positions where y_true has the null value
    masked_mape = mape[mask]
    
    # Compute the mean of the masked MAPE values
    return np.mean(masked_mape) * 100

def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def r_squared(y_true, y_pred):
    """
    Calculate the coefficient of determination R^2 of the prediction.
    
    Args:
        y_true (np.array): The ground truth values.
        y_pred (np.array): The predicted values.
    
    Returns:
        float: The R^2 score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Total sum of squares
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    # Residual sum of squares
    rss = np.sum((y_true - y_pred) ** 2)
    # R^2 score
    r2_score = 1 - (rss / tss)
    return r2_score


# Train 
def train(loader, model, optimizer, criterion, device):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    batch_r2_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm,0,1e-3)
        r2_loss = r_squared(target_unnorm,out_unnorm)
        # print(target_unnorm,out_unnorm)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss
        batch_r2_loss += r2_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1),batch_r2_loss /(idx+1)
