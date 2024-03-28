import torch
from torch import nn
from models import LSTM_Model,ODEModel
from eval import train,eval
import numpy as np 
import random
from args import args
from loguru import logger
from dataset import read_data



train_loader,test_loader,valid_loader,mean,std = read_data(args.data_path,input_steps=args.his_length,output_steps=args.pred_length,step=args.step,feature_columns=['value'],train_size=args.train_ratio,test_size=args.test_ratio,batch_size=args.batch_size)


# ---------------------------------------------------
# Define the model
# model = LSTM_Model(input_size=1, hidden_layer_size=32, future_steps=1)
model = ODEModel(input_size=1,hidden_size=32,future_steps=1)
lr = args.lr
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()
# ---------------------------------------------------

# random seed
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda:'+str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

if args.log:
    logger.add('log_{time}.log')
options = vars(args)
if args.log:
    logger.info(options)
else:
    print(options)
    
model = model.to(device)




best_valid_rmse = 1000 


for epoch in range(1, args.epochs+1):
    print("=====Epoch {}=====".format(epoch))
    print('Training...')
    loss = train(train_loader, model, optimizer, criterion, device)
    print('Evaluating...')
    train_rmse, train_mae, train_mape,train_r2= eval(train_loader, model, std, mean, device)
    valid_rmse, valid_mae, valid_mape,valid_r2 = eval(valid_loader, model, std, mean, device)

    if valid_rmse < best_valid_rmse:
        best_valid_rmse = valid_rmse
        print('New best results!')
        torch.save(model.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

    if args.log:
        logger.info(f'\n##on train data## loss: {loss}, \n' + 
                    f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}, r2 loss: {train_r2}\n' +
                    f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}, r2 loss: {valid_r2}\n')
    else:
        print(f'\n##on train data## loss: {loss}, \n' + 
                    f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}, r2 loss: {train_r2}\n' +
                    f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}, r2 loss: {valid_r2}\n')
    

model.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
test_rmse, test_mae, test_mape,test_r2 = eval(test_loader, model, std, mean, device)
print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape},r2 loss :{test_r2}')


import matplotlib.pyplot as plt

# Function to plot actual vs predicted data for the last day in the dataset
def plot_actual_vs_predicted(actual, predicted, title='Actual vs Predicted'):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

predictions_list = []
actuals_list =  []
for inputs, actuals in test_loader:
    # Assuming the model and data are on the same device, if not, adjust accordingly
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(inputs.to(device)).cpu().numpy()
    actuals = actuals.cpu().numpy()
    
    predictions_list.append(predictions.reshape(-1,))
    actuals_list.append(actuals.reshape(-1,))


pred = np.concatenate(predictions_list)
actual = np.concatenate(actuals_list)
print(pred.shape) 
print(actual.shape)   

plot_actual_vs_predicted(pred[55:144*10],actual[55:144*10])

# plot data