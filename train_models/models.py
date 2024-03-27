import torch
from torch import nn
from torchdiffeq import odeint

# LSTM
class LSTM_Model(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, future_steps=2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features per time step.
            hidden_layer_size (int): Size of the LSTM hidden layer.
            future_steps (int): Number of future steps to predict.
        """
        super(LSTM_Model, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.future_steps = future_steps

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Define a fully connected layer to map from hidden state space to output space
        self.linear = nn.Linear(hidden_layer_size, future_steps)

    def forward(self, input_seq):
        """
        Forward pass through the network.
        
        Args:
            input_seq (Tensor): Input batch of sequences; shape should be (batch, seq_len, input_size).
            
        Returns:
            Tensor: The network's output with shape (batch, future_steps).
        """
        # Passing in the input into the model and obtaining outputs
        lstm_out, _ = self.lstm(input_seq)

        # Take the outputs of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Map the outputs of the last time step to the future steps
        future_predictions = self.linear(last_time_step_out)

        return future_predictions

 
class ODEFunc(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, t, x):
        return self.net(x)

# ODE模型
class ODEModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=100, future_steps=1):
        super(ODEModel, self).__init__()
        self.hidden_size = hidden_size
        self.future_steps = future_steps

        # 定义ODE函数
        self.odefunc = ODEFunc(input_size, hidden_size)

        # 定义映射到输出空间的全连接层
        self.linear = nn.Linear(input_size, future_steps)  

    def forward(self, input_seq):
        # 确定积分时间，从0积分到1等价于在一个单位时间步内模拟动态
        integration_time = torch.tensor([0, 1]).float().to(input_seq.device)
        
        # 初始化输出预测张量
        batch_size, time_seq, _ = input_seq.shape
        predictions = torch.zeros(batch_size, self.future_steps).to(input_seq.device)
        
        # 遍历每一个时间步
        for i in range(time_seq):
            # 从时间序列中提取当前时间步的数据
            current_step = input_seq[:, i, :]
            
            # 使用odeint进行前向传播，只关心最终时间点的值
            ode_solution = odeint(self.odefunc, current_step, integration_time, method='dopri5')[-1]
            
            # 汇总每一时间步的预测结果
            predictions = self.linear(ode_solution)
        
        return predictions