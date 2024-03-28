import torch
from torch import nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )
        # 应用Xavier初始化
        self.apply(self.init_weights)

    def forward(self, t, y):
        return self.net(y)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class ODEBlock(nn.Module):
    def __init__(self, ode_func):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func

    def forward(self, x):
        t = torch.linspace(0., 1., 10)
        out = odeint(self.ode_func, x, t, method='dopri5')[-1]
        return out

class LSTM_ODE(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, future_steps=1):
        super(LSTM_ODE, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.odefunc = ODEFunc(hidden_size, hidden_size*2)
        self.linear = nn.Linear(hidden_size, future_steps)
        
        self.odeblock = ODEBlock(self.odefunc)

        # 应用Xavier初始化
        self.apply(self.init_weights)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        ode_output = self.odeblock(last_time_step_out)
        pred = self.linear(ode_output)
        
        return pred

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight_ih" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
                elif "weight_hh" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
                elif "bias" in param:
                    m._parameters[param].data.fill_(0.01)

