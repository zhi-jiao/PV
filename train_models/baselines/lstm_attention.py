import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Attention layer
        self.attention_layer = nn.Linear(hidden_dim, 1)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Initialize attention layer weights
        nn.init.xavier_uniform_(self.attention_layer.weight.data)
        nn.init.constant_(self.attention_layer.bias.data, 0)
        
        # Initialize FC weights
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)

    def attention_net(self, lstm_output):
        """
        Apply attention mechanism on the LSTM output
        """
        attention_weights = torch.tanh(self.attention_layer(lstm_output))
        attention_weights = F.softmax(attention_weights, dim=1)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        lstm_output, (hn, cn) = self.lstm(x, (h0, c0))

        # Attention layer
        attention_output = self.attention_net(lstm_output)

        # Fully connected layer
        output = self.fc(attention_output)

        return output

