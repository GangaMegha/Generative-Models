import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class LSTM(nn.Module):
    def __init__(self, embed_dim, output_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Next state prediction model
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.n_layers, bidirectional=False, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        input_size = self.hidden_dim + self.output_dim
        self.fc = nn.Linear(input_size, 32)
        self.act = nn.ReLU()

        # Maps the output of the LSTM embedding.
        self.fc2 = nn.Linear(32, self.output_dim)
        
        # Between -1 and 1
        self.output = nn.Tanh()

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)
        torch.nn.init.constant_(self.fc2.bias, 0.1)

    def init_hidden(self, batch_size):
        return (torch.randn(self.n_layers, batch_size, self.hidden_dim),
                torch.randn(self.n_layers, batch_size, self.hidden_dim))
        
    def get_pred(self, inputs, flag=True):
        if(flag):
            self.hidden = self.init_hidden(inputs.size()[0])
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        return lstm_out

    def forward(self, inputs, prev_z, flag):

        lstm_out = self.get_pred(inputs, flag)

        input_cat = torch.cat([lstm_out, prev_z], dim=2)

        out = self.dropout(input_cat)
        out = self.fc(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)

        pred = self.output(out)

        return pred


class FF(nn.Module):
    def __init__(self, embed_dim, output_dim, drop_prob=0.5):
        super(FF, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Next state prediction model

        self.dropout = nn.Dropout(drop_prob)

        input_size = self.embed_dim + self.output_dim

        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64,24)
        self.act2 = nn.ReLU()

        # Maps the output of the LSTM embedding.
        self.fc3 = nn.Linear(24, self.output_dim)

        # Between -1 and 1
        self.output = nn.Tanh()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        torch.nn.init.constant_(self.fc3.bias, 0.1)


    def forward(self, inputs, prev_z, flag):

        input_cat = torch.cat([inputs, prev_z], dim=2)

        out = self.fc1(input_cat)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.act2(out)
        out = self.dropout(out)

        out = self.fc3(out)

        pred = self.output(out)

        return pred