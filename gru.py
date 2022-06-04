import torch
import torch.nn as nn


class Audio_GRU(nn.Module):
    def __init__(self, config):
        super(Audio_GRU, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.batch_size = config['batch_size']

        # gru layer
        self.gru_layer = nn.GRU(self.input_size,
                                self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        # fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            # nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, x):
        self.gru_layer.flatten_parameters()
        # current x size : torch.Size([32, 114, 4, 988])
        # input needs to be of shape
        # x -> (batch_size, sequence_len, input_size)

        reshaped = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], -1))  # (batch_size, samples * timesteps, input_size)

        # print(reshaped.shape) # torch.Size([32, 456, 988])

        # Initializing hidden state for first input with zeros
        #h = torch.zeros(self.num_layers, reshaped.size(0), self.hidden_size).requires_grad_().cuda()
        #print("h shape is: ")
        #print(h.shape)

        out, _ = self.gru_layer(reshaped) # batch_size, seq_len, hidden_size
        # print(out.shape) #torch.Size([32, 456, 256])

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # out = out[:, -1, :]
        out = out.sum(dim=1) # shape -> (32, 256)
        out = self.fc_layer(out)
        # out = torch.reshape(out, (out.shape[0], out.shape[1]//2, 2))
        return out
