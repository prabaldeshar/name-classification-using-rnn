import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(
            in_features=input_size + hidden_size, out_features=hidden_size
        )
        self.i2o = nn.Linear(
            in_features=input_size + hidden_size, out_features=output_size
        )
        self.softmax = nn.LogSoftmax(
            dim=1
        )  # dim(int) - A dimension along which LogSoftmax will be computed

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


category_lines, all_categories = load_data()
n_categories = len(all_categories)
