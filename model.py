import torch.nn.functional as F
import torch.nn as nn

class is_she_mad(nn.Module):
    def __init__(self, modality_size):
        super(is_she_mad, self).__init__()
        self.fc1 = nn.Linear(modality_size,200)
        self.fc2 = nn.Linear(200,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out
