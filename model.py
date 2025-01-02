#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# model.py

import torch
import torch.nn as nn

class BinaryClassificationModel(nn.Module):
    """
    Simple feed-forward network for binary classification.
    """
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 42),
            nn.ReLU(),
            nn.Linear(42, 35),
            nn.ReLU(),
            nn.Dropout(0.4896),
            nn.Linear(35, 1)  # Output layer for binary classification
        )
        
    def forward(self, x):
        return self.network(x)

class FinalBinaryClassificationModel(nn.Module):
    """
    A version that can take hyperparameters for hidden dimensions and dropout.
    """
    def __init__(self, input_dim, hn1, hn2, dr):
        super(FinalBinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hn1)
        self.fc2 = nn.Linear(hn1, hn2)
        self.dropout = nn.Dropout(dr)
        self.fc3 = nn.Linear(hn2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

