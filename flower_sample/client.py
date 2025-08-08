# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Filter only 2 classes for binary classification
mask = y_train < 2
X_train = X_train[mask]
y_train = y_train[mask]

mask = y_test < 2
X_test = X_test[mask]
y_test = y_test[mask]

# Convert to tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # minimal training
            for X_batch, y_batch in trainloader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in testloader:
                output = self.model(X_batch)
                loss += self.loss_fn(output, y_batch).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(y_batch).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())