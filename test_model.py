import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


import matplotlib.pyplot as plt
import time

from torch_geometric.loader import DataLoader

torch.manual_seed(12345)

device = 'cuda'
for _ in dataset:
    _.to(device)

#dataset = dataset.data_object.tolist()

dataset = [data.to(device) for data in dataset.data_object.tolist()]

#dataset[42].is_cuda

import random

random.shuffle(dataset)

test_dataset = dataset[:150]
train_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

batchs = 96*2 # aumentar el batchsize ha hecho que aumente bastante la accuracy
train_loader = DataLoader(train_dataset, batch_size=batchs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchs, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()



from torch.nn import Linear, Tanh
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
         loss = criterion(out.cpu(), data.y.cpu())  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     readouts =  []
     labels = []
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         readoutvec = None
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred.cpu() == data.y.cpu()).sum())  # Check against ground-truth labels.
         #readouts.append(readoutvec)
         #labels.append(data.y)
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


edge_dim = dataset[0].edge_attr[0].shape[0]  # harcoded
nheads = 2

#from  torch_geometric.nn.pool import TopKPooling 

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # tiene sentido hacer dropout aqui´? para no favoreder a un subtipo diria que si. check for imbalance
        self.conv1 = TransformerConv(dataset[0].num_node_features, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=nheads) # ollo ao "hardcoded"
        self.conv2 = TransformerConv(hidden_channels*2, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=nheads)
        #self.conv3 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=1)
        #self.conv4 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=1)
        self.lin = Linear(hidden_channels*2, 19) # harcoded now

    def forward(self, x, edge_index, edge_attr, batch):

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr)
        x = x.tanh()
        x = self.conv2(x, edge_index, edge_attr)
        #x = x.tanh()
        #x = self.conv3(x, edge_index, edge_attr)
        #x = x.tanh()
        #x = self.conv4(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training) # igual hay que checkear esta función y bajar el ratio, esto puede estar bajando mucho el train!
        x = self.lin(x)
        
        return x



model = GCN(hidden_channels=25)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) # decrease to 0.001
criterion = torch.nn.CrossEntropyLoss()


max_epochs = 251
acc_test_values = []
acc_train_values = []

plt.clf()
plt.title('Accuracy results of toy drug classifier')
plt.ylim(0.4, 1.0)
plt.xlim(0., max_epochs+1)
plt.hlines(0.5, xmin=0, xmax=max_epochs,color='grey', linestyles='dashed', alpha=0.5)
plt.hlines(0.6795, xmin=0, xmax=max_epochs, color='orange', linestyles='dashed', alpha=0.8, label='Baseline RF & MFP')

init = time.time()
print("Training....")
for epoch in range(1, max_epochs):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    acc_test_values.append(test_acc)
    acc_train_values.append(train_acc)
    
    if epoch%10==0:
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        plt.plot(acc_test_values, 'r', label='test')
        plt.plot(acc_train_values, 'k', label='train')
        plt.savefig('test_ressults.png', dpi=330, bbox_inches='tight', pad_inches = 0.25)

end = time.time()

print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
print(f'Elapsed time: {(end-init)/60} min')

plt.savefig('test_ressults.png', dpi=330, bbox_inches='tight',  pad_inches = 0.25)


## check bias, do plots