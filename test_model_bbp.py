import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm
from torch_geometric.loader import DataLoader

torch.manual_seed(12345)

device = 'cpu' #'cuda'
for _ in X_val:
    _.to(device)


from rdkit import RDLogger        
RDLogger.DisableLog('rdApp.*')    
#dataset = dataset.data_object.tolist()

# dataset = [data.to(device) for data in dataset.data_object.tolist()]

#dataset[42].is_cuda


# random.shuffle(dataset)

# split_cut = round(len(dataset)*0.2);split_cut

# test_dataset = dataset[:940]
# train_dataset = dataset[940:]

train_dataset = [data.to(device) for data in X_train.tolist()]
test_dataset = [data.to(device) for data in X_val.tolist()]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

batchs = 2**6 # aumentar el batchsize ha hecho que aumente bastante la accuracy
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
    return loss

from sklearn.metrics import f1_score,roc_auc_score

def test(loader):
    model.eval()
    readouts =  []
    labels = []
    all_preds = []
    all_labels = []

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        readoutvec = None
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred.cpu() == data.y.cpu()).sum())  # Check against ground-truth labels.

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

        #readouts.append(readoutvec)
        #labels.append(data.y)
        correct_ = correct / len(loader.dataset)
    epoch_f1 = f1_score(y_true=all_labels, y_pred=all_preds, average=None)
    #acc1 = accuracy_score(y_true=all_labels, y_pred=all_preds)
    return correct_, epoch_f1  # Derive ratio of correct predictions.


edge_dim = train_dataset[0].edge_attr[0].shape[0]  # harcoded


nheads = 4

#from  torch_geometric.nn.pool import TopKPooling 

class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # tiene sentido hacer dropout aqui´? para no favoreder a un subtipo diria que si. check for imbalance
        self.conv1 = TransformerConv(train_dataset[0].num_node_features, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=nheads) # ollo ao "hardcoded"
        self.conv2 = TransformerConv(hidden_channels*nheads, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=nheads)
        self.conv3 = TransformerConv(hidden_channels*nheads, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=nheads)
        #self.conv4 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=0.2, heads=1)
        self.lin1 = Linear(hidden_channels*nheads, hidden_channels) # harcoded now
        self.lin2 = Linear(hidden_channels, 2) # harcoded now

        self.dropout = Dropout(p=0.25)

    def forward(self, x, edge_index, edge_attr, batch):

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr)
        x = x.tanh()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.tanh()
        x = self.conv3(x, edge_index, edge_attr)
        #x = x.tanh()
        #x = self.conv4(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training) # igual hay que checkear esta función y bajar el ratio, esto puede estar bajando mucho el train!
        x = self.dropout(x) # igual hay que checkear esta función y bajar el ratio, esto puede estar bajando mucho el train!
        x = self.lin1(x)
        x = x.relu()
        x = self.dropout(x) # igual hay que checkear esta función y bajar el ratio, esto puede estar bajando mucho el train!
        x = self.lin2(x)
        
        return x



import pandas as pd

hd = 15

# df_res = pd.DataFrame(columns=['hd', 'train_acc', 'test_acc'])


# for hd in range(1,100,5):

model = GCN(hidden_channels=hd)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) # it was to 0.003
criterion = torch.nn.CrossEntropyLoss()


max_epochs = 201
acc_test_values, f1_test_values = [], []
acc_train_values, f1_train_values = [], []
list_loss = []

plt.clf()
plt.title(f'Accuracy results of toy drug classifier, hd:{hd}')
plt.ylim(0.4, 1.0)
plt.xlim(0., max_epochs+1)
plt.hlines(0.5, xmin=0, xmax=max_epochs,color='grey', linestyles='dashed', alpha=0.5)
plt.hlines(0.6795, xmin=0, xmax=max_epochs, color='orange', linestyles='dashed', alpha=0.8, label='Baseline RF & MFP')

fig_name = 'tmp_results_bbbp.png'

init = time.time()
print("Training....")
for epoch in tqdm(range(1, max_epochs)):
    loss = train()
    train_acc, train_f1 = test(train_loader)
    # test_acc, test_f1= test(test_loader)

    # acc_test_values.append(test_acc)
    acc_train_values.append(train_acc)
    list_loss.append(loss)
    #f1_train_values.append(train_f1)
    #åf1_test_values.append(test_f1)

    # if epoch%10==0:
    print(f'Epoch: {epoch:03d}, Loss {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        # plt.plot(acc_test_values, 'r', label='test')
        # # plt.plot(f1_test_values, 'r--', label='test')
        # plt.plot(acc_train_values, 'k', label='train')
        # # plt.plot(f1_train_values, 'k--', label='train')
        # plt.savefig(fig_name, dpi=330, bbox_inches='tight', pad_inches = 0.25)

end = time.time()

print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
#print(f'Epoch: {epoch:03d}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')

print(f'Elapsed time: {(end-init)/60} min')


plt.savefig(fig_name, dpi=330, bbox_inches='tight',  pad_inches = 0.25)



plt.clf()
plt.plot([element.detach().numpy() for element in list_loss], 'k')
plt.savefig('test_loss.png', dpi=330)



      ## check bias, do plots

    # new_row = pd.DataFrame({'hd': hd,
    #         'train_acc':train_acc,
    #         'test_acc': test_acc}, index=[hd])

    # df_res = pd.concat([df_res, new_row], ignore_index=True)
    # print(df_res)


# import seaborn as sns

# plt.clf()
# sns.lineplot(data=df_res, x='hd', y='test_acc')
# plt.savefig('boxplot_hd.pdf')