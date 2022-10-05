from torchvision.models import resnet34, densenet161
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from preprocessing import *
from ESCdataloader import *

df = pd.read_csv('./ESC-50/meta/esc50.csv')
train = df[df['fold']!=5]
valid = df[df['fold']==5]
train_data = ESC50Data('audio', train, 'filename', 'category')
valid_data = ESC50Data('audio', valid, 'filename', 'category')
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

if torch.cuda.is_available():
	device=torch.device('cuda:0')
else:
	device=torch.device('cpu')
	
resnet_model = resnet34(pretrained=True)
resnet_model.fc = nn.Linear(512,50)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

densenet_model = densenet161(pretrained=True)
#densenet_model.fc = nn.Linear(512,50)
#densenet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
densenet_model = densenet_model.to(device)

learning_rate = 2e-4
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
epochs = 50
loss_fn = nn.CrossEntropyLoss()
resnet_train_losses=[]
resnet_valid_losses=[]

def setlr(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer

def lr_decay(optimizer, epoch):
	if epoch%10==0:
		new_lr = learning_rate / (10**(epoch//10))
		optimizer = setlr(optimizer, new_lr)
		print(f'Changed learning rate to {new_lr}')
	return optimizer

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
	for epoch in tqdm(range(1,epochs+1)):
		model.train()
		batch_losses=[]
		if change_lr:
			optimizer = change_lr(optimizer, epoch)
		for i, data in enumerate(train_loader):
			x, y = data
			optimizer.zero_grad()
			x = x.to(device, dtype=torch.float32)
			y = y.to(device, dtype=torch.long)
			y_hat = model(x)
			loss = loss_fn(y_hat, y)
			loss.backward()
			batch_losses.append(loss.item())
			optimizer.step()
		
		print(f'Epoch - {epoch} Train-Loss : {np.mean(batch_losses)}')
		train_losses.append(np.mean(batch_losses))
		model.eval()
		batch_losses=[]
		trace_y = []
		trace_yhat = []
		for i, data in enumerate(valid_loader):
			x, y = data
			x = x.to(device, dtype=torch.float32)
			y = y.to(device, dtype=torch.long)
			y_hat = model(x)
			loss = loss_fn(y_hat, y)
			trace_y.append(y.cpu().detach().numpy())
			trace_yhat.append(y_hat.cpu().detach().numpy())      
			batch_losses.append(loss.item())
		valid_losses.append(np.mean(batch_losses))
		trace_y = np.concatenate(trace_y)
		trace_yhat = np.concatenate(trace_yhat)
		accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
		print(f'Epoch - {epoch} Valid-Loss : {np.mean(batch_losses)} Valid-Accuracy : {accuracy}')
	return model, train_losses, valid_losses

def graph(train_losses, valid_losses, filepath=None):
	tl = np.asarray(train_losses)
	vl = np.asarray(valid_losses)
	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.plot(tl)
	plt.legend(['Train Loss'])
	plt.subplot(1,2,2)
	plt.plot(vl,'orange')
	plt.legend(['Valid Loss'])
	plt.savefig(filepath)
	plt.close()

densenet_model, train_losses, valid_losses = train(densenet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses, resnet_valid_losses, lr_decay)

with open('esc50resnet.pth','wb') as f:
  torch.save(densenet_model, f) # save model

graph(train_losses, valid_losses, 'densenet161_loss')
