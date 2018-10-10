
# coding: utf-8

# In[166]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image, ImageFilter


# In[167]:


data_root = './data/fashionMNIST'
image_size = 28
in_channels = 1
num_classes = 10
num_fm = 256
batch_size = 128
workers = 4
num_epochs = 20


# In[168]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

if device.type == "cuda":
    print("Using %d GPUs." % (n_gpu))
else:
    print("Using CPU.")


# In[169]:


# Own Random Noise Transformation
class RandomNoise():
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img):
        if random.random() <= self.prob:
            img = img.filter(ImageFilter.GaussianBlur(1))
            return img
        return img


# In[170]:


data = dset.FashionMNIST(data_root, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(),
            RandomNoise(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            train=True,
            download=True)


# In[171]:


dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# In[172]:


test_data = dset.FashionMNIST(data_root, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            train=False,
            download=True)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# In[173]:


num_to_label = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]


# Let's have a quick look at the example batch!

# In[174]:


batch_img, batch_label = next(iter(dataloader))


# In[175]:


my_grid = utils.make_grid(batch_img, nrow=12, normalize=True)


# In[176]:


plt.figure(figsize=(8,8))
plt.imshow(np.transpose(my_grid, (1, 2, 0)))
plt.axis("off")
plt.show()


# In[177]:


print([num_to_label[batch_label[int(i)]] for i in range(len(batch_label))])


# In[178]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, int(num_fm/4), 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14
            
            nn.Conv2d(int(num_fm/4), int(num_fm/2), 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #7x7
            
            nn.Conv2d(int(num_fm/2), int(num_fm), 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)) # 3x3
        
        self.classifier = nn.Sequential(
            nn.Linear(3*3*num_fm, 2048),
            nn.ReLU(),
            
            nn.Linear(2048, 2048),
            nn.ReLU(),
            
            nn.Linear(2048, num_classes))
            
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 3*3*num_fm)
        x = self.classifier(x)
        return x


# In[179]:


model = CNN()
model = model.to(device)
if (device.type == "cuda" and (n_gpu > 1)):
    model = nn.DataParallel(model)


# In[180]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# Start Training Loop

# In[181]:


for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        model.zero_grad()
        batch_img, batch_label = data
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
        output = model(batch_img)
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()
        
        if i%50 == 0:
            print("[%d / %d]\t[%d / %d] Loss: %.4f" % (epoch, num_epochs, i, len(dataloader), loss))


# In[ ]:


confusion = np.zeros((10,10))

for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img = test_img.to(device)
    test_label = test_label.to(device)
    
    predicted_labels = model(test_img)
    _, predicted_labels = torch.max(predicted_labels, 1)
    
    for l in range(len(test_label)):
        confusion[test_label[l], predicted_labels[l]] += 1    


# In[ ]:


print(confusion)
plt.imshow(confusion, cmap="hot")
plt.savefig('./output/fashion_mnist_confusion.png')


# Most common mistake is to take class 0 for class 6, which means t-shirt for shirt

# In[ ]:


c_sum = confusion.sum(axis=1)
accuracy = np.trace(confusion/c_sum)/10
print("We have an average accuracy of %.4f" % (accuracy))

