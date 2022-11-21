import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
import random
import torchvision.models as models

# Configuration
myseed = 66666
batch_size = 64
_dataset_dir = "../input/food11"
learning_rate = 0.0003
weight_decay = 1e-5
n_epochs = 10
patience = 300 # If no improvement in 'patience' epochs, early stop
_exp_name = "food_project"
mean = [0.554, 0.450, 0.343]
std = [0.231, 0.241, 0.241]
members = 3

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
    transforms.RandomErasing()
])

class FoodDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        if files != None:
            self.files = sorted(files)
        else: 
            self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

def get_model(device):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(int(model.fc.in_features), 11, device)
    model = model.to(device)
    return model
    
def train(model, device, train_loader, criterion, epoch):
     # ---------- Training ----------
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        # Clip the gradient norms for stable training.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        # Update the parameters with computed gradients.
        optimizer.step()
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1}/{n_epochs} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

def validate(model, device, valid_loader, criterion, epoch):
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    # Print the information.
    print(f"[ Valid | {epoch + 1}/{n_epochs} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    return valid_loss, valid_acc

def allocate(num, length):
    ans = []
    sum = 0
    for _ in range(num):
        ans.append(int(length/num))
        sum += int(length/num)
    if length - sum > 0: 
        ans[-1] += length - sum
    return ans

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)


class Member():
    def __init__(self, train_loader, valid_loader, model_state, name, device):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.local_model = get_model(device)
        self.local_model.load_state_dict(model_state)
        self.name = name
    
    def copy_model(self, model_state):
        self.local_model.load_state_dict(model_state)
    def train_model(self, device, criterion, epoch):
        print(f"Training {self.name}'s model for epoch {epoch}")
        train(self.local_model, device, self.train_loader, criterion, epoch)
        print(f"{self.name}'s model for epoch {epoch} has been trained")

    def validate_model(self, device, criterion, epoch):
        print(f"Validating {self.name}'s model for epoch {epoch}")
        validate(self.local_model, device, self.valid_loader, criterion, epoch)
        print(f"{self.name}'s model for epoch {epoch} has been validated")
    
    def aggregate_models(self, states):
        for layer in states[0]:
            for i in range(1, len(states)):
                states[0][layer] += states[i][layer]
            states[0][layer] = torch.div(states[0][layer], len(states))
        return states[0]

    def judge(self, aggregated_model, global_model, device, criterion, epoch):
        print(f"{self.name} is judging whether the aggregation is good enough")
        print("Validating global model")
        _, local_acc = validate(global_model, device, self.valid_loader, criterion, epoch)
        print("Validating aggregated model")
        _, aggr_acc = validate(aggregated_model, device, self.valid_loader, criterion, epoch)
        if aggr_acc > local_acc - 0.05: 
            print(f"{self.name} voted yes")   
            return True
        else:
            print(f"{self.name} voted no")  
            return False

# main
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
train_dir = os.path.join(_dataset_dir, "training")
valid_dir = os.path.join(_dataset_dir, "validation")
pic_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith(".jpg")]
splited_train_data = random_split(dataset=pic_list, lengths=allocate(members, len(pic_list)))
splited_train_data = [list(i) for i in splited_train_data]
train_sets = [FoodDataset(train_dir, tfm=train_tfm, files=i) for i in splited_train_data]
train_loaders = [DataLoader(train_sets[i], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) for i in range(members)]
valid_set = FoodDataset(valid_dir, tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
global_model = get_model(device)
criterion = nn.CrossEntropyLoss()
stale = 0
best_acc = 0
commitee = [Member(train_loaders[i], valid_loader, global_model.state_dict(), f"Member_{i}", device) for i in range(members)]

for epoch in range(n_epochs):
    # Each client trains its local model
    collected_models = []
    for member in commitee:
        member.train_model(device, criterion, epoch)
        member.validate_model(device, criterion, epoch)
        collected_models.append(member.local_model.state_dict())
    
    print(f"{len(collected_models)} models has been collected")
    voting = True
    while voting == True:
        vote = random.randint(0, members - 1)
        print(f"{commitee[vote].name} has been chosen as the leader, he is going to make an aggregation")
        aggregated_model = get_model(device)
        aggregated_state = commitee[vote].aggregate_models(collected_models)
        aggregated_model.load_state_dict(aggregated_state)
        # Vote on the aggregated_model
        vote_count = 0
        for member in commitee:
            if member.judge(aggregated_model, global_model, device, criterion, epoch):
                vote_count += 1

        if vote_count > members / 2:
            voting = False
            global_model.load_state_dict(aggregated_model.state_dict())
            print(f"The aggregation from {commitee[vote].name} has been accepted and updated to be the new global model")
            for member in commitee:
                member.copy_model(global_model.state_dict())
            # new global model has been updated, now we calculate some global data
            print("New global model has been determined!")
            valid_loss, valid_acc = validate(global_model, device, valid_loader, criterion, epoch)
            if valid_acc > best_acc:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1}/{n_epochs} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1}/{n_epochs} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(global_model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(f"No improvment {patience} consecutive epochs, early stopping")
                    break
        else:
            print(f"The aggregation from {commitee[vote].name} has been rejected")

test_set = FoodDataset(os.path.join(_dataset_dir,"evaluation"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
model_best = get_model(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()
    #create test csv
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)