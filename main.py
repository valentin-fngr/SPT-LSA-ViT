import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import config 
from model import ViTLSA
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def compute_accuracy(logits, target):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1) 
        return torch.mean(torch.where(preds == target, 1, 0).type(torch.float32)) 


def get_data(): 

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        torchvision.transforms.RandomHorizontalFlip(0.4), 
        torchvision.transforms.RandomRotation(20), 
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=6)

    return train_loader, test_loader


def get_criterion(): 
    return torch.nn.CrossEntropyLoss()

def get_optimizer(model): 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, verbose=True)
    return optimizer, scheduler

def get_model(): 
    model = ViTLSA(config.num_heads, config.num_blocks, d_model=config.d_model, num_classes=10)
    model = model.to(config.device) 
    return model 


def train(model, train_loader, optimizer, scheduler, criterion, epoch, writer=None): 

    train_loss = 0.0 
    train_acc = 0.0
    model.train()
    for sample in tqdm(train_loader): 

        image, target = sample 
        image = image.to(config.device) 
        target = target.to(config.device)

        # pred 
        preds = model(image) 
        train_acc += compute_accuracy(preds, target)
        loss = criterion(preds, target) 
        train_loss += loss.item()

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
    scheduler.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    print(f"Epoch : [{epoch}|{config.epochs}] train loss = {train_loss} - train acc = {train_acc}")


    if writer is not None: 
        writer.add_scalar("Train/Loss", train_loss, epoch) 
        writer.add_scalar("Train/accuracy", train_acc, epoch)
        writer.add_scalar("LR/scheduler", scheduler.get_last_lr()[0], epoch)
    return model 


def validate(model, test_loader, criterion, epoch, writer=None): 
    
    model.eval() 
    validation_loss = 0.0 
    total_acc = 0.0

    for sample in test_loader: 
        image, target = sample 
        image = image.to(config.device) 
        target = target.to(config.device) 

        preds = model(image) 
        total_acc += compute_accuracy(preds, target)
        loss = criterion(preds, target) 
        validation_loss += loss.item() 


    validation_loss = validation_loss / len(test_loader)
    total_acc = total_acc / len(test_loader)

    if writer is not None: 
        writer.add_scalar("Val/Loss", validation_loss, epoch) 
        writer.add_scalar("Val/accuracy", total_acc, epoch)
    print(f"val loss = {validation_loss} - val acc = {total_acc}")
    

    return model

    
    
def main(): 

    model = get_model() 
    print("number of parameters : ", model.get_number_parameters())
    train_loader, test_loader = get_data() 
    optimizer, scheduler = get_optimizer(model) 
    criterion = get_criterion() 

    model_name = f"diag_inf_d_model={config.d_model}_epochs={config.epochs}_#block{config.num_blocks}_#heads={config.num_heads}_lr={config.lr}_bs={config.batch_size}"
    writer = SummaryWriter("runs/" + model_name)

    print("--- Starting training ---", "\n") 
    for epoch in range(config.epochs): 

        train(model, train_loader, optimizer, scheduler, criterion, epoch, writer)
        validate(model, test_loader, criterion, epoch, writer)

    
    print("--- Training : DONE ---")



if __name__ == "__main__": 
    main()