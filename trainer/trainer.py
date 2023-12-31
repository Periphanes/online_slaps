import torch
import torch.nn as nn

def facebookpagepage_features_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x = x.type(torch.FloatTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x).squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()
    
    else:
        output = model(x).squeeze()
        loss = criterion(output, y)

    return model, loss.item(), output, y

def facebookpagepage_sampling_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x).squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        scheduler.step()

    else:
        output = model(x).squeeze()
        loss = criterion(output, y)
    
    return model, loss.item(), output, y