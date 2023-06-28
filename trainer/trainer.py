import torch
import torch.nn as nn

def facebookpagepage_random_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type=None):
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

    return model, loss.item(), output, y