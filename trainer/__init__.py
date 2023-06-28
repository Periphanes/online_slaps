from .trainer import *

def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.trainer == "example_trainer":
        # model, iter_loss = example_train(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
        iter_loss = 1
        pass
    if args.trainer == "facebookpagepage_random":
        try:
            model, iter_loss, pred, true = facebookpagepage_random_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
        except:
            model, iter_loss = facebookpagepage_random_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected Trainer is not Prepared Yet")
        raise NotImplementedError

    if flow_type == "val":
        return model, iter_loss, pred, true

    return model, iter_loss 