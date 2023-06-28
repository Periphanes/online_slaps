from .trainer import *

def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.trainer == "example_trainer":
        # model, iter_loss = example_train(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
        iter_loss = 1
        pass

    else:
        print("Selected Trainer is not Prepared Yet")
        raise NotImplementedError

    return model, iter_loss 