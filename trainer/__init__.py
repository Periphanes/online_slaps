from .trainer import *

def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.trainer == "example_trainer":
        # model, iter_loss = example_train(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
        iter_loss = 1
        pass
    if args.trainer == "facebookpagepage_features":
        model, iter_loss, pred, true = facebookpagepage_features_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "facebookpagepage_sampling":
        model, iter_loss, pred, true = facebookpagepage_sampling_trainer(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected Trainer is not Prepared Yet")
        raise NotImplementedError

    return model, iter_loss, pred, true