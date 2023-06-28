import importlib
import os

def get_model(args):
    if "baseline" in args.model.lower():
        model_module = importlib.import_module("models.baseline." + args.model)
    else:
        model_module = importlib.import_module("models." + args.model)
    
    model = getattr(model_module, args.model.upper())

    return model