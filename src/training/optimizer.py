import torch.optim as optim

def get_optimizer(model_parameters, optim_name="adam", lr=0.001, weight_decay=0.0):
    if optim_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "adamw":
        # AdamW decouples weight decay, often better for Transformer/LSTM hybrids
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optim_name} not supported.")