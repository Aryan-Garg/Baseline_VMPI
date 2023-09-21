import torch

def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)