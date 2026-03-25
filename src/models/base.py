import torch.nn as nn

class BaseModel(nn.Module):
    """
    Abstract Base Class for all HAR models.
    Provides shared utility functions for research and logging.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def count_parameters(self):
        """Calculates the total number of trainable parameters (\theta)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def format_details(self):
        """Returns a string with architecture details for logging."""
        params = self.count_parameters()
        return f"Model: {self.__class__.__name__} | Trainable Parameters: {params:,}"