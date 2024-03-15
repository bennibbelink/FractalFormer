from . import ff_config
import torch
import torch.nn as nn

class FractalLoss(nn.Module):
    verbose = False
    def __init__(self, config: ff_config.Config):
        super().__init__()
        self.config = config

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        """
        input: 
            - logits are a tuple of tuples of tensors each of shape [batch_size, max_seq_len, vocab_size]
            - target is a shape [batch_size, max_seq_len] tensor of the integer indices of the correct tokens
        output: a tensor containing a single float of the loss value
        """
        if self.verbose: 
            print("------------- FractalLoss.forward() ------------")
            print(f"logits:\n{logits}")
            
        assert type(logits) == tuple # since this function should only be used during training
            
        # should only be used during training, so we assert input_len == max_position_embeddings
        b,t,v = logits[0][0].shape
        if self.verbose: print(f"b:{b}, t:{t}, v:{v}, b*t:{b*t}")
        assert t == self.config.max_position_embeddings
        
        # Calculate losses for each output and stack them. 
        # i apologize for the weird format instead of regular for loops, but it feels better in my head
        loss = torch.stack([ # stacks across levels
                            torch.stack( # stacks across models in level
                                        [self.criterion(logits_ij.view(b*t, v), # reshapes for CELoss
                                                        target.view(b*t)) 
                                         for logits_ij in logits[i]] # iterates across models in level
                            ).sum() # sums across models in level
                            for i in range(len(logits))] # iterates across levels
                          ).sum() # sums across levels

        if self.verbose:
            print(f"final loss: {loss}")
            print("------------- END FractalLoss.forward() ------------")

        return loss