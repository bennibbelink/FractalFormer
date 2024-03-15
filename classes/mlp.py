import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    """
    This class implements a multi-layer perceptron with a GeGLU gating mechanism. The GeGLU
    activation combines a standard GeLU activation with a learned gating mechanism, enabling
    the network to control the flow of information more dynamically.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the GemmaMLP module.

        Parameters:
            hidden_size (int): The size of the input and output tensors.
            intermediate_size (int): The size of the tensor after the initial transformation
                                     and before the gating and final projection. This is typically
                                     larger than the hidden size to allow for a richer representation.
            dropout (float): the dropout rate to use during training in forwardTuple()
        """
        super().__init__()
        self.verbose = False
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        assert intermediate_size % hidden_size == 0
        self.intermediate_multiplier = intermediate_size // hidden_size

        # Linear transformation for the gating mechanism, projecting input to an intermediate size.
        self.Wgate = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
        self.Bgate = nn.Parameter(torch.Tensor(intermediate_size))

        # Linear transformation for the input tensor, also projecting to the intermediate size but
        # intended for element-wise multiplication with the gated output.
        self.Wup = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
        self.Bup = nn.Parameter(torch.Tensor(intermediate_size))

        # Linear transformation to project the gated and combined tensor back to the original
        # hidden size, completing the MLP structure.
        self.Wdown = nn.Parameter(torch.Tensor(intermediate_size, hidden_size))
        self.Bdown = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize weights with uniform distribution
        # For gate & up, where in_features is hidden_size
        limit_gateup = 1 / np.sqrt(hidden_size)
        nn.init.uniform_(self.Wgate, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Bgate, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Wup, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Bup, -limit_gateup, limit_gateup)
        
        # For down, where in_features is intermediate_size
        limit_down = 1 / np.sqrt(intermediate_size)
        nn.init.uniform_(self.Wdown, -limit_down, limit_down)
        nn.init.uniform_(self.Bdown, -limit_down, limit_down)
        
        # defining our dropout for training in forwardTuple()
        self.drop = nn.Dropout(dropout)

    def forwardTensor(self, x, model:int=0):
        """
        Defines the forward pass of the MLP module during inference.

        Parameters:
            x (Tensor): The input tensor to the MLP. 
                        shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used
            model (int): the indicator of which model we're using. 
                        used in calculating our skip length for splicing. 
                        defaults to the equivalent of what's used in MatFormer+, meaning no skip, aka we use the top-left-most splice

        Returns:
            Tensor: The output tensor after applying the GeGLU gating mechanism and the MLP transformations.
        """
        if self.verbose: 
            print("------------- MLP.forwardTensor() ------------")
            print(f"x: {x.shape}\n{x}")
            
        # figuring out how we should do our splicing
        d_dim = x.shape[-1]
        d_skip = model * d_dim
        i_dim = d_dim * self.intermediate_multiplier
        i_skip = model * i_dim
        if self.verbose: 
            print(f"d_dim: {d_dim}")
            print(f"d_skip: {d_skip}")
            print(f"i_dim: {i_dim}")
            print(f"i_skip: {i_skip}")
        
        # Applies linear transformation for gating.
        Wgate = self.Wgate[d_skip:d_skip + d_dim, i_skip:i_skip + i_dim]
        Bgate = self.Bgate[i_skip:i_skip + i_dim]
        Xgate = x @ Wgate + Bgate
        if self.verbose: 
            print(f"Wgate: {self.Wgate.shape}\n{self.Wgate}")
            print(f"Wgate spliced: {Wgate.shape}\n{Wgate}")
            print(f"Bgate: {self.Bgate.shape}\n{self.Bgate}")
            print(f"Bgate spliced: {Bgate.shape}\n{Bgate}")
            print(f"Xgate: {Xgate.shape}\n{Xgate}")

        # Applies GeLU activation to the gate, introducing non-linearity and enabling the gating mechanism.
        Xgate = nn.functional.gelu(Xgate)
        if self.verbose: print(f"GeLU'ed Xgate: {Xgate.shape}\n{Xgate}")

        # Applies another linear transformation to the input tensor for subsequent combination with the gate.
        Wup = self.Wup[d_skip:d_skip + d_dim, i_skip:i_skip + i_dim]
        Bup = self.Bup[i_skip:i_skip + i_dim]
        Xup = x @ Wup + Bup
        if self.verbose: 
            print(f"Wup: {self.Wup.shape}\n{self.Wup}")
            print(f"Wup spliced: {Wup.shape}\n{Wup}")
            print(f"Bup: {self.Bup.shape}\n{self.Bup}")
            print(f"Bup spliced: {Bup.shape}\n{Bup}")
            print(f"Xup: {Xup.shape}\n{Xup}")

        # Element-wise multiplication of the gated tensor with the transformed input tensor, modulating
        # the input based on the gate's activation.
        Xfuse = Xgate * Xup
        if self.verbose: print(f"Xfuse: {Xfuse.shape}\n{Xfuse}")

        # Applies the final linear transformation to project the modulated tensor back to the hidden size.
        Wdown = self.Wdown[i_skip:i_skip + i_dim, d_skip:d_skip + d_dim]
        Bdown = self.Bdown[d_skip:d_skip + d_dim]
        outputs = Xfuse @ Wdown + Bdown
        if self.verbose: 
            print(f"Wdown: {self.Wdown.shape}\n{self.Wdown}")
            print(f"Wdown spliced: {Wdown.shape}\n{Wdown}")
            print(f"Bdown: {self.Bdown.shape}\n{self.Bdown}")
            print(f"Bdown spliced: {Bdown.shape}\n{Bdown}")
            print(f"outputs: {outputs.shape}\n{outputs}") 
            print("------------- END MLP.forwardTensor() ------------")

        # Returns the final output tensor of the MLP, after gating and modulation.
        return outputs

    def forwardTuple(self, x, drop_bool: bool = True):
        """
        Defines the forward pass of the MLP module during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors to the MLP. 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the GeGLU gating mechanism and the MLP transformations.
        """
        if self.verbose: 
            print("------------- MLP.forwardTuple() ------------")
            print(f"x: {x}")

        # if we had sent through the config we could've just grabbed these values from there but too late now
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]
        if self.verbose: 
            print(f"num_levels: {num_levels}")
            print(f"models_per_level: {models_per_level}")
        
        out = ()
        for i in range(num_levels):
            if self.verbose: print(f"i: {i}")
            
            out_lvl = ()
            for j in range(models_per_level[i]):
                if self.verbose: print(f"j: {j}")

                output = self.forwardTensor(x[i][j], model=j)
                if self.verbose: print(f"forwardTensor() output: {output.shape}\n{output}")
                    
                out_lvl += (self.drop(output),) if drop_bool else (output,)

            # pretty sure i have to save & store everything without overwriting to prevent in-place arguments. so annoying
            if self.verbose: print(f"out_lvl: {out_lvl}")
            out += (out_lvl,)
        
        if self.verbose:
            print(f"out: {out}")
            print("------------- END MLP.forwardTuple() ------------")
        return out
        
    def forward(self, x, model=0, drop_bool = True):
        train = True if type(x) == tuple else False
        if self.verbose: print(f"---------- MLP Input: {'Tuple' if train else 'torch.Tensor'} ------------")
        return self.forwardTuple(x, drop_bool) if train else self.forwardTensor(x, model)