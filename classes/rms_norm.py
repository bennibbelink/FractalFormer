import torch

class RMSNorm(torch.nn.Module):
    """
    Implements the RMS Normalization (Root Mean Square Normalization) layer.
    RMSNorm is a variant of layer normalization that normalizes the activations
    of the previous layer based on their root mean square value.

    Parameters:
    - dim (int): The dimension of the input features the normalization is applied to.
    - eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    - add_unit_offset (bool): If True, adds a unit (1) to the learned scaling coefficient, effectively
      starting with no scaling. If False, the scaling coefficient starts from zero. Default is True.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        #add_unit_offset: bool = True,
    ):
        super().__init__() 
        self.eps = eps  # Small epsilon value for numerical stability since you can't divide by 0
        #self.add_unit_offset = add_unit_offset  # Flag to determine if a unit should be added to the weight
        
        # Initialize the weight parameter with zeros, which will be learned during training.
        # The shape of the weight is [dim], meaning one weight per feature dimension.
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    verbose = False

    def _norm(self, x):
        """
        Private helper function to normalize the input tensor.

        Parameters:
        - x (Tensor): The input tensor to normalize.

        Returns:
        - Tensor: The normalized tensor.
        """
        # Calculate the root mean square value for each feature (across the last dimension),
        # then use reciprocal square root (rsqrt) for normalization.
        # Add self.eps to the denominator for numerical stability.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, model: int = 0) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer

        Parameters:
        - x (Tensor): The input tensor to normalize.
        - model (int): the index indicating the model being used in this layer. used for splicing self.weight

        Returns:
        - output: The normalized and scaled tensor.
        """
        if self.verbose:
            print("------------- RMSNorm.forward() ------------")
            print(f"x: {x.shape}\n{x}")
            
        # Normalize the input tensor using the _norm function and ensure the data type matches the input.
        x = self._norm(x.float()).type_as(x)
        if self.verbose: print(f"normed x: {x.shape}\n{x}")
        
        # grabbing x's dimension to use for splicing
        dim = x.shape[-1]
        
        # calculating skip for our splice
        skip = model * dim
        if self.verbose: 
            print(f"dim: {dim}")
            print(f"skip: {skip}")
        
        # scale the normalized tensor by (1 + self.weight), which effectively starts with no scaling
        spliced_scale = self.weight[skip:skip + dim]
        output = x * (1 + spliced_scale)
        if self.verbose:
            print(f"spliced scale: {spliced_scale.shape}\n{spliced_scale}")
            print(f"scaled normed x: {output.shape}\n{output}")
            print("------------- END RMSNorm.forward() ------------")
                          
        return output