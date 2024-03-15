import torch.nn as nn
import torch
from . import ff_config, mlp, rms_norm, multi_query_attention as mqa
from typing import Tuple

class Layer(nn.Module):
    """
    A decoder layer that integrates the MultiQueryAttention and MLP. It includes
    normalization steps both before and after the attention mechanism to stabilize and accelerate training.
    """
    verbose = False

    def __init__(self, config: ff_config.Config):
        super().__init__()

        self.config = config

        # Initializes the GemmaAttention mechanism with parameters from the config, enabling self-attention within the decoder layer.
        self.self_attn = mqa.MultiQueryAttention(config)
        
        # Initializes the GemmaMLP module, providing a non-linear transformation after the attention mechanism.
        self.mlp = mlp.MLP(
            # the hidden dimension of the model
            hidden_size = config.hidden_size,
            # the number of nodes in the center of the two feedforward layers
            intermediate_size = config.intermediate_size,
            # the % of neurons to set to 0 during training
            dropout = config.dropout,
        )
        
        # Applies RMSNorm normalization to the input of the decoder layer for stable training dynamics.
        self.input_layernorm = rms_norm.RMSNorm(config.hidden_size,
                                       eps = config.rms_norm_eps)
        
        # Applies RMSNorm after the attention mechanism and before the MLP to ensure the output is well-conditioned for further processing.
        self.post_attention_layernorm = rms_norm.RMSNorm(config.hidden_size,
                                                eps = config.rms_norm_eps)

    def forwardTensor(self,
                # The input tensor to the decoder layer. shape (batch_size, input_len, hidden_size)
                x: torch.Tensor,
                model: int = 0,
                drop_bool: bool = False
                ) -> torch.Tensor:
        if self.verbose: print("----------------- Layer.forwardTensor() --------------------")
        
        # Self Attention Block
        # Stores the original input for use as a residual connection, aiding in mitigating the vanishing gradient problem
        residual_connection = x
        # Normalizes the input before processing by the attention mechanism.
        x = self.input_layernorm(x, model)
        # Processes the normalized input through the GemmaAttention mechanism
        x = self.self_attn(x, model, drop_bool)
        # The aforementioned residual connection
        x = residual_connection + x
        if self.verbose: print(f"x in layer after MQA & resid connection and before MLP:\n{x}")

        # MLP Block
        # Again, stores the output of the attention block for use as a residual connection before processing by the MLP.
        residual_connection = x
        # Normalizes the output of the attention block before passing it to the MLP, ensuring a stable input distribution.
        x = self.post_attention_layernorm(x, model)
        # Transforms the normalized attention output through the MLP, introducing additional non-linearity and capacity to the model.
        x = self.mlp(x, model, drop_bool)
        # Another residual connection
        x = residual_connection + x
        if self.verbose: 
            print(f"layer's final residual state:\n{x}")
            print("----------------- END Layer.forwardTensor() --------------------")

        return x

    def forwardTuple(self,
                     x: Tuple[Tuple[torch.Tensor]],
                    ) -> torch.Tensor:
        """
        Defines the forward pass of a decoder layer during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the decoder layer
        """
        if self.verbose: 
            print("------------- Layer.forwardTuple() ------------")
            print(f"x:\n{x}")
            
        # forwardTuple() should only be used during training, so we assert input_len == max_position_embeddings
        input_len = x[0][0].shape[1]
        if self.verbose: print(f"input_len: {input_len}")
        assert input_len == self.config.max_position_embeddings

        # we could define these from the config but this way the method is more flexible to testing
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]
        if self.verbose: 
            print(f"num_levels: {num_levels}")
            print(f"models_per_level: {models_per_level}")

        # the loop that iterates over levels, aka the different potential sizes of models
        out = ()
        for i in range(num_levels):
            if self.verbose: print(f"Level {i} from range({num_levels})")

            # now for the loop that iterates over models in this level
            out_lvl = ()
            for j in range(models_per_level[i]):
                if self.verbose: print(f"Model {j} from range({models_per_level[i]})")

                output = self.forwardTensor(x[i][j], model = j, drop_bool = True)
                if self.verbose: print(f"forwardTensor() output: {output.shape}\n{output}")
                
                out_lvl += (output,)
            
            out += (out_lvl,)
        
        if self.verbose:
            print(f"final output: {out}")
            print("------------- END Layer.forwardTuple() ------------")

        return out
        
    def forward(self, x, model=0):
        train = True if type(x) == tuple else False
        if self.verbose: print(f"---------- Layer Input: {'Tuple' if train else 'torch.Tensor'} ------------")
        return self.forwardTuple(x) if train else self.forwardTensor(x, model)