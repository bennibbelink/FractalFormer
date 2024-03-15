from . import ff_config, utils
import torch.nn as nn
import torch
import numpy as np
from typing import Tuple

class MultiQueryAttention(nn.Module):
    """
    Implements Multi-Query Attention which supports a distinct number of attention heads for queries and key-values (KV).
    In the case where the same number of queries and key-values are used, this implemenation is equivalent to regular Multi-Head Attention.  
    """

    verbose = False
    
    def __init__(self, config: ff_config.Config):
        super().__init__()

        self.config = config

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # Determines the number of query heads associated with each KV head.
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.theta = config.rope_theta

        # Calculates the total size for all query projections.
        self.q_size = self.num_heads * self.head_dim
        # Calculates the total size for all key and value projections.
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Initialize our learnable matrices
        # the linear projection layer for queries, keys, and values
        # no real reason why we're creating one matrix instead of separate ones. cleaner model summary view?
        self.Wqkv = nn.Parameter(torch.Tensor(self.hidden_size,
                                              (self.num_heads + 2 * self.num_kv_heads) * self.head_dim))
        # the output projection layer, mapping the concatenated attention outputs back to the hidden size.
        self.Wo = nn.Parameter(torch.Tensor(self.num_heads * self.head_dim, self.hidden_size))
        
        # Initialize weights with uniform distribution
        # For qkv_proj, where in_features is hidden_size
        limit_Wqkv = 1 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.Wqkv, -limit_Wqkv, limit_Wqkv)
        # for o_proj, where in_features is self.num_heads * self.head_dim
        limit_Wo = 1 / np.sqrt(self.num_heads * self.head_dim)
        nn.init.uniform_(self.Wo, -limit_Wo, limit_Wo)
        
        # for our attention mask we'll use very large negative values to prevent attending to certain tokens
        mask_negatives = torch.full((1, 1, self.config.max_position_embeddings, self.config.max_position_embeddings),
                                 -2.3819763e38).to(torch.float)
        # then we'll replace the lower triangular ones with 0's to allow attention to see past tokens
        mask = torch.triu(mask_negatives, diagonal=1).to(config.device)
        # to define self.mask as a tensor that shouldn't undergo gradient descent
        self.register_buffer('mask', mask)
        
        # defining our dropout
        self.drop = nn.Dropout(config.dropout)

    def forwardTensor(self,
                      x: torch.Tensor,
                      model: int = 0,
                     ) -> torch.Tensor:
        """
        Inputs:
            x (torch.Tensor): Te input tensor to the attention mechanism.
                        shape (batch_size, input_len, hidden_size)
            model (int): the indicator of which model we're using. 
                        used in calculating our skip length for splicing. 
                        defaults to the equivalent of what's used in MatFormer+, meaning no skip, aka we use the top-left-most splice
        
        Returns:
            Tensor: The output tensor after applying the attention mechanism
        """
        if self.verbose: print("----------------- MultiQueryAttention.forwardTensor() --------------------")
        
        # Ensures the input tensor is 3-dimensional (batch_size, input_len, hidden_size).
        x_shape = x.shape
        assert len(x_shape) == 3
        if self.verbose: print(f"x shape: {x_shape}")

        # Extracts input sequence length and embedding dimension length from the hidden states tensor.
        batch_size, input_len, d_dim = x_shape
        
        # figuring out how we should do our splicing
        # first along the embedding dimension
        d_skip = model * d_dim  # the size of our skip along the model's embedding dimension
        if self.verbose: print(f"d_skip: {d_skip}")
        
        # then for splicing along the head sizes dimension
        index = self.config.model_dim_list.index(d_dim)
        models_in_this_level = self.config.model_count[index] # how many models are in this level
        h_dim = self.config.head_dim_list[index] # the head dimension size of this model in this level
        h_skip = model * h_dim # the size of our skip along the head dimension
        if self.verbose: 
            print(f"models_in_this_level: {models_in_this_level}")
            print(f"h_dim: {h_dim}")
            print(f"h_skip: {h_skip}")

        # Splits the Wqkv tensor into separate tensors for queries, keys, and values based on their respective sizes.
        if self.verbose: print(f"self.Wqkv: {self.Wqkv.shape}\n{self.Wqkv}")
        Wq, Wk, Wv = self.Wqkv.split([self.q_size,
                                      self.kv_size,
                                      self.kv_size],dim=-1)
        if self.verbose: 
            print(f"Wq: {Wq.shape}\n{Wq}")
            print(f"Wk: {Wk.shape}\n{Wk}")
            print(f"Wv: {Wv.shape}\n{Wv}")
        
        # splicing to get our correct weight matrices for each respective head
        # d_dim is relatively self-explanatory
        # i*self.head_dim is bc we initialized one single q, k, and v matrix for all heads so we have to
        # iterate through said matrix to get to the correct head
        Wq = torch.cat([Wq[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_heads)], dim=1)
        Wk = torch.cat([Wk[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_kv_heads)], dim=1)
        Wv = torch.cat([Wv[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_kv_heads)], dim=1)
        if self.verbose:
            print(f"Wq spliced: {Wq.shape}\n{Wq}")
            print(f"Wk spliced: {Wk.shape}\n{Wk}")
            print(f"Wv spliced: {Wv.shape}\n{Wv}")
        
        # this needs to be size (d_dim, (self.num_heads + 2 * self.num_kv_heads) * h_dim) aka (32,24)
        # recombine the spliced Wq Wk and Wv. Now they're the right size for matmul against x
        Wqkv_spliced = torch.cat((Wq, Wk, Wv), dim=-1)
        if self.verbose:
            print(f"Wqkv_spliced: {Wqkv_spliced.shape}\n{Wqkv_spliced}")
        

        # finally we can project x to get our queries, keys and values
        xqkv = x @ Wqkv_spliced
        if self.verbose: print(f"xqkv: {xqkv.shape}\n{xqkv}")
            
        # Splits the combined Xqkv tensor into separate tensors for queries (xq), keys (xk), and values (xv) based on their respective sizes.
        xq, xk, xv = xqkv.split([self.q_size // models_in_this_level,
                                 self.kv_size // models_in_this_level,
                                 self.kv_size // models_in_this_level],dim=-1)
        if self.verbose:
            print(f"xq: {xq.shape}\n{xq}")
            print(f"xk: {xk.shape}\n{xk}")
            print(f"xv: {xv.shape}\n{xv}")

        # Reshapes each of the Q, K, and V tensors to separate the heads and align the dimensions for attention operations.
        xq = xq.view(batch_size, input_len, self.num_heads, h_dim)#, self.head_dim)
        xk = xk.view(batch_size, input_len, self.num_kv_heads, h_dim)#, self.head_dim)
        xv = xv.view(batch_size, input_len, self.num_kv_heads, h_dim)#, self.head_dim)
        if self.verbose:
            print(f"xq reshaped: {xq.shape}\n{xq}")
            print(f"xk reshaped: {xk.shape}\n{xk}")
            print(f"xv reshaped: {xv.shape}\n{xv}")

        # Applies rotary positional embeddings to queries and keys to incorporate positional information.
        xq = utils.apply_rotary_emb(xq, h_dim, self.theta)#self.head_dim
        xk = utils.apply_rotary_emb(xk, h_dim, self.theta)#self.head_dim
        # is the differring head dimension going to mess with RoPE? Not sure
        if self.verbose:
            print(f"rotated xq: {xq.shape}\n{xq}")
            print(f"rotated xk: {xk.shape}\n{xk}")

        # If the number of KV heads is different from the number of query heads, adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_heads:
            # [batch_size, input_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)
            if self.verbose:
                print(f"repeat_interleaved xk: {xk.shape}\n{xk}")
                print(f"repeat_interleaved xv: {xv.shape}\n{xv}")

        # Transposes Q, K, and V tensors to align them for the batch matrix multiplication in attention calculation.
        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        if self.verbose:
            print(f"transposed xq: {q.shape}\n{q}")
            print(f"transposed xk: {k.shape}\n{k}")
            print(f"transposed xv: {v.shape}\n{v}")

        # Calculates attention scores by performing a batch matrix multiplication between queries and keys, followed by scaling.
        # [batch_size, n_local_heads, input_len, input_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * h_dim**-0.5#self.scaling
        if self.verbose: print(f"scores: {scores.shape}\n{scores}")
        
        # Applies the lower-triangular mask to the attention scores
        if self.verbose: print(f"mask: {self.mask[...,:input_len, :input_len].shape}\n{self.mask[...,:input_len, :input_len]}")
        scores = scores + self.mask[...,:input_len, :input_len] # make sure mask is the correct size. input_len <= max_seq_len
        if self.verbose: print(f"masked scores: {scores.shape}\n{scores}")

        # Applies softmax to the scores to obtain attention probabilities
        scores = torch.nn.functional.softmax(scores, dim=-1)
        if self.verbose: print(f"softmaxed scores: {scores.shape}\n{scores}")
        
        # Computes the weighted sum of values based on the attention scores to obtain the output of the attention mechanism.
        # [batch_size, n_local_heads, input_len, head_dim]
        attention = torch.matmul(scores, v)
        if self.verbose: print(f"attention: {attention.shape}\n{attention}")

        # Reshapes the attention output to match the expected output dimensions, combining the heads back into the hidden dimension.
        # [batch_size, input_len, hidden_dim]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        if self.verbose: print(f"reshaped attention: {attention.shape}\n{attention}")

        # Splice the output projection
        Wo = torch.cat([self.Wo[i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim,\
                                d_skip:d_skip + d_dim,\
                               ] for i in range(self.num_heads)], dim=0)
        if self.verbose: 
            print(f"self.Wo: {self.Wo.shape}\n{self.Wo}")
            print(f"spliced Wo: {Wo.shape}\n{Wo}")
            
        # Applies the final linear projection to the attention output, mapping it back to the hidden size dimension.
        output = attention @ Wo
        if self.verbose: 
            print(f"projected output: {output.shape}\n{output}")
            print("----------------- END MultiQueryAttention.forwardTensor() --------------------")
            
        return output

    def forwardTuple(self,
                     x: Tuple[Tuple[torch.Tensor]],
                     drop_bool: bool = True
                    ) -> torch.Tensor:
        """
        Defines the forward pass of the Attention module during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the MQA mechanism
        """
        if self.verbose: 
            print("------------- MultiQueryAttention.forwardTuple() ------------")
            print(f"x: {x}")
            
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

                output = self.forwardTensor(x[i][j], model=j)
                if self.verbose: print(f"forwardTensor() output: {output.shape}\n{output}")
                
                out_lvl += (self.drop(output),) if drop_bool else (output,)
            
            out += (out_lvl,)
        
        if self.verbose:
            print(f"final output: {out}")
            print("------------- END MultiQueryAttention.forwardTuple() ------------")

        return out
        
    def forward(self, x, model=0, drop_bool = True):
        train = True if type(x) == tuple else False
        if self.verbose: print(f"---------- Attention Input: {'Tuple' if train else 'torch.Tensor'} ------------")
        return self.forwardTuple(x, drop_bool) if train else self.forwardTensor(x, model)