import torch
from . import rms_norm, ff_config

class OutputLayer(torch.nn.Module):
    verbose = False

    def __init__(self, embedding: torch.Tensor, config: ff_config.Config):
        super().__init__()
        self.embedding = embedding
        self.config = config
        self.v = config.vocab_size
        self.model_dim_list = config.model_dim_list

        # applies RMSNorm to the embedding matrix
        self.embedding_norm = rms_norm.RMSNorm(config.hidden_size,
                                      eps = config.rms_norm_eps)
        
        # Applies RMSNorm to the model's final residual state before we use the embedding matrix to get logits
        self.final_norm = rms_norm.RMSNorm(config.hidden_size,
                                  eps = config.rms_norm_eps)

    def forwardTensor(self, x, model=0):
        if self.verbose: 
            print("------------- OutputLayer.forwardTensor() ------------")
            print(f"x: {x.shape}\n{x}")

        # setting up our splicing logic
        d_i = x.shape[-1]
        skip = model * d_i
        if self.verbose:
            print(f"d_i: {d_i}")
            print(f"skip: {skip}")
            print(f"embedding: {self.embedding.shape}\n{self.embedding}")

        # splice out our embedding matrix according to what model we're using
        sliced_embed = self.embedding[:,skip:skip + d_i]
        if self.verbose: print(f"sliced_embed: {sliced_embed.shape}\n{sliced_embed}")

        # normalize our sliced embedding matrix
        normed_sliced_embed = self.embedding_norm(sliced_embed)
        if self.verbose: print(f"normed & sliced embedding: {normed_sliced_embed.shape}\n{normed_sliced_embed}")

        # normalize the residual state before the final linear layer
        x = self.final_norm(x, model)
        if self.verbose: print(f"normed x: {x.shape}\n{x}")

        # calculating the final output logits of the model
        logits = x @ normed_sliced_embed.t()
        if self.verbose: 
            print(f"final logits: {logits.shape}\n{logits}")
            print("------------- END OutputLayer.forwardTensor() ------------")

        return logits

    def forwardTuple(self, x):
        """
        Defines the forward pass of the final embedding classification layer during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            output (Tuple[Tuple[Tensor]]): 
                The output tuple of tuples of tensors after applying the final embedding classification
        """
        if self.verbose: 
            print("------------- OutputLayer.forwardTuple() ------------")
            print(f"x:\n{x}")
            
        # forwardTuple() should only be used during training, so we assert input_len == max_position_embeddings
        assert type(x) == tuple
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

                output = self.forwardTensor(x[i][j], model = j)
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