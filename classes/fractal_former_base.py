import torch.nn as nn
import torch
from . import ff_config, simple_tokenizer, rms_norm, layer, output_layer, fractal_loss

class FractalFormer_base(nn.Module):
    verbose = False
    def __init__(self, config: ff_config.Config, tokenizer: simple_tokenizer.SimpleTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # hyperparameters
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size

        ### FractalFormer-specific hyperparameters
        self.num_levels = config.levels # the number of levels for sub-models to exist on
        self.split = config.split # the number of splits to make at a given level
        self.model_count = config.model_count # list of number of models at a given level
        self.model_dim_list = config.model_dim_list # list of hidden dimensions corresponding to each given level
        self.head_dim_list = config.head_dim_list # list of attention head dimensions corresponding to each given level    

        # the embedding matrix. for converting tokens to the first residual state, and the last residual state to logits
        self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)

        # for normalizing the initial embeddings
        self.embedder_norm = rms_norm.RMSNorm(config.hidden_size)

        # Initialize a sequence of DecoderLayer instances as specified by the number of hidden layers in the config
        self.layers = nn.ModuleList(layer.Layer(config) for _ in range(config.num_hidden_layers))

        # initializing output layer
        self.output_layer = output_layer.OutputLayer(self.embedder.weight, config)
        # i think i need to do this bc in the above version you can't use `self.` inside the init
        #@property 
        #def output_layer(self):
            #return OutputLayer(self.embedder.weight, config)

        # the loss function
        self.criterion = fractal_loss.FractalLoss(config)

    def forwardTensor(self,
                      input_token_ids: torch.Tensor,
                      level: int = 0, # integer designating the level of model to use. 0 is largest model, -1 is smallest
                      model: int = 0, # integer designating the model in that level to use. 0 is top-left, -1 is bottom right
                     ) -> torch.Tensor:
        """
        inputs: 
            - input_token_ids (torch.Tensor): a tensor of integers size (batch_size, sequence_length)
            - level: integer designating the level of model to use. 0 is largest model, -1 is smallest
            - model: integer designating the model in that level to use. 0 is top-left, -1 is bottom right
        output: a torch.Tensor shape (batch_size, sequence_length, vocab_size)
        """
        if self.verbose: 
            print("------------- FractalFormer.forwardTensor() ------------")
            print(f"input_token_ids: {input_token_ids.shape}\n{input_token_ids}")
        
        # adjusting everything to the specified level & model
        d_dim = self.hidden_size // (2**level)
        d_skip = model * d_dim
        if self.verbose:
            print(f"d_dim: {d_dim}")
            print(f"d_skip: {d_skip}")
        
        # turn the input tokens into the first residual state using the embedding matrix
        # (batch_size, input_len) & (vocab_size, hidden_size) -> (batch_size, input_len, hidden_size) -> (batch_size, input_len, d_dim)
        x = self.embedder(input_token_ids)
        if self.verbose: print(f"x0: {x.shape}\n{x}")

        x = x[:,:, d_skip:d_skip + d_dim]
        # if self.verbose: print(f"spliced x0: {x0.shape}\n{x0}")
        
        # Gemma normalizes the embedding by sqrt(hidden_size)
        # the question is, should I do this with the full sized hidden_size or do it at the splice size????
        # imma do it at the splice size and change it later if i think the models aren't learning well
        #x = x * (d_dim**0.5)
        # alternatively i could just switch to doing a regular RMSNorm which would be more like me
        # if i figure out this different sizes of hyperspheres thing it'd be more in line with that
        x = self.embedder_norm(x, model)
        if self.verbose: print(f"normalized initial x: {x.shape}\n{x}")

        # Iteratively process the input through each Layer
        for i, layer in enumerate(self.layers):
            if self.verbose: print(f"begin layer {i}")
            x = layer(x, model)
            if self.verbose: print(f"output of layer {i}: {x.shape}\n{x}")

        logits = self.output_layer(x, model)
        if self.verbose: 
            print(f"output logits: {logits.shape}\n{logits}")
            print("------------- END FractalFormer.forwardTensor() ------------")

        return logits

    def forwardTuple(self,
                     input_token_ids: torch.Tensor,
                     target_token_ids: torch.Tensor,
                    ) -> torch.Tensor:
        if self.verbose: 
            print("------------- FractalFormer.forwardTuple() ------------")
            print(f"input_token_ids: {input_token_ids.shape}\n{input_token_ids}")
            print(f"target_token_ids: {target_token_ids.shape}\n{target_token_ids}")
        
        # use the embedding matrix to turn the input tokens into the first residual state of the largest model
        # (batch_size, input_len) & (vocab_size, hidden_size) -> (batch_size, input_len, hidden_size)
        x0 = self.embedder(input_token_ids)
        if self.verbose: print(f"initial x: {x.shape}\n{x}")

        # create the first fractal tuple of residual states
        x = ()
        for i, models_in_level in enumerate(self.config.model_count):
            if self.verbose: print(f"i: {i}, models_in_level: {models_in_level}, iterating over {self.config.model_count}")
            
            x_lvl = ()
            for j, d_dim in enumerate(self.config.model_dim_list):
                if self.verbose: print(f"j: {j}, d_dim: {d_dim}, iterating over {self.config.model_dim_list}")

                skip = j * d_dim
                if self.verbose: print(f"skip: {skip}")
                
                x_ij_spliced = x0[:,:,skip:skip + d_dim]
                if self.verbose: print(f"initial x[{i}][{j}] spliced: {x_ij_spliced.shape}\n{x_ij_spliced}")
                    
                x_ij_spliced_normed = self.embedder_norm(x_ij_spliced, model=j) # * (d_dim**0.5) # if i want to do Gemma normalization instead
                if self.verbose: print(f"initial x[{i}][{j}] spliced & normed: {x_ij_spliced_normed.shape}\n{x_ij_spliced_normed}")
                
                x_lvl += (x_ij_spliced_normed,)  
            x += (x_lvl,)
        if self.verbose: print(f"full tuple initial x: {x0}")

        # Iteratively process the input through each Layer
        for i, layer in enumerate(self.layers):
            if self.verbose: print(f"begin layer {i}")
            
            x = layer(x)
            if self.verbose: print(f"output of layer {i}: {x}")

        logits = self.output_layer(x)
        if self.verbose: 
            print(f"output logits: {logits}")
            print("------------- END FractalFormer.forwardTuple() ------------")

        return logits

    def forward(self,
                input_token_ids: torch.Tensor, # a shape (batch_size, input_seq_len OR max_seq_len)list of integer token ids
                target_token_ids: torch.Tensor = None, # a shape (batch_size, max_seq_len) list of token ids to train on
                level: int = 0, # integer designating the level of model to use. 0 is largest model
                model: int = 0, # integer designating the model in that level to use. 0 is top-left model in level
                ):
        if self.verbose: 
            print("------------- FractalFormer.forward() ------------")
            print(f"input_token_ids: {input_token_ids.shape}\n{input_token_ids}")
            print(f"target_token_ids: {target_token_ids}")
            print(f"level: {level}")
            print(f"model: {model}")
        
        if target_token_ids is None: # if we're not training, then we don't need to calculate loss
            logits = self.forwardTensor(input_token_ids, level, model)
            loss = None
        else:
            # if we are training
            # training uses a tuple of tuples of tensors
            logits = self.forwardTuple(input_token_ids, target_token_ids) # -> Tuple[Tuple[Tensor shape (batch_size, max_seq_len, vocab_size)]]
            
            # custom Fractal CELoss function
            loss = self.criterion(logits, target_token_ids) 
        
        if self.verbose: 
            print(f"logits: {logits}")
            print(f"loss: {loss}")
            print("------------- END FractalFormer.forward() ------------")
        
        return logits, loss

    @torch.no_grad() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions from Gemma's output.
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling 
        The class operates as follows:
    
        1. Selects the last hidden state for each sequence in the batch
    
        2. Computes logits by multiplying the selected hidden states with the transposed embedding matrix. 
    
        3. Temperature is used to scale the logits, making the distribution over tokens sharper (lower temperature) 
        or flatter (higher temperature), which affects the randomness of the sampling (flatter -> more random)
    
        4. The softmax function is applied to the scaled logits to obtain a probability distribution over the vocabulary.
    
        5. For top-p sampling, the function computes the cumulative sum of the sorted probabilities and masks out tokens until the 
        cumulative probability exceeds the threshold defined by `top_ps`. This allows the model to focus on a subset of the most 
        probable tokens while ignoring the long tail of less likely tokens. 
        We to ignore long tail probabilities to avoid nonsensical output
    
        7. For top-k sampling, the function masks out all tokens except the `k` most likely ones, as specified by `top_ks`. 
        This ensures that the model only considers a fixed number of the most probable tokens for the next token prediction.
    
        8. After applying both the top-p and top-k masks, the probabilities are re-normalized so that they sum up to 1
    
        9. The function then samples from the re-normalized probability distribution to select the next token. 
        """
        if self.config.verbose['Sampler']:
            print("----------------- FractalFormer.Sampler() --------------")
            print(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
            
        # Select the last element for each sequence.
        # (batch_size, input_len, vocab_size) -> (batch_size, vocab_size)
        logits = logits[:,-1,:]
        if self.config.verbose['Sampler']: print(f"logits: {logits.shape}\n{logits}")
        
        # Apply temperature scaling
        # (batch_size, vocab_size) / float -> (batch_size, vocab_size)
        logits.clone().div_(temperature) # the clone() is because i didn't properly prevent gradient tracking and i'm too lazy to fix the issue at its cause
        if self.config.verbose['Sampler']: print(f"logits w temperature: {logits.shape}\n{logits}")

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along
        if self.config.verbose['Sampler']: print(f"probs: {probs.shape}\n{probs}")

        # sort the probabilities to for use in top-p & top-k
        # both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # probs_sort contains float probabilities while probs_idx contains integer indices
        if self.config.verbose['Sampler']: 
            print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")
            print(f"probs_idx: {probs_idx.shape}\n{probs_idx}")

        # calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        if self.config.verbose['Sampler']: print(f"probs_sum: {probs_sum.shape}\n{probs_sum}")
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        if self.config.verbose['Sampler']: print(f"top_ps_mask: {top_ps_mask.shape}\n{top_ps_mask}")
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 
        if self.config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")

        # calculating top_k
        # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) 
        if self.config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")
        # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        # "expand" means copy the original into this new size, so each length vocab_size row is the same
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        if self.config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")
        # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks
        top_ks_mask = top_ks_mask >= top_k
        if self.config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        # this trims probs_sort to also fit within our top_k requirement
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        if self.config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")

        # Re-normalization so that total probabilities add up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        if self.config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")
        
        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))
        if self.config.verbose['Sampler']: print(f"probs: {probs.shape}\n{probs}")
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        if self.config.verbose['Sampler']: print(f"next_token_id: {next_token_id.shape}\n{next_token_id}")
        
        return next_token_id # returns the predicted token
        
    def generate(
        self,
        prompt: str,
        output_len: int = 100, # the model will output 100 tokens
        temperature: float = 0.7, # 0.95 is pretty close to not even using temperature at all (1.0 would be no effect)
        top_p: float = 1.0, # defaulting to 1 means we essentially don't use top-p
        top_k: int = None, # setting top_k = vocab_size means we're effectively not using top_k at all
        level: int = 0, # which size model we want to perform inference with
        model: int = 0, # which model in that level we want to perform inference with
    ) -> str: 
        if top_k is None:
            self.config.vocab_size
        
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=self.config.device).unsqueeze(0)
        
        # we wouldn't want to go past the maximum context length we trained on
        assert len(tokens) + output_len <= self.config.max_position_embeddings

        for i in range(output_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits, _ = self(tokens[:,:self.max_seq_len], level=level, model=model)
            
            next_token = self.Sampler(
                logits = logits, # the actual output of the model
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )
            #print(next_token)

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output