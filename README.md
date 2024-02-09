# FractalFormer
the idea branches off from [MATFORMER](https://arxiv.org/pdf/2310.07707.pdf) which is a model that creates a russian nesting doll type structure along the embedding dimension and FFWD layer of a transformer such that when you train the model, you're also training a bunch of smaller models nested inside of it that you can just splice out if your compute needs limit you from using the full version. As a good primer before getting into this repo, i recommend you check out the predecessor paper to matformer called [matryoshka representation learning](https://arxiv.org/abs/2205.13147), the [github repo](https://github.com/evintunador/matryoshkaGPT) where i create a simple implementation of matryoshka embeddings and a MatFormer, and my corresponding [youtube video](https://youtu.be/dUeM_yDuGbg) explaining the concepts

the goal in this repo is to scale `matryoshkaGPT.ipynb` from the aforementioned repo up to what i'll call `FractalFormer.ipynb`. The idea here is that instead of 1 russian nesting doll of each size, when you open up a given russian nesting doll you find inside of it two side-by-side russian nesting dolls that are each half the size. maybe a quarter or an eighth or something instead of a half, but you get the point. Let's say we have one primary size, the inner layer to that is 4 matryoshkaGPT models, each 1/4 of the original's size. then the inner layer to that is divided into quarters again, giving us a total of 16 matryoshka models at that third layer, each 1/16 size of the original. 

Then while training, let's say we use a batch size of 16 which the primary model experiences. The 4 models within it each only experience 4 of the sequences from the batch, aka a minibatch but i'm changing the way that term is usually used, and the 16 models at the next level each only get trained on one sequence from the batch. 

Furthermore, if we split the batch up into data streams that have different characteristics, then we can essentially create a weird fractal-style MOE model
    - i think i may be able to implement the emergent groups idea from `emergent_hierarchical_embeddings_GPT.ipynb` to somehow help with interpretability & this MOE idea; not sure how yet though
    - if it works, i also imagine this being huge for my hierarchical conversational swarm intelligence idea that i've discussed in may youtube videos

i think if i do this right then it will also be very easy to take a given model and combine it with 4 others to create a super-model. this would have implications for training huge LLMs because rather than having to start from scratch, you'd be able to just use models that have already been made and concatenate then continually train them. when we do combine models, my first guess is that we might only need to add linear layers as the glue. if i'm right, that'd be big because the beauty there is that we can set these few linear layers to be the only parameters that train, thus making the big model training really easy during the connection stage. and because if it really does work with just linear layers, those can just be composed back into the models they're touching as long as we connect them between linear layers because the composition of two linear layers is just a linear layer. 

