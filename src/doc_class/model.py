import torch.nn as nn
import torch

class MultilayerPerceptron(nn.Module):

    def __init__(self, vocab_size, embedding_weights, embedding_dim, hidden_dim, n_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #Load pre-trained embeddings and dont update them
        self.embedding.weight.data.copy_(embedding_weights)
        self.embedding.weight.requires_grad = False
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, n_tags)
               
    def forward(self, model_input):
        input_embeddings = self.embedding(model_input)
        context_representation = torch.mean(input_embeddings, 1)
        output = self.linear2(self.relu1(self.linear1(context_representation)))
        return output
    
