import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, category_offsets):
        super(SimpleModel, self).__init__()
        self.category_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.category_offsets = torch.tensor(category_offsets, device='cuda')

    def forward(self, x_cat):
        adjusted_indices = x_cat.long() + self.category_offsets[None]
        print('Adjusted indices min, max:::', adjusted_indices.min(), adjusted_indices.max())
        problem_indices = (adjusted_indices >= 52).nonzero()
        print("Problem indices:", problem_indices)
        print("Problematic x_cat values:", x_cat[problem_indices[:, 0]].cpu().numpy())
        x = self.category_embeddings(adjusted_indices).view(x_cat.size(0), -1)
        return x

# Initialize model
num_embeddings = 5400  # This should be 52 to account for max x_cat value of 51


embedding_dim = 50
category_offsets = [2]  # Example offset, can be adjusted

model = SimpleModel(num_embeddings, embedding_dim, category_offsets).cuda()

# Generate random x_cat data
torch.manual_seed(42)
sample_size = 256
x_cat = torch.randint(0, 52, (sample_size, 1)).float().cuda()  # random values between 0 and 51

# Forward pass
output = model(x_cat)
print(output)
