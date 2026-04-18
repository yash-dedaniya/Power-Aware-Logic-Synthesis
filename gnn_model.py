import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm

class BullsEyePredictor(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=8, recipe_length=20):
        super(BullsEyePredictor, self).__init__()
        
        # ==========================================
        # LOBE 1: THE AIG EMBEDDING NETWORK (Fig 1b)
        # ==========================================
        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = GCNConv(in_channels=64, out_channels=64)
        self.bn2 = BatchNorm(64)
        
        # ==========================================
        # LOBE 2: THE SYNTHESIS RECIPE EMBEDDING (Fig 1a)
        # ==========================================
        self.recipe_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=16, kernel_size=3, padding=1)
        self.recipe_fc = nn.Linear(in_features=16 * recipe_length, out_features=64)
        
        # ==========================================
        # LOBE 3: THE GRAPH-BASED REGRESSION
        # ==========================================
        self.mlp = nn.Sequential(
            nn.Linear(in_features=192, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            # Output: [Power_per_node, Area_per_node, Absolute_Delay]
            nn.Linear(in_features=64, out_features=3) 
        )

        # Apply robust weight initialization to prevent NaN loss on Epoch 1
        self._init_weights()

    def _init_weights(self):
        """Ensures stable mathematical starting points for all Linear layers."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.recipe_fc.weight)
        nn.init.zeros_(self.recipe_fc.bias)

    def forward(self, x, edge_index, batch, recipe):
        # --- 1. PROCESS THE GRAPH ---
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        
        # The Bulls-Eye Pooling Strategy
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1) 
        
        # --- 2. PROCESS THE RECIPE ---
        r = self.recipe_embedding(recipe)
        r = r.transpose(1, 2) 
        r = self.conv1d(r)
        r = F.relu(r) 
        
        r_flat = torch.flatten(r, start_dim=1)
        r_vec = F.relu(self.recipe_fc(r_flat)) 
        
        # --- 3. THE PREDICTION ---
        combined_context = torch.cat([h_graph, r_vec], dim=1) 
        predictions = self.mlp(combined_context)
        
        return predictions