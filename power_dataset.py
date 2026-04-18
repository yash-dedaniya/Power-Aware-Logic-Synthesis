import os
import pandas as pd
import torch
from torch_geometric.data import Dataset
import zipfile
import io

class CircuitPowerDataset(Dataset):
    def __init__(self, csv_file, zip_path):
        # 1. Setup: Load the CSV map and define the ZIP file
        self.df = pd.read_csv(csv_file)
        self.zip_path = zip_path
        self.archive = None # We will open the zip file later
        
        self.vocab = {
            "refactor": 1, "refactor -z": 2, 
            "rewrite": 3, "rewrite -z": 4, 
            "resub": 5, "resub -z": 6, 
            "balance": 7
        }

    def __len__(self):
        return len(self.df) * 20

    def __getitem__(self, idx):
        # Open the zip archive once and keep it open for speed
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.zip_path, 'r')
            
        row_idx = idx // 20
        step_idx = idx % 20
        row = self.df.iloc[row_idx]
        
        circuit_name = row['Circuit']
        run_id = row['Run_ID']
        graph_name = f"{circuit_name}_run{run_id}_step{step_idx}.pt"
        
        # Internal zip path
        internal_zip_path = f"dataset/pytorch_graphs/{graph_name}"
        
        # --- STREAM DIRECTLY FROM THE ZIP INTO RAM ---
        with self.archive.open(internal_zip_path) as f:
            graph_data = torch.load(io.BytesIO(f.read()), weights_only=False)
        
        # --- DIGEST THE RECIPE ---
        full_recipe = row['Recipe'].split('; ')
        remaining_recipe = full_recipe[step_idx:]
        
        recipe_numbers = [self.vocab.get(cmd, 0) for cmd in remaining_recipe]
        while len(recipe_numbers) < 20:
            recipe_numbers.append(0)
            
        recipe_tensor = torch.tensor(recipe_numbers, dtype=torch.long)
        
        # ==========================================
        # BULLS-EYE INTENSIVE TARGET FIX (Scale Collapse Prevention)
        # ==========================================
        # 1. Get the exact size of the current circuit
        num_nodes = graph_data.x.shape[0]
        
        # 2. Divide absolute targets by the node count to get "Per-Node" metrics
        # This makes the AI immune to the overall size of the circuit
        power_per_node = row['Power'] / num_nodes
        area_per_node = row['Area'] / num_nodes
        
        # 3. Attach the normalized targets
        graph_data.y_power = torch.tensor([power_per_node], dtype=torch.float)
        graph_data.y_area = torch.tensor([area_per_node], dtype=torch.float)
        
        # Note: Delay is based on critical path depth, not total nodes. 
        # Therefore, Delay remains exactly as it is.
        graph_data.y_delay = torch.tensor([row['Delay']], dtype=torch.float)
        
        return graph_data, recipe_tensor