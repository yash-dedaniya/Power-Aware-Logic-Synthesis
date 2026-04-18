import torch
import torch.nn as nn
import torch.optim as optim
import abc_oracle
import step1_anchors
from gnn_model import BullsEyePredictor

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
YOUR_VERILOG_FILE = "c1355.v"  # Make sure this matches your Verilog file!

VOCAB = {
    "refactor": 1, "refactor -z": 2, 
    "rewrite": 3, "rewrite -z": 4, 
    "resub": 5, "resub -z": 6, 
    "balance": 7
}

def encode_recipe(recipe_str):
    """Exactly mirrors your Phase 1 CircuitPowerDataset padding logic."""
    cmds = recipe_str.split('; ')
    nums = [VOCAB.get(cmd, 0) for cmd in cmds]
    while len(nums) < 20:
        nums.append(0)
    return torch.tensor(nums, dtype=torch.long).unsqueeze(0)

# ==========================================
# THE POWER-AWARE MICRO-TUNE
# ==========================================
def run_power_aware_microtune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"⚡ INITIATING MULTI-OBJECTIVE MICRO-TUNE ⚡")
    print(f"==================================================\n")

    # ---------------------------------------------------------
    # PHASE A: DYNAMIC GROUND TRUTH GENERATION
    # ---------------------------------------------------------
    print("1. Extracting PyTorch Graph from Verilog...")
    graph = abc_oracle.extract_initial_graph(YOUR_VERILOG_FILE).to(device)
    num_nodes = graph.x.shape[0]
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
    print(f"   -> Found {num_nodes} Nodes. Using this for Per-Node Normalization.")

    print("\n2. Generating 5 Anchor Recipes (K-Means Math Bypass)...")
    anchors = step1_anchors.generate_smart_anchors()

    print("\n3. Running Oracle Simulations to get Ground Truths...")
    target_tensors = []
    for i, recipe in enumerate(anchors):
        p_abs, a_abs, d_abs = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, recipe)
        
        # APPLY YOUR EXACT DATASET SCALING LOGIC
        p_per_node = p_abs / num_nodes
        a_per_node = a_abs / num_nodes
        
        # Target vector: [Power_per_node, Area_per_node, Absolute_Delay]
        target = torch.tensor([[p_per_node, a_per_node, d_abs]], dtype=torch.float32).to(device)
        target_tensors.append(target)
        print(f"   Anchor {i+1} Targets -> P_node: {p_per_node:.4f}, A_node: {a_per_node:.4f}, Delay: {d_abs:.1f}")

    # ---------------------------------------------------------
    # PHASE B: NEURAL PREPARATION
    # ---------------------------------------------------------
    print("\n4. Loading Base Predictor (best_model.pth)...")
    model = BullsEyePredictor().to(device)
    checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Apply Graph Shield (Protect the 50-hour base knowledge)
    for param in model.conv1.parameters(): param.requires_grad = False
    for param in model.conv2.parameters(): param.requires_grad = False
    for param in model.bn1.parameters(): param.requires_grad = False
    for param in model.bn2.parameters(): param.requires_grad = False

    # Neural Reset (Wipe the Dead ReLUs in the text reader)
    model.recipe_embedding.reset_parameters()
    model.conv1d.reset_parameters()
    model._init_weights()

    # Learning rate set slightly lower than Phase 1 to ensure smooth fine-tuning
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    criterion = nn.MSELoss()

    # ---------------------------------------------------------
    # PHASE C: GRADIENT-BALANCED TRAINING LOOP
    # ---------------------------------------------------------
    print("\n5. Beginning Multi-Objective Calibration...")
    model.train()
    
    for epoch in range(1, 251):
        optimizer.zero_grad()
        total_p_loss, total_a_loss, total_d_loss = 0, 0, 0
        
        for i in range(5):
            recipe_tensor = encode_recipe(anchors[i]).to(device)
            target = target_tensors[i] # Shape: [1, 3]
            
            # Forward pass
            predictions = model(graph.x, graph.edge_index, dummy_batch, recipe_tensor)
            
            # Extract individual metrics (Shape: [1])
            pred_p = predictions[:, 0]
            pred_a = predictions[:, 1]
            pred_d = predictions[:, 2]
            
            targ_p = target[:, 0]
            targ_a = target[:, 1]
            targ_d = target[:, 2]
            
            # Calculate individual errors
            loss_power = criterion(pred_p, targ_p)
            loss_area = criterion(pred_a, targ_a)
            loss_delay = criterion(pred_d, targ_d)

            # ---------------------------------------------------------
            # DYNAMIC GRADIENT BALANCING (The True Fix)
            # ---------------------------------------------------------
            # Delay is massive (e.g. 4000), Power_node is tiny (e.g. 5). 
            # We dynamically calculate the mathematical ratio between them for THIS specific circuit.
            ratio_p = (targ_d.detach() / (targ_p.detach() + 1e-8))
            ratio_a = (targ_d.detach() / (targ_a.detach() + 1e-8))
            
            # # EXACT LOSS FUNCTION FROM YOUR TRAIN_MODEL.PY
            # batch_loss = (1.0 * loss_power) + (0.1 * loss_area) + (0.1 * loss_delay)
            batch_loss = (ratio_p * loss_power) + (ratio_a * loss_area) + (1.0 * loss_delay)
            
            batch_loss.backward()
            
            total_p_loss += loss_power.item()
            total_a_loss += loss_area.item()
            total_d_loss += loss_delay.item()
            
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/250 | MSE -> P_node: {total_p_loss/5:.4f} | A_node: {total_a_loss/5:.4f} | Delay: {total_d_loss/5:.1f}")

    print("\n✅ Micro-Tune Complete! The Recipe Lobe is fully calibrated to this specific circuit.")
    torch.save(model.state_dict(), "tuned_model.pth")
    print("Saved as 'tuned_model.pth'.")

if __name__ == "__main__":
    run_power_aware_microtune()