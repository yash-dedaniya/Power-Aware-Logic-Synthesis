import torch
import random
import abc_oracle
from gnn_model import BullsEyePredictor

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
YOUR_VERILOG_FILE = "c17.v"  # Must match the Verilog file you tuned on!

VOCAB = {
    "refactor": 1, "refactor -z": 2, 
    "rewrite": 3, "rewrite -z": 4, 
    "resub": 5, "resub -z": 6, 
    "balance": 7
}
ABC_COMMANDS = list(VOCAB.keys())

def encode_recipe(recipe_str):
    """Exactly mirrors your dataset logic."""
    cmds = recipe_str.split('; ')
    nums = [VOCAB.get(cmd, 0) for cmd in cmds]
    while len(nums) < 20:
        nums.append(0)
    return torch.tensor(nums, dtype=torch.long).unsqueeze(0)

def generate_random_recipe():
    """Generates a completely unseen 20-step sequence."""
    return "; ".join(random.choices(ABC_COMMANDS, k=20))

# ==========================================
# 🧪 THE BLIND EVALUATOR
# ==========================================
def run_blind_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==================================================")
    print("🧪 INITIATING TRUE BLIND VALIDATION (UNSEEN DATA) 🧪")
    print("==================================================\n")

    # 1. Generate an unseen sequence
    test_recipe = generate_random_recipe()
    print(f"1. Generated completely random, unseen recipe:\n   -> {test_recipe}\n")

    # 2. Get Ground Truth from ABC Oracle
    print("2. Running physical Berkeley ABC Simulation (The Oracle)...")
    try:
        true_power, true_area, true_delay = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, test_recipe)
    except Exception as e:
        print(f"❌ Oracle failed: {e}")
        return

    # 3. Get Prediction from Calibrated AI
    print("3. Consulting the Calibrated Neural Physics Engine...")
    graph = abc_oracle.extract_initial_graph(YOUR_VERILOG_FILE).to(device)
    num_nodes = graph.x.shape[0]
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

    model = BullsEyePredictor().to(device)
    try:
        checkpoint = torch.load("tuned_model.pth", map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("❌ ERROR: 'tuned_model.pth' not found. Did Step 2 finish successfully?")
        return
        
    model.eval() # Lock into inference mode
    recipe_tensor = encode_recipe(test_recipe).to(device)

    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index, dummy_batch, recipe_tensor)
        
        # Extract the Per-Node predictions
        pred_p_node = predictions[0, 0].item()
        pred_a_node = predictions[0, 1].item()
        pred_delay = predictions[0, 2].item()

    # REVERSE THE SCALING: Multiply by num_nodes to get Absolute physics
    pred_power = pred_p_node * num_nodes
    pred_area = pred_a_node * num_nodes

    # 4. Calculate Mathematical Accuracy
    err_power = abs(pred_power - true_power) / true_power * 100
    err_area = abs(pred_area - true_area) / true_area * 100
    err_delay = abs(pred_delay - true_delay) / true_delay * 100

    print("\n==================================================")
    print("🎯 FINAL MULTI-OBJECTIVE VALIDATION RESULTS")
    print("==================================================")
    print(f"⚡ POWER:")
    print(f"   Oracle (Real): {true_power:.2f} uW")
    print(f"   AI Predicted:  {pred_power:.2f} uW")
    print(f"   Error Margin:  {err_power:.2f}%")
    
    print(f"\n📐 AREA:")
    print(f"   Oracle (Real): {true_area:.2f}")
    print(f"   AI Predicted:  {pred_area:.2f}")
    print(f"   Error Margin:  {err_area:.2f}%")
    
    print(f"\n⏱️ DELAY:")
    print(f"   Oracle (Real): {true_delay:.2f} ps")
    print(f"   AI Predicted:  {pred_delay:.2f} ps")
    print(f"   Error Margin:  {err_delay:.2f}%")
    print("==================================================")

    if err_power <= 15.0:
        print("\n🚀 SUCCESS! Absolute Power prediction accuracy is above your 85% threshold.")
        print("The Physics Engine is perfectly calibrated. We are ready to build the Phase 4 Search Algorithm.")
    else:
        print("\n⚠️ Power accuracy is slightly below the 85% target.")
        print("Recommendation: Run `step2_microtune.py` again but increase epochs from 250 to 500.")

if __name__ == "__main__":
    run_blind_validation()