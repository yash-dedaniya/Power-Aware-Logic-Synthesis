import torch
import random
import math
import abc_oracle
from gnn_model import BullsEyePredictor

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
YOUR_VERILOG_FILE = "c1355.v"  

MAX_QUERIES = 5000       
INITIAL_TEMP = 100.0     
COOLING_RATE = 0.99      
TIMING_SLACK = 1.02      
TOP_K_COUNT = 10         # NEW: Keep the top 10 best recipes to filter out hallucinations

VOCAB = {
    "refactor": 1, "refactor -z": 2, 
    "rewrite": 3, "rewrite -z": 4, 
    "resub": 5, "resub -z": 6, 
    "balance": 7
}
ABC_COMMANDS = list(VOCAB.keys())

def encode_recipe(recipe_str):
    cmds = recipe_str.split('; ')
    nums = [VOCAB.get(cmd, 0) for cmd in cmds]
    while len(nums) < 20: nums.append(0)
    return torch.tensor(nums, dtype=torch.long).unsqueeze(0)

def generate_random_recipe():
    return "; ".join(random.choices(ABC_COMMANDS, k=20))

def get_neighbor_recipe(current_recipe_str):
    cmds = current_recipe_str.split('; ')
    mutate_idx = random.randint(0, 19)
    possible_cmds = [cmd for cmd in ABC_COMMANDS if cmd != cmds[mutate_idx]]
    cmds[mutate_idx] = random.choice(possible_cmds)
    return "; ".join(cmds)

# ==========================================
# 🚀 THE POWER-AWARE SA ENGINE
# ==========================================
def run_simulated_annealing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==================================================")
    print("🚀 INITIATING POWER-AWARE SIMULATED ANNEALING 🚀")
    print("==================================================\n")

    print("1. Measuring the physical limits of the Verilog file...")
    base_power, base_area, base_delay = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, "print_stats")
    max_allowed_delay = base_delay * TIMING_SLACK
    print(f"   -> Baseline Delay: {base_delay:.2f} ps")
    print(f"   -> Dynamic Timing Shield: {max_allowed_delay:.2f} ps\n")

    print("2. Loading Calibrated Physics Engine...")
    model = BullsEyePredictor().to(device)
    try:
        checkpoint = torch.load("tuned_model.pth", map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
    except Exception:
        print("❌ ERROR: 'tuned_model.pth' not found.")
        return
    model.eval()

    graph = abc_oracle.extract_initial_graph(YOUR_VERILOG_FILE).to(device)
    num_nodes = graph.x.shape[0]
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

    def calculate_energy(recipe_str):
        recipe_tensor = encode_recipe(recipe_str).to(device)
        with torch.no_grad():
            predictions = model(graph.x, graph.edge_index, dummy_batch, recipe_tensor)
            pred_p_node = predictions[0, 0].item()
            pred_delay = predictions[0, 2].item()
            if pred_delay > max_allowed_delay:
                return float('inf') 
            return pred_p_node * num_nodes

    print(f"3. Searching {MAX_QUERIES} combinations...")
    T = INITIAL_TEMP
    
    best_recipe = generate_random_recipe()
    best_energy = calculate_energy(best_recipe)
    attempts = 0
    while best_energy == float('inf') and attempts < 100:
        best_recipe = generate_random_recipe()
        best_energy = calculate_energy(best_recipe)
        attempts += 1
        
    if best_energy == float('inf'):
        best_recipe = "print_stats"
        best_energy = calculate_energy(best_recipe)

    current_recipe = best_recipe
    current_energy = best_energy
    
    # NEW: The Top-K Memory Bank
    top_k_recipes = [(best_energy, best_recipe)]

    for q in range(MAX_QUERIES):
        neighbor_recipe = get_neighbor_recipe(current_recipe)
        neighbor_energy = calculate_energy(neighbor_recipe)
        delta_e = neighbor_energy - current_energy
        safe_T = max(T, 1e-5) 
        
        if delta_e < 0 or (neighbor_energy != float('inf') and random.random() < math.exp(-delta_e / safe_T)):
            current_recipe = neighbor_recipe
            current_energy = neighbor_energy
            
            # Save to Top-K if it's a valid, unique sequence
            if current_energy != float('inf') and current_recipe not in [r for e, r in top_k_recipes]:
                top_k_recipes.append((current_energy, current_recipe))
                top_k_recipes = sorted(top_k_recipes, key=lambda x: x[0])[:TOP_K_COUNT]
                
        T = T * COOLING_RATE
        if q % 1000 == 0:
            print(f"   Query {q}/5000 | Best AI Power: {top_k_recipes[0][0]:.2f} uW")

    # ---------------------------------------------------------
    # STEP 4: PHYSICAL TOP-K VERIFICATION
    # ---------------------------------------------------------
    # print("\n==================================================")
    # print("🛡️ INITIATING PHYSICAL TOP-K VERIFICATION 🛡️")
    # print("==================================================")
    # print("Simulating the AI's Top 10 favorite recipes to filter out hallucinations...\n")

    true_best_power = float('inf')
    true_best_delay = 0.0
    true_best_recipe = ""

    for i, (ai_pred, rec) in enumerate(top_k_recipes):
        p, a, d = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, rec)
        status = "❌ Hallucination" if p > base_power else f"✅ Verified: {p:.2f} uW"
        # print(f"   Candidate {i+1}: AI Guessed {ai_pred:.1f} uW -> {status}")
        
        if p < true_best_power and d <= max_allowed_delay:
            true_best_power = p
            true_best_delay = d
            true_best_recipe = rec

    print("\n==================================================")
    print("🏆 FINAL DOMINANCE LEADERBOARD 🏆")
    print("==================================================")
    
    exact_resyn2 = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance"
    exact_resyn2rs = "balance; resub; rewrite; refactor; balance; resub; rewrite; rewrite -z; balance; resub -z; refactor -z; rewrite -z; balance"
    
    std_resyn2_p, _, std_resyn2_d = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, exact_resyn2)
    std_resyn2rs_p, _, std_resyn2rs_d = abc_oracle.simulate_recipe(YOUR_VERILOG_FILE, exact_resyn2rs)

    print(f"{'RECIPE':<20} | {'TRUE POWER (uW)':<15} | {'TRUE DELAY (ps)':<15} | {'STATUS'}")
    print("-" * 75)
    print(f"{'resyn2 (Industry)':<20} | {std_resyn2_p:<15.2f} | {std_resyn2_d:<15.2f} | Standard")
    print(f"{'resyn2rs (Heavy)':<20} | {std_resyn2rs_p:<15.2f} | {std_resyn2rs_d:<15.2f} | Standard")
    print("-" * 75)
    print(f"{'AI Bulls-Eye (True)':<20} | {true_best_power:<15.2f} | {true_best_delay:<15.2f} | 🌟 CHAMPION")
    print("-" * 75)

    if true_best_power < std_resyn2_p:
        print(f"\n🎉 WIN: The AI beat `resyn2` by {((std_resyn2_p - true_best_power) / std_resyn2_p) * 100:.2f}%!")
        print(f"   -> Winning Sequence: {true_best_recipe}")
    else:
        print("\n✅ Run successful, but standard scripts remain highly competitive.")
        print(f"   -> Best Sequence Found: {true_best_recipe}")

if __name__ == "__main__":
    run_simulated_annealing()