import os
import re
import subprocess
import torch
from torch_geometric.data import Data

# ==========================================
# ⚙️ SYSTEM CONFIGURATION (UPDATE THESE!)
# ==========================================
ABC_BINARY_PATH = "abc/abc"             # e.g., "/home/komal/abc/abc"
LIB_PATH = "abc/NangateOpenCellLibrary_typical.lib"           # e.g., "/home/komal/libs/mcnc.lib"

def parse_bench_to_pyg(bench_filepath):
    """YOUR EXACT PARSER: Converts .bench to PyTorch Geometric Data."""
    node_to_id = {}
    node_features = []
    edges_src = []
    edges_dst = []
    edge_attrs = []
    current_node_id = 0
    
    def get_or_create_node(name, node_type):
        nonlocal current_node_id
        if name not in node_to_id:
            node_to_id[name] = current_node_id
            current_node_id += 1
            if node_type == "PI": node_features.append([1.0, 0.0, 0.0])
            elif node_type == "PO": node_features.append([0.0, 1.0, 0.0])
            else: node_features.append([0.0, 0.0, 1.0])
        return node_to_id[name]

    with open(bench_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line.startswith('INPUT'):
            name = line[line.find('(')+1 : line.find(')')]
            get_or_create_node(name, "PI")
        elif line.startswith('OUTPUT'):
            name = line[line.find('(')+1 : line.find(')')]
            get_or_create_node(name, "PO")
        elif '=' in line:
            dest, logic = [part.strip() for part in line.split('=')]
            get_or_create_node(dest, "AND")

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('INPUT') or line.startswith('OUTPUT'): continue
        if '=' in line:
            dest_name, logic = [part.strip() for part in line.split('=')]
            dest_id = node_to_id[dest_name]
            gate_type = logic[:logic.find('(')]
            inputs_str = logic[logic.find('(')+1 : logic.find(')')]
            input_names = [in_name.strip() for in_name in inputs_str.split(',')]
            
            for in_name in input_names:
                if in_name not in node_to_id: continue
                src_id = node_to_id[in_name]
                edges_src.append(src_id)
                edges_dst.append(dest_id)
                if gate_type == "NOT": edge_attrs.append([1.0])
                else: edge_attrs.append([0.0])

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def extract_initial_graph(verilog_filepath):
    """Converts a Verilog file into a PyTorch Geometric Graph automatically."""
    temp_bench = "temp_initial_graph.bench"
    
    # Run ABC: read_verilog -> strash (AIG) -> write_bench
    abc_command = f"read {verilog_filepath}; strash; write_bench {temp_bench}"
    subprocess.run([ABC_BINARY_PATH, '-c', abc_command], capture_output=True, text=True)
    
    if not os.path.exists(temp_bench):
        raise FileNotFoundError(f"ABC failed to generate .bench for {verilog_filepath}.")

    # Parse the generated bench file using your exact logic
    graph_data = parse_bench_to_pyg(temp_bench)
    
    os.remove(temp_bench) # Clean up
    return graph_data

def simulate_recipe(verilog_filepath, recipe_str):
    """Runs the recipe in ABC and extracts true Power, Area, and Delay."""
    abc_command = f"read {verilog_filepath}; strash; {recipe_str}; read_lib {LIB_PATH}; map; print_stats -p"
    
    result = subprocess.run([ABC_BINARY_PATH, '-c', abc_command], capture_output=True, text=True)
    output = result.stdout

   # ADDED: re.IGNORECASE to handle lowercase "area =", "delay =", "power ="
    area_match = re.search(r'area\s*=\s*([0-9.]+)', output, re.IGNORECASE)
    delay_match = re.search(r'delay\s*=\s*([0-9.]+)', output, re.IGNORECASE)
    power_match = re.search(r'power\s*=\s*([0-9.]+)', output, re.IGNORECASE)

    if not (area_match and delay_match and power_match):
        print(f"❌ ERROR: ABC Output Parse Failed. Output was:\n{output}")
        raise ValueError("Failed to extract Power, Area, and Delay.")

    return float(power_match.group(1)), float(area_match.group(1)), float(delay_match.group(1))