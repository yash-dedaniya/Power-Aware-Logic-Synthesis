import os
import torch
from torch_geometric.data import Data

# Configuration
BENCH_DIR = "dataset/bench_files"
OUTPUT_PT_DIR = "dataset/pytorch_graphs"

def parse_bench_to_pyg(bench_filepath):
    # [THIS FUNCTION REMAINS EXACTLY THE SAME AS BEFORE]
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

def process_all_bench_files():
    os.makedirs(OUTPUT_PT_DIR, exist_ok=True)
    bench_files = [f for f in os.listdir(BENCH_DIR) if f.endswith('.bench')]
    
    print(f"Found {len(bench_files)} .bench files remaining to process.")
    
    for count, filename in enumerate(bench_files):
        bench_path = os.path.join(BENCH_DIR, filename)
        pt_filename = filename.replace('.bench', '.pt')
        pt_path = os.path.join(OUTPUT_PT_DIR, pt_filename)
        
        # --- NEW RESUME & HEAL LOGIC ---
        if os.path.exists(pt_path):
            # Check if file is suspiciously small (under 1KB) indicating corruption
            if os.path.getsize(pt_path) < 1024:
                print(f"Deleting corrupted file: {pt_filename}")
                os.remove(pt_path)
            else:
                # File is healthy. We can delete the raw .bench to free up space!
                if os.path.exists(bench_path):
                    os.remove(bench_path)
                continue
        # --------------------------------
            
        try:
            # Parse the graph
            graph_data = parse_bench_to_pyg(bench_path)
            # Save the PyTorch tensor
            torch.save(graph_data, pt_path)
            
            # --- DYNAMIC SPACE SAVING ---
            # Now that it's safely saved as a .pt, delete the raw text file
            os.remove(bench_path)
            # ----------------------------
            
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            
        if (count + 1) % 1000 == 0:
            print(f"Processed {count + 1} graphs...")
            
    print("Graph conversion complete!")

if __name__ == "__main__":
    process_all_bench_files()