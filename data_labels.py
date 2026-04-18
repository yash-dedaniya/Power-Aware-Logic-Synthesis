import os
import re
import csv

# Configuration
LOG_DIR = "dataset/abc_logs"
OUTPUT_CSV = "dataset/labels.csv"

# Regex Patterns (Tailored exactly to your successful log output)
# Extracts the recipe string
recipe_pattern = re.compile(r"^RECIPE:\s*(.*)")

# Extracts area, delay, and power numbers from the stats line
# Example target: area =2338.21  delay =3066.81  lev = 102  power =3019.50
stats_pattern = re.compile(r"area\s*=\s*([0-9.]+)\s+delay\s*=\s*([0-9.]+).*power\s*=\s*([0-9.]+)")

def parse_all_logs():
    if not os.path.exists(LOG_DIR):
        print(f"Error: Directory {LOG_DIR} not found.")
        return

    # Prepare the data list
    dataset_rows = []
    
    # Iterate through all log files
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    print(f"Found {len(log_files)} log files. Parsing...")

    for filename in log_files:
        filepath = os.path.join(LOG_DIR, filename)
        
        # Extract circuit name and run ID from the filename (e.g., "max_run71.log")
        # Removing the ".log" extension and splitting by "_run"
        base_name = filename.replace(".log", "")
        parts = base_name.split("_run")
        
        if len(parts) != 2:
            continue
            
        circuit_name = parts[0]
        run_id = parts[1]
        
        recipe_str = None
        area = None
        delay = None
        power = None
        
        # Read the file and apply Regex
        with open(filepath, 'r') as file:
            for line in file:
                # Check for Recipe
                if recipe_str is None:
                    recipe_match = recipe_pattern.match(line)
                    if recipe_match:
                        # Clean up the trailing semicolon and spaces
                        recipe_str = recipe_match.group(1).strip().rstrip(';')
                        continue
                
                # Check for Stats
                stats_match = stats_pattern.search(line)
                if stats_match:
                    area = float(stats_match.group(1))
                    delay = float(stats_match.group(2))
                    power = float(stats_match.group(3))
                    break # Found the stats, no need to read the rest of the file
        
        # If we successfully found all data points, add it to our list
        if recipe_str and area is not None and delay is not None and power is not None:
            dataset_rows.append([
                circuit_name, 
                run_id, 
                recipe_str, 
                power,  # Placing power first since it's the primary target
                area, 
                delay
            ])
        else:
            print(f"Warning: Missing data in {filename}. Skipping.")

    # Write everything to a clean CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write headers
        writer.writerow(['Circuit', 'Run_ID', 'Recipe', 'Power', 'Area', 'Delay'])
        # Write data
        writer.writerows(dataset_rows)
        
    print(f"Success! Extracted {len(dataset_rows)} valid records to {OUTPUT_CSV}.")

if __name__ == "__main__":
    parse_all_logs()