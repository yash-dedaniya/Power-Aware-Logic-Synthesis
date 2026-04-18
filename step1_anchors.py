import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

NUM_RECIPES_TO_GENERATE = 1500
NUM_ANCHORS_NEEDED = 5
RECIPE_LENGTH = 20

ABC_COMMANDS = ["refactor", "refactor -z", "rewrite", "rewrite -z", "resub", "resub -z", "balance"]

def generate_random_recipe():
    return "; ".join(random.choices(ABC_COMMANDS, k=RECIPE_LENGTH))

def generate_smart_anchors():
    print("🌟 GENERATING 5 K-MEANS SMART ANCHORS (MATH BYPASS)")
    
    raw_recipes = [generate_random_recipe() for _ in range(NUM_RECIPES_TO_GENERATE)]
    
    # Math bypass: Count the exact frequency of commands
    recipe_embeddings = []
    for recipe_str in raw_recipes:
        cmds = recipe_str.split('; ')
        counts = [cmds.count(cmd) for cmd in ABC_COMMANDS]
        recipe_embeddings.append(counts)

    embedding_matrix = np.array(recipe_embeddings)
    
    # Cluster into 5 diverse groups
    kmeans = KMeans(n_clusters=NUM_ANCHORS_NEEDED, random_state=42, n_init=10)
    kmeans.fit(embedding_matrix)
    
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embedding_matrix)
    
    final_anchors = [raw_recipes[idx] for idx in closest_indices]
    
    print("✅ 5 Anchors Generated Successfully.")
    return final_anchors

if __name__ == "__main__":
    anchors = generate_smart_anchors()
    for i, a in enumerate(anchors):
        print(f"Anchor {i+1}: {a}")