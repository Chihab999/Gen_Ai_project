import os
import subprocess
import shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Configuration
ASSETS_DIR = "assets"
MODELS = {
    "C-GLD": "C-GLD",
    "GraphGAN": "graphGan",
    "GraphGAN-VAE": "graph_gan_vae",
    "UltimateGen": "ultimate_gen"
}

METRICS_FILE = "metrics.json"

def run_evaluation(model_name, model_dir):
    print(f"\n--- Processing {model_name} ---")
    
    # Check if evaluate_advanced.py exists
    script_path = os.path.join(model_dir, "evaluate_advanced.py")
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return None

    # Run the evaluation script
    # We assume the script has been modified to output metrics.json and images to 'evaluation_results'
    try:
        print(f"Running evaluation for {model_name}...")
        subprocess.run([sys.executable, "evaluate_advanced.py"], cwd=model_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {model_name}: {e}")
        # return None # Continue to try to collect assets anyway if they exist, or just fail for this model
        # We will try to continue to see if others work

    # Define results directory (assumed to be 'evaluation_results' inside model_dir)
    results_dir = os.path.join(model_dir, "evaluation_results")
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found.")
        return None

    # Create model asset directory
    start_assets_dir = os.path.join(ASSETS_DIR, model_name)
    os.makedirs(start_assets_dir, exist_ok=True)

    # Move/Copy assets
    metrics_data = {}
    
    # 1. Metrics JSON
    json_path = os.path.join(results_dir, METRICS_FILE)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics_data = json.load(f)
        shutil.copy2(json_path, os.path.join(start_assets_dir, f"{model_name}_metrics.json"))
    else:
        print(f"Warning: {METRICS_FILE} not found for {model_name}")

    # 2. Images
    # Map from script output names to standardized names
    image_mapping = {
        "dist_qed.png": "qed_distribution.png",
        "dist_logp.png": "logp_distribution.png",
        "dist_similarity.png": "similarity_distribution.png",
        "generated_molecules.png": "samples.png",
        "qed_dist_enhanced.png": "qed_distribution.png", # Handle variation
        "logp_dist_enhanced.png": "logp_distribution.png", # Handle variation
        "ultimate_generated.png": "samples.png" # Handle variation
    }

    for filename in os.listdir(results_dir):
        if filename.endswith(".png"):
            src = os.path.join(results_dir, filename)
            target_name = image_mapping.get(filename, filename)
            dst = os.path.join(start_assets_dir, target_name)
            shutil.copy2(src, dst)
            print(f"Copied {filename} -> {target_name}")

    return metrics_data

def generate_summary_report(all_metrics):
    print("\n--- Generating Summary Report ---")
    
    # Convert to DataFrame for easier handling
    data = []
    for model_name, metrics in all_metrics.items():
        if metrics:
            row = {"Model": model_name}
            row.update(metrics)
            data.append(row)
    
    if not data:
        print("No metrics collected.")
        return

    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(ASSETS_DIR, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Create Markdown Report
    md_content = "# Generative Models Evaluation Summary\n\n"
    md_content += df.to_markdown(index=False)
    
    md_content += "\n\n## Comparison Charts\n"
    
    # Plotting comparison charts
    metrics_to_plot = ["Validity", "Uniqueness", "Novelty", "QED_Mean", "LogP_Mean", "Similarity_Mean"]
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(df["Model"], df[metric], color=['blue', 'green', 'orange', 'red'])
            plt.title(f"Model Comparison - {metric}")
            plt.ylabel(metric)
            plt.savefig(os.path.join(ASSETS_DIR, f"comparison_{metric}.png"))
            plt.close()
            md_content += f"![{metric}](comparison_{metric}.png)\n"

    md_path = os.path.join(ASSETS_DIR, "summary_metrics.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Summary Markdown saved to {md_path}")

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    all_metrics = {}
    
    for model_key, model_dir in MODELS.items():
        metrics = run_evaluation(model_key, model_dir)
        if metrics:
            all_metrics[model_key] = metrics
            
    generate_summary_report(all_metrics)
    print("\nDone! Assets collected in 'assets/' folder.")

if __name__ == "__main__":
    main()
