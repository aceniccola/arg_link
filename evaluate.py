import json
import argparse
import os
from typing import List, Dict, Any, Tuple, Set

# --- Assumption ---
# This script assumes you can import the core processing logic from your main script.
# You might need to adjust the import path or refactor your main.py
# so that the function processing a single data entry is callable.
# We'll assume the function is named `iteration` and is available in `main.py`.
# If your function has a different name or location, update the import below.
try:
    # Assuming main.py is in the same directory or Python path
    from main import iteration as process_brief_pair 
    # If main.py cannot be imported directly (e.g., due to __main__ block),
    # you'll need to refactor its core logic into importable functions.
except ImportError:
    print("Error: Could not import the processing function from main.py.")
    print("Please ensure main.py is structured to allow importing its core logic (e.g., the 'iteration' function).")
    exit(1)
# --- End Assumption ---


def load_data(json_path: str) -> List[Dict[str, Any]]:
    """Loads the evaluation data from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: Input JSON file not found at {json_path}")
        exit(1)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Error: Expected JSON root to be a list of entries.")
            exit(1)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file at {json_path}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        exit(1)

def format_links(links: List[List[str]]) -> Set[Tuple[str, str]]:
    """Converts a list of [moving, response] pairs into a set of tuples for comparison."""
    # Ensure input is a list of lists, each with two strings
    formatted_set = set()
    for link in links:
        if isinstance(link, list) and len(link) == 2 and all(isinstance(item, str) for item in link):
             # Normalize whitespace or case if needed, e.g., link[0].strip()
            formatted_set.add(tuple(link))
        else:
            print(f"Warning: Skipping malformed link entry: {link}")
            
    return formatted_set

def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1-score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main(json_path: str):
    """Main function to load data, run processing, evaluate, and print results."""
    print(f"Loading evaluation data from: {json_path}")
    evaluation_data = load_data(json_path)
    print(f"Loaded {len(evaluation_data)} entries.")

    all_true_links = set()
    all_predicted_links = set()
    
    print("Processing entries...")
    for i, entry in enumerate(evaluation_data):
        print(f"  Processing entry {i+1}/{len(evaluation_data)}...")
        
        # --- Ground Truth ---
        if "true_links" not in entry or not isinstance(entry["true_links"], list):
             print(f"Warning: Entry {i+1} missing or has malformed 'true_links'. Skipping.")
             continue
        true_links_set = format_links(entry["true_links"])
        all_true_links.update(true_links_set)

        # --- Prediction ---
        try:
            # Call the core logic from your main script
            # This function should take the data entry and return a list of predicted links
            # in the same format as true_links: [[moving_heading, response_heading], ...]
            predicted_links_list = process_brief_pair(entry) 
            
            if not isinstance(predicted_links_list, list):
                 print(f"Warning: Processing function did not return a list for entry {i+1}. Skipping prediction for this entry.")
                 predicted_links_set = set()
            else:
                predicted_links_set = format_links(predicted_links_list)
                
            all_predicted_links.update(predicted_links_set)
            
        except Exception as e:
            print(f"Error processing entry {i+1}: {e}")
            # Decide if you want to stop or continue
            # continue 
            
    print("Processing complete.")
    print("-" * 30)

    # --- Calculate Performance ---
    print("Calculating performance metrics...")
    
    true_positives_set = all_true_links.intersection(all_predicted_links)
    
    tp = len(true_positives_set)
    fp = len(all_predicted_links) - tp
    fn = len(all_true_links) - tp

    metrics = calculate_metrics(tp, fp, fn)

    print("\n--- Evaluation Summary ---")
    print(f"Total True Links: {len(all_true_links)}")
    print(f"Total Predicted Links: {len(all_predicted_links)}")
    print(f"True Positives (Correctly Predicted): {tp}")
    print(f"False Positives (Incorrectly Predicted): {fp}")
    print(f"False Negatives (Missed Links): {fn}")
    print("-" * 30)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the legal argument linking system.")
    parser.add_argument("json_file", help="Path to the input JSON file containing evaluation data.")
    
    args = parser.parse_args()
    
    main(args.json_file)
