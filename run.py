#!/usr/bin/env python3
"""
Simple runner script for the Argument Link project.
This handles the import path setup and runs the main processing pipeline.
"""

import sys
import os
import json

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the main argument linking pipeline."""
    try:
        from main import main as process_data
        
        # Load data
        data_path = os.path.join('data', 'stanford_hackathon_brief_pairs_clean.json')
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            print("Please ensure the data file exists in the data/ folder.")
            return 1
            
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print("Processing legal brief pairs...")
        results = process_data(data)
        
        # Save results
        output_path = 'output.txt'
        with open(output_path, 'w') as f:
            f.write(str(results))
        
        print(f"Results saved to {output_path}")
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())