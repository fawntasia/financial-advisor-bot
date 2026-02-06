import os
import argparse
from huggingface_hub import hf_hub_download

def download_llama3(model_path="models/llama3/"):
    """
    Downloads Llama 3 GGUF model from Hugging Face.
    """
    repo_id = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF"
    filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Starting download of {filename} from {repo_id}...")
    
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print(f"Download complete. Model saved to: {path}")
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama 3 GGUF model.")
    parser.add_argument("--path", type=str, default="models/llama3/", help="Directory to save the model.")
    args = parser.parse_args()
    
    # We don't actually download unless the user confirms or we are told to in the script logic.
    # For now, we just implement it as requested.
    print("This script will download Llama 3 (approx. 5GB).")
    confirm = input("Proceed with download? (y/n): ").lower()
    if confirm == 'y':
        download_llama3(args.path)
    else:
        print("Download cancelled.")
