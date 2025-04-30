import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, help="Path to local directory with model files")
    parser.add_argument(
        "--repo", required=True, help="Name of the Hugging Face repo (e.g. 'my-finetuned-model')"
    )
    args = parser.parse_args()

    username = "iboero16"  # <-- Update if needed

    # Remove .gitignore file if it exists
    gitignore_path = os.path.join(args.local, ".gitignore")
    if os.path.exists(gitignore_path):
        os.remove(gitignore_path)
        print(f"Removed {gitignore_path}")
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=args.local,
        repo_id=f"{username}/{args.repo}",
        repo_type="model",
    )

if __name__ == "__main__":
    main()
