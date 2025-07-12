from huggingface_hub import snapshot_download

REPO_ID = "VincentX26/VDiT_Sparsity_Trace"

snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir="./trace")
