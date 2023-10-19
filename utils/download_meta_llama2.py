from huggingface_hub import snapshot_download

repo_id = "meta-llama/Llama-2-7b-hf"  # the name in huggingface
local_dir = "weights/meta-llama/Llama-2-7b-hf/"  # the local path to save weights
local_dir_use_symlinks = False  # no blob
token = "your access token in huggingface"  # your access token in huggingface



snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token,
)
