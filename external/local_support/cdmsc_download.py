from huggingface_hub import snapshot_download
p=snapshot_download(repo_id="aptemvs/mosquitoes-biodcase2026-task5",repo_type="dataset",revision="main",local_dir=".",
    allow_patterns=["data/metadata/**","data/perch/**","reports/fm_embeddings/birdmae.parquet","reports/fm_embeddings/birdmae.json"])
print("DONE",p,flush=True)
