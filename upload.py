from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    repo_id="YYT-t/data_and_models",
    repo_type="model",
    folder_path="data_and_models"
)
