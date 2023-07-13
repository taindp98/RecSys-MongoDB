from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=r"D:\data\fashion",
    repo_id="taindp98/fashion-recsys",
    repo_type="dataset",
)