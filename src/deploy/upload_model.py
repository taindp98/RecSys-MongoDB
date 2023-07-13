from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

# api.upload_file(
#     path_or_fileobj=r"C:\Users\Admin\working\python\mine\Clothes-Classification\weights\checkpoint.pth",
#     path_in_repo=r"checkpoint.pth",
#     repo_id="taindp98/siamese-model",
#     repo_type="dataset",
# )

model_local_path = hf_hub_download(
    repo_id="taindp98/siamese-model",
    filename="checkpoint.pth",
    repo_type="dataset",
    local_files_only=True,
)
print(f'model_local_path: {model_local_path}')