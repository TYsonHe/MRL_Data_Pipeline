import kagglehub

# Download latest version
path = kagglehub.dataset_download("willarevalo/chexpert-v10-small")

print("Path to dataset files:", path)
