import os
import gdown



files_to_download = [
    {'id': '10x1qO8eNLqJRYvQY3RjnrivpD4ZwRuMn', 'name': 'vocab.txt'},
    {'id': '16muInGzJUPRtZUL9XPwzmnipSzNMH9oA', 'name': 'tokenizer_config.json'},
    {'id': '1czJLhoTfOfkjQZ8iWyR3mnPm9SQILiZ0', 'name': 'special_tokens_map.json'},
    {'id': '1f-oQ2pXAUaEjCJqnC1_j1sHv_dpNyi4S', 'name': 'model.safetensors'},
    {'id': '1KexmxEKOPdZaNClp6O-6oLoUzSBO9_wk', 'name': 'config.json'}
]



save_dir = 'saved_model'
os.makedirs(save_dir, exist_ok=True)

# for file in files_to_download:
#     url = f"https://drive.google.com/uc?id={file['id']}"
#     output_path = os.path.join(save_dir, file['name'])
#     print(f"Downloading {file['name']} from {url}...")
#     gdown.download(url, output_path, quiet=False)


url = f"https://drive.google.com/uc?id={files_to_download[0]['id']}"
output_path = os.path.join(save_dir, files_to_download[0]['name'])
print(f"Downloading {files_to_download[0]['name']} from {url}...")
gdown.download(url, output_path, quiet=False)

url = f"https://drive.google.com/uc?id={files_to_download[1]['id']}"
output_path = os.path.join(save_dir, files_to_download[1]['name'])
print(f"Downloading {files_to_download[1]['name']} from {url}...")
gdown.download(url, output_path, quiet=False)

url = f"https://drive.google.com/uc?id={files_to_download[2]['id']}"
output_path = os.path.join(save_dir, files_to_download[2]['name'])
print(f"Downloading {files_to_download[2]['name']} from {url}...")
gdown.download(url, output_path, quiet=False)

url = f"https://drive.google.com/uc?id={files_to_download[3]['id']}"
output_path = os.path.join(save_dir, files_to_download[3]['name'])
print(f"Downloading {files_to_download[3]['name']} from {url}...")
gdown.download(url, output_path, quiet=False)

url = f"https://drive.google.com/uc?id={files_to_download[4]['id']}"
output_path = os.path.join(save_dir, files_to_download[4]['name'])
print(f"Downloading {files_to_download[4]['name']} from {url}...")
gdown.download(url, output_path, quiet=False)



print("All files successfully downloaded.")