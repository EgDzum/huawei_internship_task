import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "Elriggs/openwebtext-100k"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


if __name__=='__main__':
    print(DEVICE)