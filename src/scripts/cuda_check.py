import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA version PyTorch is compiled with: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
else:
    print("CUDA is NOT available. PyTorch will run on CPU.")

print(f"\nPyTorch version: {torch.__version__}")