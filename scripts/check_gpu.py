import torch

def main():
    cuda_avail = torch.cuda.is_available()
    print("CUDA Available:", cuda_avail)
    if cuda_avail:
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Device Count:", torch.cuda.device_count())
    else:
        print("Device Name: CPU (No CUDA detected on native Windows)")

if __name__ == "__main__":
    main()
