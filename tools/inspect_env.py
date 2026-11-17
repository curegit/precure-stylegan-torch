#!/usr/bin/env python3

import torch
import torch.cuda
import torch.backends.cudnn
import torch.backends.mps


def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CPU Threads:     {torch.get_num_threads()}")
    print(f"MPS Available:   {torch.backends.mps.is_available()}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"cuDNN Version:   {torch.backends.cudnn.version()}")
        print(f"Device Count:    {torch.cuda.device_count()}")
        print(f"Current Device:  {(current := torch.cuda.current_device())} ({torch.cuda.get_device_name(current)})")


if __name__ == "__main__":
    main()
