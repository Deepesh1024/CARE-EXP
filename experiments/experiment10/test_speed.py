import torch
import time

def test_speed():
    d = 2048
    device = "cuda"
    A = torch.randn(d, d, device=device)
    C_A = A @ A.T
    
    start = time.time()
    for _ in range(100):
        torch.linalg.eigvalsh(C_A)
    torch.cuda.synchronize()
    end = time.time()
    print(f"100 eigvalsh: {end - start:.3f}s")
    
if __name__ == "__main__":
    test_speed()
