import torch
import matplotlib.pyplot as plt
import numpy as np
from improved_diffusion.unet_v2 import RingBasedPE

def visualize_ring_pe():
    """
    Visualize the Ring-based Positional Encoding patterns
    """
    # Create the PE module
    pe_module = RingBasedPE(channels=42, height=173, width=360, W_ring=1, T=500.0)
    
    # Get the encoding tensor
    pe_tensor = pe_module.pe.squeeze(0).numpy()  # Shape: [42, 173, 180]
    
    # Visualize a few layers
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    layers_to_show = [1,4,7,39,40,41]
    
    for idx, (ax, layer) in enumerate(zip(axes.flat, layers_to_show)):
        im = ax.imshow(pe_tensor[layer], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'Layer {layer}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Ring-based Positional Encoding Patterns (sin/cos alternating rings)')
    plt.tight_layout()
    plt.savefig('ring_pe_visualization.png', dpi=150)
    
    print(f"PE tensor shape: {pe_tensor.shape}")
    print(f"PE tensor range: [{pe_tensor.min():.3f}, {pe_tensor.max():.3f}]")
    
    # Verify the alternating pattern
    center_h, center_w = 86, 89  # (173-1)/2, (180-1)/2
    for r in range(0, 100, 10):
        if r < 173/2:  # Within bounds
            h, w = int(center_h), int(center_w + r)
            if w < 180:
                k_ring = r // 10
                func_type = "sin" if k_ring % 2 == 0 else "cos"
                print(f"Distance ~{r}, Ring {k_ring}: using {func_type}")

if __name__ == "__main__":
    visualize_ring_pe()