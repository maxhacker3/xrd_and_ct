import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from qdml.fem.datasets import DatasetFEM
import numpy as np



dir = Path.resolve(Path(__file__).parent) 
data_dir = dir / "data"


def read_ct_data(file_path: Path):
    
    h5path = list(file_path.glob("*.h5"))[0]
    with h5py.File(h5path, "r") as f:
        data = f["dataset"][:]
        print(f.keys())
        
    return data
    
    
def plot_ct_data(data):
    
    plt.imshow(data[0], cmap="gray")
    plt.title("CT Scan Slice")
    plt.axis("off")
    plt.show()
    
def plot_ct_data_napari(data):
    print(data.shape)
    manipulated_data = np.repeat(data[None, ...], 3, axis=0)[None]
    
    
    # manipulate data
    vertical_sclice = np.s_[:, :, 25:295, ...]
    n_slices = 295 - 25
    
    top_start, bottom_start, left_start, right_start = 116, 240, 88, 212 # 2d roi at lowest z slice
    top_end, bottom_end, left_end, right_end = 113, 250, 127, 277 # 2d roi at highest z slice
    
    print(manipulated_data.shape)
    data_m = manipulated_data[vertical_sclice]
    valid_mask = np.zeros_like(data_m, dtype=bool)
    for i, t in enumerate(np.linspace(0, 1, data_m.shape[2])):
        top, bottom, left, right = np.rint(
        (1 - t) * np.array([116, 240, 88, 212])
        + t * np.array([105, 270, 127, 295])
        ).astype(int)

        valid_mask[:, :, i, top:bottom, left:right] = True

    masked_data = data_m * valid_mask
    print(masked_data.shape)
    DatasetFEM(masked_data).view(axis=0)
    
    
    
    
if __name__ == "__main__":
    data = read_ct_data(data_dir)
    
    plot_ct_data_napari(data)
    