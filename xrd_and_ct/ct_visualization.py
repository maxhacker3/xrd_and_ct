from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.ndimage as sps
from pathlib import Path
from xrd_and_ct import PathManager
 
from numpy.typing import NDArray


class CTVisualization:
    def __init__(self, mode: str):
        self.mode = mode
        self.angular_steps = 720
        
    
    def __call__(self, phantom: NDArray, animate: bool = False):
        if self.mode == "sino":
            if not animate:
                return self.compute_sino(phantom)
            else:
                return self.animate_sino(phantom)
                
    
    def compute_sino(self, phantom: NDArray, show: bool = True, single_proj_check: bool = False) -> NDArray:
        phantom = np.asarray(phantom, dtype=float)
        
        angular_domain = np.linspace(0, 360, self.angular_steps, endpoint=False)
        
        n_rows = phantom.shape[0]
        sinogram = np.zeros((self.angular_steps, n_rows))
        
        
        
        for i, angle in enumerate(angular_domain):
            rotated_sample = sps.rotate(phantom, -angle, reshape=False)
            sinogram[i, :] = np.sum(rotated_sample, axis = 1)
            
        if show:
            self._plot_pair(phantom, sinogram)
            plt.show()
            
        return sinogram
    
    def compute_fourier_sino(
        self, phantom: NDArray, show: bool = True, *, n_angles: int = 30
    ) -> NDArray:
        
        
        phantom = np.asarray(phantom, dtype=float)
       
        
        size = int(np.ceil(np.hypot(*phantom.shape)))
        size += 1 - size % 2
        
        padding = [(size - length) // 2 for length in phantom.shape]
        padded = np.zeros((size, size))
        y0, x0 = padding
        padded[y0:y0 + phantom.shape[0], x0:x0 + phantom.shape[1]] = phantom
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        sinogram = np.empty((n_angles, size))
        for i, angle in enumerate(angles):
            rotated = sps.rotate(padded, -angle, reshape=False, order=1)
            sinogram[i] = rotated.sum(axis=1)

        projection_ffts = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(sinogram, axes=1), axis=1), axes=1
        )
        frequencies = np.fft.fftshift(np.fft.fftfreq(size))
        theta = np.deg2rad(angles)[:, None]


        fx = np.sin(theta) * frequencies[None, :]
        fy = np.cos(theta) * frequencies[None, :]
        columns = np.rint(fx * size).astype(int) + size // 2
        rows = np.rint(fy * size).astype(int) + size // 2
        totals = np.zeros((size, size), dtype=complex)
        counts = np.zeros((size, size), dtype=int)
        
        np.add.at(totals, (rows.ravel(), columns.ravel()), projection_ffts.ravel())
        np.add.at(counts, (rows.ravel(), columns.ravel()), 1)
        fourier_space = np.full((size, size), complex(np.nan, np.nan))
        sampled = counts > 0
        fourier_space[sampled] = totals[sampled] / counts[sampled]

        if show:
            fig, (phantom_ax, fourier_ax) = plt.subplots(
                1, 2, figsize=(12, 5), layout="constrained"
            )
            
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#eeeeee")
            half_bin = 0.5 / size
            limits = (frequencies[0] - half_bin, frequencies[-1] + half_bin)
            image = fourier_ax.imshow(
                np.ma.masked_invalid(np.abs(fourier_space)),
                origin="lower", cmap=cmap, interpolation="nearest",
                extent=(*limits, *limits),
            )
            fourier_ax.set(
                title=f"Fourier slices from {n_angles} projections",
                xlabel=r"$f_x$ (cycles/pixel)", ylabel=r"$f_y$ (cycles/pixel)",
            )
            fig.colorbar(image, ax=fourier_ax, label="log(1 + FFT magnitude)")
            
            filled_fourier = np.nan_to_num(fourier_space, nan=0.0)

            reconstruction = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(filled_fourier))
            ).real
            phantom_ax.imshow(reconstruction, cmap="gray", origin="upper")
            phantom_ax.set(title="Phantom", xlabel="Column (pixels)", ylabel="Row (pixels)")
            plt.show()

        return fourier_space

    def _plot_pair(self, phantom, sinogram):
        """Use the same axes and intensity scale for static and animated views."""
        fig, (phantom_ax, sino_ax) = plt.subplots(
            1, 2, figsize=(12, 5), layout="constrained"
        )
        phantom_ax.imshow(phantom, cmap="gray", origin="upper")
        phantom_ax.set(title="Phantom", xlabel="Column (pixels)", ylabel="Row (pixels)")
        image = sino_ax.imshow(
            sinogram.T, cmap="gray", origin="upper", aspect="auto",
            extent=(0, 360, phantom.shape[0] - 0.5, -0.5),
            vmin=min(0, float(sinogram.min())), vmax=float(sinogram.max()),
        )
        sino_ax.set(title="Sinogram", xlabel="Projection angle (degrees)",
                    ylabel="Detector position (pixels)")
        return fig, phantom_ax, sino_ax, image

    def animate_sino(self, phantom: NDArray, *, duration=12, fps=30, save_path=None):
        """Reveal projections alongside a rotating central ray direction.

        The bar indicates the direction of the parallel rays; each sinogram
        column contains all detector positions at that angle. Keep the returned
        animation alive. Optional export supports GIF (Pillow) or MP4 (ffmpeg).
        """
        if duration <= 0 or fps <= 0:
            raise ValueError("Duration and fps must be positive.")
        sinogram = self.compute_sino(phantom, show=False)
        fig, phantom_ax, sino_ax, image = self._plot_pair(phantom, sinogram)
        revealed = np.full(sinogram.T.shape, np.nan)
        image.set_data(np.ma.masked_invalid(revealed))
        image.cmap = image.cmap.copy()
        image.cmap.set_bad("#eeeeee")
        bar, = phantom_ax.plot([], [], color="darkblue", lw=3)
        cursor = sino_ax.axvline(0, color="darkblue", lw=1.5)
        angle_text = phantom_ax.text(
            0.03, 0.97, "", transform=phantom_ax.transAxes,
            va="top", color="darkblue",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        rows, columns = np.shape(phantom)
        cx, cy = (columns - 1) / 2, (rows - 1) / 2
        radius = np.hypot(rows, columns) / 2
        # Prevent the bar from changing the phantom's axis limits.
        phantom_ax.set_xlim(-0.5, columns - 0.5)
        phantom_ax.set_ylim(rows - 0.5, -0.5)
        frame_count = max(2, round(duration * fps))

        def init():
            revealed[:] = np.nan
            image.set_data(np.ma.masked_invalid(revealed))
            bar.set_data([], [])
            cursor.set_xdata([0, 0])
            angle_text.set_text("Ready to acquire")
            return image, bar, cursor, angle_text

        def update(frame):
            count = 1 + round(frame * (self.angular_steps - 1) / (frame_count - 1))
            revealed[:] = np.nan
            revealed[:, :count] = sinogram[:count].T
            image.set_data(np.ma.masked_invalid(revealed))
            angle = (count - 1) * 360 / self.angular_steps
            radians = np.deg2rad(angle)
            dx, dy = radius * np.cos(radians), -radius * np.sin(radians)
            bar.set_data([cx - dx, cx + dx], [cy - dy, cy + dy])
            cursor.set_xdata([angle, angle])
            angle_text.set_text(f"Projection angle: {angle:.1f}°")
            return image, bar, cursor, angle_text

        animation = FuncAnimation(
            fig, update, frames=frame_count, init_func=init,
            interval=1000 / fps, repeat=False, blit=False,
        )
        if save_path is not None:
            writer = "pillow" if str(save_path).lower().endswith(".gif") else "ffmpeg"
            animation.save(str(save_path), writer=writer, fps=fps, dpi=150)
        plt.show()
        return animation
    
def main():
    vis = CTVisualization(mode = "sino")
    phantom = np.load("data/phantom.npy")
    print(phantom.shape)
    
    file_path = PathManager.FIGURES / "ct_animation.gif"
    
    #vis.animate_sino(phantom, duration=5, save_path=file_path)
    #vis.compute_sino(phantom, show = True)
    vis.compute_fourier_sino(phantom, show = True, n_angles = 180)
    
if __name__ == "__main__":
    main()
    
    
