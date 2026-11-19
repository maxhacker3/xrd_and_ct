
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from pathlib import Path
import xraydb
import os


from xrd_and_ct import PathManager


Al = (0, 0, 0.3522)
O = (0.3061, 0, 0.25)

a, c = 4.759, 12.991  # Angstrom

A = np.array([
    [a, -a/2, 0],
    [0, np.sqrt(3)*a/2, 0],
    [0, 0, c]
])

B = 2*np.pi * np.linalg.inv(A).T


def compute_frequency(h, k, l):
    hkl = np.array([h, k, l])
    G = B @ hkl
    return np.linalg.norm(G)


translations = np.array([
    [0, 0, 0],
    [2/3, 1/3, 1/3],
    [1/3, 2/3, 2/3]
])


def symmetry_ops(x, y, z):
    return np.array([
        [x, y, z],
        [-y, x-y, z],
        [-x+y, -x, z],
        [y, x, -z+0.5],
        [x-y, -y, -z+0.5],
        [-x, -x+y, -z+0.5],
        [-x, -y, -z],
        [y, -x+y, -z],
        [x-y, x, -z],
        [-y, -x, z+0.5],
        [-x+y, y, z+0.5],
        [x, x-y, z+0.5]
    ])


def generate_positions(atom):
    positions = []

    for p in symmetry_ops(*atom):
        for t in translations:
            positions.append((p + t) % 1)

    positions = np.round(positions, 8) % 1
    return np.unique(positions, axis=0)


def structure_factor(al_positions, o_positions, h, k, l):
    hkl = np.array([h, k, l])

    q = compute_frequency(h, k, l)

    s = q / (4 * np.pi)

    f_al = xraydb.f0("Al", s)
    f_o = xraydb.f0("O", s)

    phase_al = -2j * np.pi * (al_positions @ hkl)
    phase_o = -2j * np.pi * (o_positions @ hkl)

    F_al = f_al * np.sum(np.exp(phase_al))
    F_o = f_o * np.sum(np.exp(phase_o))

    return F_al + F_o


def abs_structure_factor(factor):
    return np.abs(factor)**2


def lorentian(domain, amplitude, center, gamma):
    return amplitude / np.pi * (
        gamma / ((domain - center)**2 + gamma**2)
    )
    
def gaussian_noise(values, sigma):
    return np.random.normal(values, sigma)


def polarization_factor(scatter_freq: float, lmbda: float):
    return 1 / 2.0 * (1 + np.cos(2 * np.arcsin(scatter_freq * lmbda / (4*np.pi)))**2)

def lorentz_factor(scatter_freq: float, lmbda: float):
    theta = np.arcsin(scatter_freq * lmbda / (4*np.pi))
    return 1 / (np.sin(theta) * np.sin(2*theta) + 1e-12)
    

def conduct_diffractometry(al_pos, o_pos, lmbda: float, *, return_hkls=False):
    """Calculation of extinction behavior with Miller indices.
    
    :param al_pos: Aluminium positions
    :param o_pos: Oxid positions
    :returns: Intensities and frequencies, plus Miller indices if return_hkls=True.
    """
    freqs = []
    intensities = []
    hkls = []

    for h in range(-10, 11):
        for k in range(-10, 11):
            for l in range(-10, 11):

                if h == k == l == 0:
                    continue

                q = compute_frequency(h, k, l)
                if not (1 < q < 6):
                    continue
                F = structure_factor(
                    al_pos, o_pos, h, k, l
                )

                I = abs_structure_factor(F) * lorentz_factor(q, lmbda) * polarization_factor(q, lmbda)

                if I < 1e-6:
                    continue

                freqs.append(q)
                intensities.append(I)
                hkls.append((h, k, l))

    if return_hkls:
        return np.array(intensities), np.array(freqs), np.array(hkls)
    return np.array(intensities), np.array(freqs)

################################## plotting and animation ##################################

def annotate_reflections(ax, domain, spectrum, freqs, hkls, *, prominence=0.03,
                         visible=True, experiment=None, match_tolerance=0.04):
    
    peaks, _ = find_peaks(spectrum, prominence=prominence)
    peak_positions = {}
    if experiment is not None:
        measured = np.flatnonzero(np.isfinite(experiment))
        exp_peaks, _ = find_peaks(experiment[measured], prominence=prominence)
        exp_peaks = measured[exp_peaks]
        candidates = sorted(
            (abs(domain[s] - domain[e]), s, e)
            for s in peaks for e in exp_peaks
            if abs(domain[s] - domain[e]) <= match_tolerance
        )
        used_exp = set()
        for _, s, e in candidates:
            if s not in peak_positions and e not in used_exp:
                peak_positions[s] = (domain[e], experiment[e])
                used_exp.add(e)
    annotations = []
    freqs, hkls = np.asarray(freqs), np.asarray(hkls)
    if hkls.shape != (len(freqs), 3):
        raise ValueError("hkls must have shape (len(freqs), 3).")
    for peak in peaks:
        q = freqs[np.argmin(np.abs(freqs - domain[peak]))]
        equivalents = hkls[np.isclose(freqs, q, rtol=0, atol=1e-7)]
        h, k, l = min(
            equivalents,
            key=lambda indices: (
                np.count_nonzero(indices < 0), indices[0] < indices[1],
                np.abs(indices).sum(), tuple(indices),
            ),
        )
        label = "(" + " ".join(
            str(int(index)) if index >= 0 else rf"\overline{{{abs(int(index))}}}"
            for index in (h, k, l)
        ) + ")"
        annotation = ax.annotate(
            f"${label}$", peak_positions.get(peak, (domain[peak], spectrum[peak])),
            xytext=(-6, 6), textcoords="offset points", ha="right",
            va="bottom", rotation=0, fontsize=8, color="black",
            visible=visible,
        )
        annotations.append((peak, annotation))
    return annotations

def load_in_exp_data():
    file_path = PathManager.DATA / "xrd_data"
    
    files = list(file_path.glob("*.txt"))
    print(files)

    def extract_temp(fname):
        base = os.path.basename(fname)
        temp_str = base.split("_")[-1].replace("C.txt", "")
        print(temp_str)
        if temp_str.upper() == "RT.TXT":
            return 25   # room temperature in °C
        return int(temp_str)

    files = sorted(files, key=extract_temp)


    M = []
    for f in files:         # Loade the data
        data = np.loadtxt(f, skiprows=1)
        M.append(data)

    M = np.stack(M, axis=2)
    
    theta = M[:, 0, 0]
    I = M[:, 1, 0]
    lambda_A = 1.54056
    Q = (4 * np.pi / lambda_A) * np.sin(np.deg2rad(theta / 2))
    
    return Q, I
    

def plot_results(reflexes, freqs, hkls=None):
    
    
    domain = np.linspace(1, 6, 3000)
    spectrum = np.zeros_like(domain)

    # simulation
    for I, q in zip(reflexes, freqs):
        spectrum += lorentian(domain, I, q, gamma=0.01)
        
    # experiment
    q_data, intensity = load_in_exp_data()
    experimental_spectrum = np.interp(
        domain,       # positions where you want intensities
        q_data,       # positions of the measured intensities
        intensity,    # measured intensities
        left=np.nan,
        right=np.nan,
    )
    
        
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(domain, gaussian_noise(spectrum / spectrum.max(), 1e-2))
    ax.plot(domain, experimental_spectrum / np.nanmax(experimental_spectrum), marker="^", markersize=3, lw = 1, color = "orange")

    ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Simulated corundum powder XRD")
    ax.set_xlim(1, 6)
    if hkls is not None:
        annotate_reflections(
            ax, domain, spectrum / spectrum.max(), freqs, hkls,
            experiment=experimental_spectrum / np.nanmax(experimental_spectrum),
        )
        ax.set_ylim(-0.03, 1.3)

    plt.tight_layout()
    plt.show()


def animate_results(
    reflexes, freqs, lmbda, hkls=None, *, duration=10, initial_pause=2, final_pause=3,
    fps=30, match_tolerance=0.04, prominence=0.03, save_path=None,
):
    """Reveal the simulation over the experiment and mark matching peaks.

    Times are in seconds; match_tolerance is in inverse angstroms.
    prominence is relative to each curve's maximum. Peaks are paired
    one-to-one by nearest position within the tolerance. Keep the returned
    animation alive while viewing. Set save_path to a .gif or .mp4 to export
    (MP4 requires ffmpeg).
    """
    if duration <= 0 or fps <= 0 or initial_pause < 0 or final_pause < 0:
        raise ValueError("Duration and fps must be positive; pauses nonnegative.")
    if match_tolerance <= 0 or prominence <= 0:
        raise ValueError("Peak tolerance and prominence must be positive.")

    domain = np.linspace(1, 6, 3000)
    spectrum = np.zeros_like(domain)
    for intensity, q in zip(reflexes, freqs):
        
        corrected_intensity = intensity * (2*np.pi) / lmbda * np.cos(np.arcsin(q * lmbda / (4*np.pi)))
        spectrum += lorentian(domain, corrected_intensity, q, gamma=0.01)
    if not np.isfinite(spectrum).all() or spectrum.max() <= 0:
        raise ValueError("The simulated spectrum must have a positive finite peak.")
    spectrum /= spectrum.max()

    q_data, intensity = load_in_exp_data()
    valid = np.isfinite(q_data) & np.isfinite(intensity)
    q_data, intensity = q_data[valid], intensity[valid]
    order = np.argsort(q_data)
    q_data, intensity = q_data[order], intensity[order]
    experiment = np.interp(domain, q_data, intensity, left=np.nan, right=np.nan)
    overlap = np.isfinite(experiment)
    if not overlap.any() or np.nanmax(experiment) <= 0:
        raise ValueError("Experimental data must overlap q=1–6 with positive intensity.")
    experiment /= np.nanmax(experiment)

    # Detect peaks only inside the measured range; do not extrapolate.
    measured = np.flatnonzero(overlap)
    exp_peaks, _ = find_peaks(experiment[measured], prominence=prominence)
    exp_peaks = measured[exp_peaks]
    sim_peaks, _ = find_peaks(spectrum, prominence=prominence)
    candidates = sorted(
        (abs(domain[s] - domain[e]), s, e)
        for s in sim_peaks for e in exp_peaks
        if abs(domain[s] - domain[e]) <= match_tolerance
    )
    used_sim, used_exp, matches = set(), set(), []
    for _, s, e in candidates:
        if s not in used_sim and e not in used_exp:
            used_sim.add(s)
            used_exp.add(e)
            matches.append(s)
    matches.sort()

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(domain, experiment, color="darkorange", marker="o", markersize=3, lw=1.5, label="Experiment")
    simulation_line, = ax.plot([], [], color="tab:blue", lw=1.8, label="Simulation")
    markers = [
        ax.axvline(domain[s], color="seagreen", ls="--", lw=1, alpha=0.7,
                   visible=False, label="Matched peak" if i == 0 else "_nolegend_")
        for i, s in enumerate(matches)
    ]
    annotations = (
        annotate_reflections(ax, domain, spectrum, freqs, hkls,
                             prominence=prominence, visible=False,
                             experiment=experiment, match_tolerance=match_tolerance)
        if hkls is not None else []
    )
    status = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel=r"$q$ ($\AA^{-1}$)", ylabel="Normalized intensity",
           title="Corundum powder diffraction", xlim=(1, 6), ylim=(-0.03, 1.3))
    ax.legend(loc="upper left")

    hold_frames = round(initial_pause * fps)
    reveal_frames = max(2, round(duration * fps))
    total_frames = hold_frames + reveal_frames + round(final_pause * fps)

    def update(frame):
        progress = np.clip((frame - hold_frames) / (reveal_frames - 1), 0, 1)
        count = int(progress * len(domain))
        simulation_line.set_data(domain[:count], spectrum[:count])
        for s, marker in zip(matches, markers):
            marker.set_visible(s < count)
        for s, annotation in annotations:
            annotation.set_visible(s < count)
        status.set_text(
            "Experimental data" if frame < hold_frames else
            f"Matched peaks: {sum(s < count for s in matches)} / {len(matches)}"
        )
        return simulation_line, status, *markers, *(a for _, a in annotations)

    animation = FuncAnimation(
        fig, update, frames=total_frames, init_func=lambda: update(0),
        interval=1000 / fps, repeat=False, blit=False,
    )
    if save_path is not None:
        writer = "pillow" if Path(save_path).suffix.lower() == ".gif" else "ffmpeg"
        animation.save(str(save_path), writer=writer, fps=fps, dpi=150)
    plt.show()
    return animation


if __name__ == "__main__":
    al_positions = generate_positions(Al)
    o_positions = generate_positions(O)

    print("Al atoms:", len(al_positions))
    print("O atoms:", len(o_positions))
    print("Total:", len(al_positions) + len(o_positions))

    assert len(al_positions) == 12
    assert len(o_positions) == 18

    lmbda = 1.54056

    reflexes, freqs, hkls = conduct_diffractometry(
        al_positions, o_positions, lmbda = lmbda, return_hkls=True
    )

    animation = animate_results(reflexes, freqs, lmbda, hkls, save_path = PathManager.FIGURES  / "corundum_animation_an.gif")
