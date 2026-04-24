import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 0.2, 0.2

# Workspace limits
limits = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.4, 0.0)
}

# -----------------------------
# HEXAPOD LEGS (IK INPUTS)
# -----------------------------
legs = {
    "FR": [0.0, 0.0, -0.3],
    "FL": [0.0, 0.0, -0.3],
    "MR": [0.0, 0.0, -0.3],
    "ML": [0.0, 0.0, -0.3],
    "RR": [0.0, 0.0, -0.3],
    "RL": [0.0, 0.0, -0.3]
}

defaults = {k: v[:] for k, v in legs.items()}

# -----------------------------
# HEXAGON BODY
# -----------------------------
hex_radius = 0.35
angles = np.deg2rad([0, 60, 120, 180, 240, 300])

body = np.array([
                    [hex_radius * np.cos(a), hex_radius * np.sin(a), 0]
                    for a in angles
                ] + [[hex_radius * np.cos(angles[0]), hex_radius * np.sin(angles[0]), 0]])

base_pos = {
    "FR": body[5],
    "FL": body[4],
    "ML": body[3],
    "MR": body[0],
    "RL": body[2],
    "RR": body[1]
}


# -----------------------------
# IK 
# -----------------------------
def ik(x, y, z, flip_elbow=False):
    t1 = np.arctan2(y, -z)

    R = np.sqrt(y ** 2 + z ** 2)
    R = max(R, 1e-6)

    D = x ** 2 + R ** 2
    cos_t3 = (D - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)

    # Standard t3 is Elbow Down
    t3 = np.arccos(cos_t3)

    # If flip_elbow is True (for right legs), make it Elbow Up
    if flip_elbow:
        t3 = -t3

    alpha = np.arctan2(x, R)
    beta = np.arctan2(L2 * np.sin(t3), L1 + L2 * np.cos(t3))

    t2 = alpha - beta

    return t1, t2, t3


# -----------------------------
# FK 
# -----------------------------
def fk(t1, t2, t3):
    X = L1 * np.sin(t2) + L2 * np.sin(t2 + t3)
    R = L1 * np.cos(t2) + L2 * np.cos(t2 + t3)
    Y = R * np.sin(t1)
    Z = -R * np.cos(t1)
    return np.array([X, Y, Z])


# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


def draw():
    ax.clear()

    # Draw body
    ax.plot(body[:, 0], body[:, 1], body[:, 2], linewidth=3)

    angles_dict = {}

    for leg, (x, y, z) in legs.items():
        base = base_pos[leg]

        # Logic: Right legs (FR, MR, RR) = Elbow Up (flip_elbow=True)
        # Left legs (FL, ML, RL) = Elbow Down (flip_elbow=False)
        is_right = leg in ["FR", "MR", "RR"]

        # IK call with the specific elbow toggle
        t1, t2, t3 = ik(x, y, z, flip_elbow=is_right)
        angles_dict[leg] = np.degrees([t1, t2, t3])

        # Knee
        knee = base + np.array([
            L1 * np.sin(t2),
            L1 * np.cos(t2) * np.sin(t1),
            -L1 * np.cos(t2) * np.cos(t1)
        ])

        # Foot
        foot = base + fk(t1, t2, t3)

        # Draw leg segments
        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]],
                marker='o', linewidth=2)
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]],
                marker='o', linewidth=2)

        # Target point
        ax.scatter(base[0] + x, base[1] + y, base[2] + z, s=25, color='red')

    # -----------------------------
    # INFO BOX
    # -----------------------------
    info_text = "        θ1     θ2     θ3\n"
    for leg in ["FR", "FL", "MR", "ML", "RR", "RL"]:
        t1, t2, t3 = angles_dict[leg]
        info_text += f"{leg}: {t1:6.1f} {t2:6.1f} {t3:6.1f}\n"

    ax.text2D(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.4])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# -----------------------------
# Tkinter UI
# -----------------------------
root = tk.Tk()
root.title("Hexapod IK Simulator (L-Down, R-Up)")
root.geometry("1200x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)


# -----------------------------
# Update
# -----------------------------
def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()


sliders = []

for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    for i, name in enumerate(["X", "Y", "Z"]):
        s = tk.Scale(
            f,
            from_=limits[name.lower()][0],
            to=limits[name.lower()][1],
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=name,
            command=lambda v, l=leg, j=i: update(l, j, v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))


# -----------------------------
# Reset
# -----------------------------
def reset():
    for l in legs:
        legs[l] = defaults[l][:]
    for s, l, i in sliders:
        s.set(legs[l][i])
    draw()
    canvas.draw()


tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()