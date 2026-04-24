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

# -----------------------------
# Base pose
# -----------------------------
base_pose = [0, 0, 0, 0, 0, 0]
base_defaults = base_pose[:]

# -----------------------------
# HEXAPOD GEOMETRY
# -----------------------------
hex_radius = 0.35
angles = np.deg2rad([0, 60, 120, 180, 240, 300])

body = np.array([
                    [hex_radius * np.cos(a), hex_radius * np.sin(a), 0]
                    for a in angles
                ] + [[hex_radius * np.cos(angles[0]), hex_radius * np.sin(angles[0]), 0]])

hip_offsets = {
    "FR": body[5],
    "FL": body[4],
    "ML": body[3],
    "MR": body[0],
    "RL": body[2],
    "RR": body[1]
}


# -----------------------------
# FK 
# -----------------------------
def fk(t1, t2, t3):
    X = L1 * np.sin(t2) + L2 * np.sin(t2 + t3)
    Y = (L1 * np.cos(t2) + L2 * np.cos(t2 + t3)) * np.sin(t1)
    Z = -(L1 * np.cos(t2) + L2 * np.cos(t2 + t3)) * np.cos(t1)
    return np.array([X, Y, Z])


# -----------------------------
# DEFAULT ANGLES
# -----------------------------
default_angles = {
    "FR": [0.0, 0.5, -1.0],
    "FL": [0.0, -0.5, 1.0],
    "MR": [0.0, 0.5, -1.0],
    "ML": [0.0, -0.5, 1.0],
    "RR": [0.0, 0.5, -1.0],
    "RL": [0.0, -0.5, 1.0]
}

# -----------------------------
# COMPUTE FIXED FEET
# -----------------------------
feet_world = {}
for leg in default_angles:
    t1, t2, t3 = default_angles[leg]
    feet_world[leg] = hip_offsets[leg] + fk(t1, t2, t3)


# -----------------------------
# ROTATION MATRIX
# -----------------------------
def rot_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


# -----------------------------
# IK 
# -----------------------------
def ik(x, y, z, flip_elbow=False):
    t1 = np.arctan2(y, -z)

    R = np.sqrt(y ** 2 + z ** 2)
    R = max(R, 1e-6)

    D = x ** 2 + R ** 2
    c = np.clip((D - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)

    # Standard positive t3 = Elbow Down
    t3 = np.arccos(c)

    # Flip t3 to negative for Elbow Up (Right legs)
    if flip_elbow:
        t3 = -t3

    t2 = np.arctan2(x, R) - np.arctan2(
        L2 * np.sin(t3),
        L1 + L2 * np.cos(t3)
    )

    return t1, t2, t3


# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


def draw():
    ax.clear()

    bx, by, bz, roll, pitch, yaw = base_pose
    Rb = rot_matrix(roll, pitch, yaw)
    base = np.array([bx, by, bz])

    # Body
    body_world = (Rb @ body.T).T + base
    ax.plot(body_world[:, 0], body_world[:, 1], body_world[:, 2], linewidth=3)

    angles_dict = {}

    for leg in feet_world:
        foot_w = feet_world[leg]
        hip_w = base + Rb @ hip_offsets[leg]

        foot_local = Rb.T @ (foot_w - hip_w)

        # Logic: Right legs (FR, MR, RR) set to Elbow Up
        is_right = leg in ["FR", "MR", "RR"]
        t1, t2, t3 = ik(*foot_local, flip_elbow=is_right)

        angles_dict[leg] = np.degrees([t1, t2, t3])

        # Knee
        knee = hip_w + Rb @ np.array([
            L1 * np.sin(t2),
            L1 * np.cos(t2) * np.sin(t1),
            -L1 * np.cos(t2) * np.cos(t1)
        ])

        # Foot
        foot_calc = hip_w + Rb @ fk(t1, t2, t3)

        # Draw leg segments
        ax.plot([hip_w[0], knee[0]],
                [hip_w[1], knee[1]],
                [hip_w[2], knee[2]], marker='o', linewidth=2)

        ax.plot([knee[0], foot_calc[0]],
                [knee[1], foot_calc[1]],
                [knee[2], foot_calc[2]], marker='o', linewidth=2)

        ax.scatter(*foot_w, s=30, color='red')

    # INFO BOX
    info = "         θ1     θ2     θ3\n"
    for leg in ["FR", "FL", "MR", "ML", "RR", "RL"]:
        t1, t2, t3 = angles_dict.get(leg, [0, 0, 0])
        info += f"{leg}: {t1:6.1f} {t2:6.1f} {t3:6.1f}\n"

    ax.text2D(0.02, 0.98, info,
              transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8),
              va='top')

    ax.set_xlim([-0.8, 0.8]);
    ax.set_ylim([-0.8, 0.8]);
    ax.set_zlim([-0.8, 0.4])
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.set_zlabel("Z")


# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Hexapod IK (Left: Down, Right: Up)")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel = tk.Frame(root)
panel.pack(side=tk.RIGHT, fill=tk.Y)


def update(i, val):
    base_pose[i] = float(val)
    draw()
    canvas.draw_idle()


labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
limits = [
    (-0.2, 0.2), (-0.2, 0.2), (-0.35, 0.05),
    (-0.17, 0.17), (-0.17, 0.17), (-0.6, 0.6)
]

sliders = []
for i, name in enumerate(labels):
    s = tk.Scale(panel, from_=limits[i][0], to=limits[i][1],
                 resolution=0.01, orient=tk.HORIZONTAL, label=name,
                 command=lambda v, j=i: update(j, v))
    s.set(base_pose[i])
    s.pack(fill="x")
    sliders.append((s, i))


def reset():
    for i in range(len(base_pose)):
        base_pose[i] = base_defaults[i]
    for s, i in sliders:
        s.set(base_pose[i])
    draw()
    canvas.draw()


tk.Button(panel, text="Reset", command=reset).pack(fill="x", pady=10)

draw()
canvas.draw()
root.mainloop()