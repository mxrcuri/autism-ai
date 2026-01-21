from pipelines.step2_preprocessing.preprocess import preprocess_video
from pipelines.step3_pose_gaze.extract import run_step3
import numpy as np

# --- Run Step 2 ---
step2 = preprocess_video("storage/raw_videos/test_video.mp4")

# --- Run Step 3 (25 Hz for now) ---
sequence = run_step3(step2)   # no stride yet

# ---------------- SANITY CHECKS ---------------- #
# Print one example frame
for s in sequence:
    if s["valid"]:
        print("Example frame:", s)
        break

# 1️⃣ Sequence length
print("Sequence length:", len(sequence))

# 2️⃣ Missing pose frames
missing = sum(1 for s in sequence if s["pose"] is None)
print("Missing pose frames:", missing)

# 3️⃣ Plot left wrist X trajectory
import matplotlib.pyplot as plt

xs = [s["pose"]["left_wrist"][0] for s in sequence if s["pose"] is not None]

plt.figure(figsize=(10, 4))
plt.plot(xs)
plt.xlabel("Time step (25 Hz)")
plt.ylabel("Left wrist X (normalized)")
plt.title("Left wrist X over time")
plt.grid(True)
plt.tight_layout()
plt.savefig("left_wrist_x.png")
print("Saved plot to left_wrist_x.png")

gxs = [s["gaze"]["gx"] for s in sequence if s["gaze"] is not None]

plt.figure(figsize=(10, 4))
plt.plot(gxs)
plt.xlabel("Time step (25 Hz)")
plt.ylabel("Gaze gx")
plt.title("Horizontal gaze over time")
plt.grid(True)
plt.savefig("gaze_gx.png")
print("Saved gaze_gx.png")

yaws = [s["head"]["yaw"] for s in sequence if s["head"] is not None]
pitch = [s["head"]["pitch"] for s in sequence if s["head"] is not None]

plt.figure(figsize=(10, 4))
plt.plot(yaws, label="yaw")
plt.plot(pitch, label="pitch")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Angle proxy")
plt.title("Head pose over time")
plt.grid(True)
plt.savefig("head_pose.png")
print("Saved head_pose.png")

xs = np.array([s["pose"]["left_wrist"][0] for s in sequence if s["pose"]])
vx = np.diff(xs)

plt.figure(figsize=(10, 4))
plt.plot(vx)
plt.xlabel("Time step")
plt.ylabel("Δ wrist X")
plt.title("Left wrist horizontal velocity")
plt.grid(True)
plt.savefig("wrist_velocity.png")
print("Saved wrist_velocity.png")

