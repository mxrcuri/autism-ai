from pipelines.step2_preprocessing.preprocess import preprocess_video
from pipelines.step3_pose_gaze.extract import run_step3
from pipelines.step4_features.extract import extract_features

# Step 2
step2 = preprocess_video("storage/raw_videos/test_video.mp4")

# Step 3
sequence = run_step3(step2)

# Step 4
features = extract_features(sequence)

print("Number of windows:", len(features))
print("Example feature vector:")
print(features[0])

