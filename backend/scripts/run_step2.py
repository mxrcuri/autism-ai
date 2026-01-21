from pipelines.step2_preprocessing.preprocess import preprocess_video

out = preprocess_video("storage/raw_videos/test_video.mp4")

print("Usable:", out["usable"])
print("Reason:", out["reason"])
print("Stats:")
for k, v in out["stats"].items():
    print(f"  {k}: {v}")

