from pipelines.step5_model.dream_loader import load_user_sessions
from pipelines.step5_model.dataset import WindowSequenceDataset
from pipelines.step5_model.train import train_autoencoder
from pipelines.step5_model.score import reconstruction_error
from pipelines.step3_pose_gaze.dream_adapter import load_dream_sequence
from pipelines.step3_pose_gaze.extract import run_step3
from pipelines.step4_features.extract import extract_features
from pipelines.step2_preprocessing.preprocess import preprocess_video

# ---------------- CONFIG ----------------
DREAM_ROOT = "/home/kriti/Downloads/snd1156-1-1"
SEQ_LEN = 10


# ---------------- LOAD DATA ----------------
users = load_user_sessions(DREAM_ROOT)

user_features = {}

for user_id, sessions in users.items():
    feats = []
    #for session in sessions:
    #    sequence = run_step3(session)
    #    feats.extend(extract_features(sequence))
    #user_features[user_id] = feats
    


    for session in sessions:
        print(f"Processing {session}")
        sequence = load_dream_sequence(session)
        feats.extend(extract_features(sequence))
    user_features[user_id] = feats


# ---------------- USER SPLIT ----------------
user_ids = sorted(user_features.keys())

n = len(user_ids)
train_ids = user_ids[:int(0.7 * n)]
val_ids   = user_ids[int(0.7 * n):int(0.85 * n)]
test_ids  = user_ids[int(0.85 * n):]


def collect(ids):
    out = []
    for uid in ids:
        out.extend(user_features[uid])
    return out


train_feats = collect(train_ids)
val_feats   = collect(val_ids)
test_feats  = collect(test_ids)


# ---------------- DATASETS ----------------
train_ds = WindowSequenceDataset(
    train_feats,
    seq_len=SEQ_LEN,
    fit_scaler=True
)

val_ds = WindowSequenceDataset(
    val_feats,
    seq_len=SEQ_LEN,
    scaler=train_ds.scaler
)

test_ds = WindowSequenceDataset(
    test_feats,
    seq_len=SEQ_LEN,
    scaler=train_ds.scaler
)


# ---------------- TRAIN ----------------
model = train_autoencoder(train_ds)


# ---------------- SCORE ----------------
scores = reconstruction_error(model, test_ds)

print("\nDeviation statistics (TEST):")
print("Mean:", scores.mean())
print("Std :", scores.std())
print("Max :", scores.max())

