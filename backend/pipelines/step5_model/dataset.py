import numpy as np
from sklearn.preprocessing import StandardScaler


class WindowSequenceDataset:
    """
    Builds (N, T, F) tensors from Step-4 window features.
    Handles scaling correctly.
    """

    def __init__(
        self,
        features,
        seq_len=10,
        scaler=None,
        fit_scaler=False
    ):
        assert len(features) > seq_len, "Not enough windows"

        self.keys = list(features[0].keys())

        X = np.array(
            [[f[k] for k in self.keys] for f in features],
            dtype=np.float32
        )

        # ---------- SCALING ----------
        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        self.scaler = scaler
        self.X = self._build_sequences(X, seq_len)

    def _build_sequences(self, X, seq_len):
        seqs = []
        for i in range(len(X) - seq_len + 1):
            seqs.append(X[i:i + seq_len])
        return np.array(seqs, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

