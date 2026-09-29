import argparse
from pathlib import Path

import mne
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Locate the subject's files in the BIDS dataset (works on any OS)
parser = argparse.ArgumentParser(description="Single-subject KNN prototype.")
parser.add_argument("--bids_root", required=True, type=Path,
                    help="path to the ds001787 BIDS dataset")
parser.add_argument("--subject", default="024")
parser.add_argument("--session", default="01")
args = parser.parse_args()

stem = f"sub-{args.subject}_ses-{args.session}_task-meditation"
eeg_dir = args.bids_root / f"sub-{args.subject}" / f"ses-{args.session}" / "eeg"

raw = mne.io.read_raw_bdf(eeg_dir / f"{stem}_eeg.bdf", preload=True)

# 2. Load events.tsv
events_df = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")


# Map response values to labels
label_map = {2: "mind_wandering", 4: "concentration", 8: "tiredness"}

# Extract stimulus onsets (value=128) and following responses
events = []
labels = []
for i, row in events_df.iterrows():
    if row['value'] == 128:  # stimulus
        # look ahead for next response
        resp_rows = events_df[(events_df['onset'] > row['onset']) & (events_df['trial_type'] == 'response')]
        if not resp_rows.empty:
            resp_val = resp_rows.iloc[0]['value']
            if resp_val in label_map:
                onset_sample = int(row['sample'])
                events.append([onset_sample, 0, resp_val])
                labels.append(label_map[resp_val])

events = np.array(events)

# 3. Create epochs around stimulus
event_id = {v: k for k, v in label_map.items()}
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.8,
                    baseline=(None, 0), preload=True)

# 4. Extract theta (4–8 Hz) and gamma (30–50 Hz) band power
theta = epochs.copy().filter(4, 8).get_data()
gamma = epochs.copy().filter(30, 50).get_data()

theta_power = theta.var(axis=2).mean(axis=1)
gamma_power = gamma.var(axis=2).mean(axis=1)

X = np.vstack([theta_power, gamma_power]).T
y = labels

# 5. Train simple kNN classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
