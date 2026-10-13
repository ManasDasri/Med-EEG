"""Tiny epoch-to-feature-to-export integration plus a failed rerun regression."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import mne
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import preprocess
from config import BIOSEMI64_TO_1020
from extract_features import probe_features
from export_dashboard import build_subjects_payload
with TemporaryDirectory() as tmp:
    root=Path(tmp); channels=list(BIOSEMI64_TO_1020.values())
    rng=np.random.default_rng(0)
    raw=mne.io.RawArray(rng.normal(0,4e-6,(64,256*25)),mne.create_info(channels,256,'eeg'))
    probes=pd.DataFrame({'probe_index':[0], 'probe_onset_s':[24.0], 'window_start_s':[0.0]})
    epochs,_=preprocess.epoch_probes(raw,probes)
    path=root/'tiny-epo.fif'; epochs.save(path,overwrite=True)
    row={'subject':'sub-001','session':'ses-01','recording':'r','probe_index':0,'probe_onset_s':24.,'epochs_file':str(path),'concentration_rating':2}
    feature,epoch_features=probe_features(row,None)
    assert np.isfinite(feature['minimal_ratio']) and len(epoch_features)>0
    assert build_subjects_payload(pd.DataFrame([feature]),None)[0]['probes'][0]['session']=='ses-01'
    (root/'sub-001').mkdir(); (root/'sub-001'/'sub-001_ses-01_task-meditation_eeg.bdf').touch()
    deriv=root/'derivatives';deriv.mkdir()
    pd.DataFrame([row]).to_csv(deriv/'probe_labels.csv',index=False)
    pd.DataFrame([{'subject':'sub-001','session':'ses-01','recording':'r'}]).to_csv(deriv/'preprocessing_qc.csv',index=False)
    before=(deriv/'probe_labels.csv').read_text()
    with patch.object(sys,'argv',['preprocess','--bids_root',str(root),'--subjects','sub-001','--derivatives',str(deriv)]), patch.object(preprocess,'process_recording',side_effect=OSError('simulated damaged raw file')):
        try: preprocess.main()
        except SystemExit as exc: assert exc.code==1
        else: raise AssertionError('Failed preprocessing returned success')
    after=pd.read_csv(deriv/'probe_labels.csv')
    assert after.subject.tolist()==['sub-001'] and after.epochs_file.iloc[0]==str(path)
print('ok: epoch-to-dashboard integration and failed-rerun preservation')
