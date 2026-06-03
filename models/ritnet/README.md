# RITnet pre-trained weights

The `ritnet` detector module (`pupillab/modules/ritnet_seg.py`) needs the official
RITnet pre-trained weights here:

```
models/ritnet/best_model.pkl
```

The module degrades gracefully if this file is absent — it just shows an
"unavailable" message in the dashboard/montage, so the rest of the harness runs
without it.

## Get the weights

RITnet (Chaudhary et al., *RITnet: Real-time Semantic Segmentation of the Eye for
Gaze Tracking*, ICCVW 2019), MIT-licensed:

- Repo: https://github.com/AayushKrChaudhary/RITnet
- The repo ships the trained model as `best_model.pkl` (released via its README /
  Google Drive link). Download it and place it at `models/ritnet/best_model.pkl`.

The file is a PyTorch `state_dict` for the `DenseNet2D` network, which is vendored
(layer-name-compatible) at `pupillab/models/ritnet_arch.py`. No other files from
the upstream repo are needed.

## Install the runtime

```powershell
pip install -r requirements-ml.txt   # torch, torchvision (CPU is fine)
```

## Notes

- Input is the **green channel** (cleanest under violet light), gamma- and
  CLAHE-normalised and resized to RITnet's 640×400 input — matching the upstream
  preprocessing — then argmax'd into 4 classes (0 bg, 1 sclera, 2 iris, 3 pupil).
  An ellipse is fitted to the pupil class.
- RITnet was trained on **near-IR** head-mounted eye images. Our violet-illuminated
  macro frames are structurally similar but not identical, so treat results as a
  comparison baseline, not ground truth. No fine-tuning is performed.
