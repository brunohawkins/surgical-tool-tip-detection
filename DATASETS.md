# External datasets

List every dataset you trained on apart from ours, with a link and its licence. Leave the table empty if you only used ours.

| Dataset | Link | Licence | How you used it |
| --- | --- | --- | --- |
| ImageNet-1k (via timm `resnet18.a1_in1k` / `resnet34.a1_in1k` pretrained weights) | https://huggingface.co/timm/resnet18.a1_in1k | timm weights Apache-2.0; ImageNet terms of access | Backbone initialisation only; all tip training is on the XRlabs data |
| PitVis-2023 (UCL, endoscopic pituitary surgery) | https://doi.org/10.5522/04/26531686 | CC BY-NC-SA 4.0 | Test only (instrument presence on real surgery, `training/eval_pitvis.py`). One rejected experiment used its frames as hard negatives; the submitted model was not trained on it |
| MACS (UCL, microsurgical aneurysm clipping) | https://doi.org/10.5522/04/23533731 | CC BY-NC 4.0 | Qualitative check only (where the model fires on microscope neurosurgery); never trained on |
