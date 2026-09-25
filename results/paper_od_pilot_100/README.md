# Separate COCO OD pilot (not a V2-C AR result)

The [raw result](probe_bgsuppress.json) came from the COCO val2017 visual/evaluation
notebook defined by [`kaggle/paper_coco_visual_cell.sh`](../../kaggle/paper_coco_visual_cell.sh):
100 images selected with seed 20260924, 320-pixel letterbox, QPs 35/40/45,
frozen MobileNet Faster R-CNN for the protection mask, and a different frozen
ResNet50 Faster R-CNN for COCO mAP@[.5:.95]. The intervention is `blur4`
background suppression; it is **not** the six-candidate V2-C AR selector.
The percentile intervals use only 100 paired image-bootstrap draws.

| Codec | OD BD-rate vs codec-only | Paired bootstrap 95% interval |
|---|---:|---:|
| H.264 | −13.55% | [−22.20%, −3.00%] |
| H.265 | −7.81% | [−16.09%, +2.66%] |

The H.265 interval includes zero; this pilot does not establish a negative OD
BD-rate for both codecs. Even the H.264 point and interval are exploratory at
this small sample and low bootstrap count. OD mAP and AR Top-1 are different
outcomes on different datasets under different interventions. Do not combine
them into a single V2-C or unified AR/OD success claim.

SHA-256 of the downloaded Kaggle `probe_bgsuppress.json` and this exact JSON
before commit: `438e47b31ee5ea6fdb58473075f30cd21c4a49f3e636eb65802a71974051fa41`.
