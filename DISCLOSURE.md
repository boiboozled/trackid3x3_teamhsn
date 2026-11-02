# Disclosure: TrackID3x3 (9th ABC Challenge) submission

Predictions for IMG_0114 and IMG_0115 were produced on 2 Nov 2026 with

```
python scripts/run_pipeline.py +preset=final run_name=final_submission
python scripts/make_submission.py --run-dir outputs/runs/2026-11-02_20-21-02_final_submission
```

(MLflow run `final_submission`, experiment `trackid3x3`). `configs/preset/final.yaml` holds the full configuration.

## Pretrained models (public, general purpose, not trained on TrackID3x3)

| Model | Source | Used as |
|---|---|---|
| YOLO11m (COCO) | Ultralytics `yolo11m.pt` | Initialisation of the player detector |
| ConvNeXt-Tiny, ImageNet-22k | timm `convnext_tiny.fb_in22k` (Hugging Face hub) | Initialisation of the player classifier |

## Training data

Only the official Training set: the eight training videos IMG_0104–IMG_0111, with their player boxes, court positions, teams and jersey numbers.
- **Detector and classifier:** trained on IMG_0104–0107 and IMG_0110–0111. IMG_0108 and IMG_0109 were held out for evaluation.
- **Colour features:** the squad colour prototypes and the 10 cm annotation-bias correction come from training videos only.

Nothing was trained, fitted or pseudo-labelled on the Challenge videos. No other TrackID3x3 annotations were used: the public top-view annotations of the Challenge games were deleted unread from our copy of the official repository.

## Manual input

Court landmarks were clicked by hand on one median background image per Challenge video, which the rules allow. The homography uses the 4 court corners: `/data/trackid3x3/plates/court_landmarks.xml`, giving `/data/trackid3x3/registration_clicks/IMG_011{4,5}.npy`. There was no manual player annotation, identity assignment or correction.

## Frozen artefacts (SHA-256)

| File | SHA-256 |
|---|---|
| `models/detector/det_yolo11m_ft/weights/best.pt` | `f9e36f898d7cc8817c5a5c4531b27bd0fa8d4e88084a4a6600d5df10e22455f8` |
| `models/player_cls/cls_v1_convnext_t_20261020_212654/best.pt` | `4196a7402151d7af47333290dfd8afe064d0435466b31e84af39384323e408c0` |
| `yolo11m.pt` (Ultralytics COCO, initialisation) | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| `registration_clicks/IMG_0114.npy` | `3b25ed10b02d6564ae799de9d9c8a5d85d9536e75da50d97448b649526f2102e` |
| `registration_clicks/IMG_0115.npy` | `8b7b27cb0a5e874c6fc447352ce186cd1b3d4cb340c4017dd09462676e1b165d` |
| `plates/court_landmarks.xml` | `26a76414616fca8386d276bce15c0bfe5ebcd5d48005bc0c60c08d76ea37d4e2` |
| `IMG_0114.txt` (submitted) | `72c0d8889dd3b16002767ddb419535fc2db750808a6150f9675ef7814f9ff46e` |
| `IMG_0115.txt` (submitted) | `115e38664ba6d30fa0f9f5f0755a8acf90953a9876ab25b47eb1f0666c7b83a5` |
| `submission.zip` | `e0a6b500fa71c68e907899386f6dd2ef38d222f01f07c29c12b374e743f6b123` |

## Held-out result of the submitted configuration

On training videos IMG_0108 and IMG_0109, which were never used to train the detector or classifier, transcoded to 720p:

| τ | TI-HOTA |
|---|---|
| 1 m (`tau = 100` in the official notebook) | 95.8 |
| 0.5 m | 88.4 |
