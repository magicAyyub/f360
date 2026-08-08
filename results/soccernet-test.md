# SoccerNet tracking, test split

Baseline measurement of the current vision layer on all 49 sequences of the
SoccerNet Tracking test split, 36750 frames at 1080p25, 530418 ground-truth
boxes. The ball tracklet is excluded from ground truth throughout: it is
annotated but a few pixels wide, and the detector only produces people.
Referees and goalkeepers are kept.

Reproduce with:

```bash
uv run run-sequence data/SoccerNet/test --output-dir outputs/soccernet
uv run eval-dataset data/SoccerNet/test outputs/soccernet --show-sequences

uv run run-sequence data/SoccerNet/test --output-dir outputs/soccernet-oracle --oracle
uv run eval-dataset data/SoccerNet/test outputs/soccernet-oracle
```

## Results

| Configuration | HOTA | DetA | AssA | MOTA | IDF1 | Recall | ID switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv5su + ByteTrack | 44.17 | 59.42 | 33.11 | 73.25 | 49.67 | 79.91 | 5858 |
| Oracle detections + ByteTrack | 82.31 | 97.59 | 69.42 | 97.02 | 76.96 | 98.01 | 5251 |
| sn-tracking published baseline | 42.38 | 34.41 | 52.21 | | | | |
| Kalisteo, 2023 challenge winner | 75.61 | 75.38 | 75.94 | | | | |

Oracle detections are the ground-truth boxes with track identities stripped,
which is what SoccerNet ships as `det/det.txt`. Feeding them to the same tracker
gives the score under perfect detection.

## What it says

The harness is sound. On the same test set the pipeline scores 44.17 against a
published 42.38, so the numbers land where a standard detector and tracker
should land rather than somewhere impossible.

The composition is inverted against the baseline. Detection is much stronger
(59.42 against 34.41) and association much weaker (33.11 against 52.21). The
overall similarity hides two errors pointing opposite ways.

Detection dominates the error budget. Replacing our detector with perfect boxes
moves HOTA from 44.17 to 82.31, so 38 of the missing points sit behind detection
alone. Across clips, recall correlates 0.69 with HOTA and 0.51 with AssA:
missing detections fragment tracks, and fragmentation costs association more
than mismatching does. ID switches barely move between the two runs, 5858
against 5251, while AssA doubles, which is the same story from the other side.

There is a real association ceiling underneath that. Even with perfect
detection, AssA reaches only 69.42 and 5251 switches remain, so roughly 30
points of association error belong to the tracker itself and no detector will
recover them.

Per-clip spread is wide: HOTA from 26.92 to 65.62, standard deviation 7.88, and
recall from 56.1% to 93.4%. Single-clip numbers are not trustworthy on their
own; the earlier SNMOT-116 measurement gave 40.25 against 44.17 for the full
set.

## Caveats

The combined score pools detection counts and weights association by true
positives, following TrackEval, so it is not the mean of the per-clip scores.
The mean of clips is 43.66 against a combined 44.17.

The published baseline and the challenge winner are quoted from the sn-tracking
repository. They were produced by different methods on this same test split, so
the comparison is like-for-like on data but not on method.

No parameter was tuned on this split. ByteTrack runs at its defaults with the
frame rate taken from each sequence.

## Next

Detection recall is the bottleneck worth attacking first, and a football-tuned
detector against generic COCO-person is the obvious experiment. The association
ceiling is the second, and worth revisiting only once detection stops dominating.
