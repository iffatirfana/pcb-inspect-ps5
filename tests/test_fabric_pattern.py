import cv2
import numpy as np

import fabric_pattern as P
import fabric_profile as fp


def grid(missing=()):
    """Cream cloth with a 5x6 grid of dark discs, 190 px apart (about 38 mm at 5 px/mm)."""
    img = np.full((1100, 1300, 3), 225, np.uint8)
    for r in range(5):
        for c in range(6):
            if (r, c) in missing:
                continue
            cv2.circle(img, (160 + c * 190, 160 + r * 190), 62, (60, 60, 120), -1)
    return img


def test_clean_grid_has_no_gap():
    r = P.find_gaps(grid())
    assert r["regular"] and r["n_motifs"] >= 20 and r["gaps"] == []


def test_missing_motif_found_at_right_place():
    r = P.find_gaps(grid(missing={(2, 3)}))
    assert len(r["gaps"]) == 1
    x, y = r["gaps"][0]
    assert abs(x - (160 + 3 * 190)) < 40 and abs(y - (160 + 2 * 190)) < 40


def test_plain_cloth_is_not_regular():
    r = P.find_gaps(np.full((1100, 1300, 3), 225, np.uint8))
    assert not r["regular"] and r["gaps"] == []


def test_findings_are_low_severity():
    _, f = P.pattern_findings(grid(missing={(2, 3)}), 5.0, fp.zone_3x3)
    assert f and all(x["severity"] == "Low" and x["type"] == "Pattern gap" for x in f)
