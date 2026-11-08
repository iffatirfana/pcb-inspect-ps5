import numpy as np
import onnx
from onnx import TensorProto, helper

from fabric_scorer import FabricScorer, MEAN, STD, gray_world

W = 512


def make_onnx(path):
    """Tiny stand-in backbone: 8x8 average pool -> (1,3,64,64) 'features'."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, W, W])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, W // 8, W // 8])
    node = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[8, 8], strides=[8, 8])
    m = helper.make_model(helper.make_graph([node], "g", [x], [y]),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def normal_img(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((W, W, 3)) * 20 + 120).astype(np.uint8)


def feats(img_bgr):
    rgb = gray_world(img_bgr)[:, :, ::-1].astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD)
    p = x.reshape(64, 8, 64, 8, 3).mean((1, 3))
    return p.reshape(-1, 3)


def test_scorer_flags_defect_region(tmp_path):
    make_onnx(tmp_path / "b.onnx")
    bank = np.concatenate([feats(normal_img(s)) for s in range(5)])
    np.save(tmp_path / "bank.npy", bank)
    sc = FabricScorer(tmp_path / "b.onnx", tmp_path / "bank.npy", (W, W), 5)

    clean = normal_img(99)
    hm_clean = sc.heatmap(clean)
    bad = clean.copy()
    bad[200:260, 300:360] = 20                       # dark 'stain'
    hm_bad = sc.heatmap(bad)

    assert hm_bad.shape == bad.shape[:2]
    assert hm_bad[230, 330] > 5 * hm_clean.max()      # defect lights up
    assert hm_bad[20, 20] < hm_bad[230, 330] / 5      # far from defect stays low


def test_feature_size_mismatch_raises(tmp_path):
    import pytest
    make_onnx(tmp_path / "b.onnx")
    np.save(tmp_path / "bank.npy", np.zeros((10, 7), np.float32))
    sc = FabricScorer(tmp_path / "b.onnx", tmp_path / "bank.npy", (W, W), 5)
    with pytest.raises(ValueError):
        sc.heatmap(normal_img(1))


def test_gray_world_removes_colour_and_brightness_drift():
    base = normal_img(7).astype(np.float32)
    tinted = np.clip(base * np.array([1.5, 1.0, 0.7]) * 0.8, 0, 255).astype(np.uint8)  # bluish + darker
    plain = base.astype(np.uint8)
    a, b = gray_world(plain).reshape(-1, 3).mean(0), gray_world(tinted).reshape(-1, 3).mean(0)
    assert np.abs(a - 128).max() < 3 and np.abs(b - 128).max() < 3   # both land on the same grey level


def test_tinted_clean_photo_not_flagged(tmp_path):
    make_onnx(tmp_path / "b.onnx")
    np.save(tmp_path / "bank.npy", np.concatenate([feats(normal_img(s)) for s in range(5)]))
    sc = FabricScorer(tmp_path / "b.onnx", tmp_path / "bank.npy", (W, W), 5)
    clean = normal_img(42)
    tinted = np.clip(clean.astype(np.float32) * np.array([1.4, 1.0, 0.8]) * 0.85, 0, 255).astype(np.uint8)
    bad = clean.copy(); bad[200:260, 300:360] = 20
    assert sc.heatmap(tinted).max() < 0.5 * sc.heatmap(bad).max()
