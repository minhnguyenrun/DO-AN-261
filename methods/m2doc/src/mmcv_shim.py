"""MMCV Shim for M2Doc on Python 3.12 / Windows environments.

Provides fallback shims for mmcv.ops (batched_nms, nms, roi_align, RoIAlign)
using torchvision.ops, allowing MMDetection and M2Doc to run seamlessly
without requiring C++ compiler toolchains or binary wheel extensions.
"""

import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
import torch
import torchvision.ops as tv_ops


class AutoModule(types.ModuleType):
    """Dynamic module placeholder that safely returns dummy functions for unneeded C++ ops."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__file__ = f"{name}.py"
        self.__path__ = []

    def __getattr__(self, name: str):
        if name in ("__file__", "__name__", "__path__", "__spec__", "__package__"):
            raise AttributeError(name)

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


class OpsLoader:
    def __init__(self, mod: types.ModuleType):
        self.mod = mod

    def create_module(self, spec):
        return self.mod

    def exec_module(self, module):
        pass


class OpsFinder:
    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        if fullname.startswith("mmcv.ops"):
            if fullname in sys.modules:
                mod = sys.modules[fullname]
            else:
                mod = AutoModule(fullname)
                sys.modules[fullname] = mod
            spec = ModuleSpec(fullname, OpsLoader(mod), is_package=True)
            return spec
        return None


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Pure PyTorch/torchvision implementation of batched NMS."""
    iou_threshold = nms_cfg.get("iou_threshold", 0.5)
    keep = tv_ops.batched_nms(boxes, scores, idxs, iou_threshold)
    max_num = nms_cfg.get("max_num", -1)
    if max_num > 0:
        keep = keep[:max_num]
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
    return dets, keep


def nms(boxes, scores, iou_threshold):
    """Pure PyTorch/torchvision implementation of standard NMS."""
    return tv_ops.nms(boxes, scores, iou_threshold)


def roi_align(input, rois, output_size, spatial_scale=1.0, sampling_ratio=0, pool_mode="avg", aligned=True):
    """Torchvision backed RoIAlign implementation."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return tv_ops.roi_align(
        input,
        rois,
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )


class RoIAlign(torch.nn.Module):
    """PyTorch RoIAlign layer wrapping torchvision.ops.roi_align."""

    def __init__(
        self,
        output_size,
        spatial_scale=1.0,
        sampling_ratio=0,
        pool_mode="avg",
        aligned=True,
        use_torchvision=False,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = aligned

    def forward(self, input, rois):
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            aligned=self.aligned,
        )


_SHIM_INITIALIZED = False


def setup_m2doc_environment(mmdet_dir: str = None):
    """Initializes the MMCV shims and registers M2Doc mmdetection path."""
    global _SHIM_INITIALIZED
    if _SHIM_INITIALIZED:
        return

    # Add OpsFinder to meta_path
    if OpsFinder not in sys.meta_path:
        sys.meta_path.insert(0, OpsFinder)

    ops_mod = AutoModule("mmcv.ops")
    nms_mod = AutoModule("mmcv.ops.nms")
    roi_align_mod = AutoModule("mmcv.ops.roi_align")

    nms_mod.batched_nms = batched_nms
    nms_mod.nms = nms
    ops_mod.nms = nms_mod
    ops_mod.batched_nms = batched_nms

    roi_align_mod.roi_align = roi_align
    roi_align_mod.RoIAlign = RoIAlign
    ops_mod.roi_align = roi_align
    ops_mod.RoIAlign = RoIAlign

    sys.modules["mmcv.ops"] = ops_mod
    sys.modules["mmcv.ops.nms"] = nms_mod
    sys.modules["mmcv.ops.roi_align"] = roi_align_mod

    # Monkeypatch mmcv version check
    try:
        import mmcv

        mmcv.__version__ = "2.1.0"
        mmcv.ops = ops_mod
        import mmcv.cnn.bricks.transformer as trans

        trans.MultiScaleDeformableAttention = type(
            "MultiScaleDeformableAttention", (torch.nn.Module,), {}
        )
    except Exception as e:
        print(f"[mmcv_shim] Warning initializing mmcv: {e}")

    # Add custom mmdetection to sys.path
    if mmdet_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        mmdet_dir = str(base_dir / "mmdetection")

    if mmdet_dir not in sys.path:
        sys.path.insert(0, mmdet_dir)

    try:
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
    except Exception:
        pass

    _SHIM_INITIALIZED = True
