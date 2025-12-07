# ONNX Model Specification (Image, Video, Audio)

This document defines the ONNX requirements for **image**, **video**, and **audio** discriminative models used with GASBench on Subnet 34.

---

## 1. Key Requirements

### 1.1 Input Shape

Your ONNX models must specify shapes suitable for batching with **fixed spatial/temporal dimensions**:

---

### **Image models**
- Shape: `['batch_size', 3, H, W]`
- `H` and `W` must be fixed (e.g., 224, 256, 384)
- GASBench resizes & center-crops to your declared dimensions

⚠️ If dynamic `H/W` is used, GASBench will **default to 224×224**.

---

### **Video models**
- Shape: `['batch_size', frames, 3, H, W]`
- `H` and `W` fixed; `frames` may be dynamic
- Model should internally perform temporal aggregation so the output is `(batch_size, num_classes)`

---

### **Audio models**
- Shape: `['batch_size', 96000]`
- Sample rate: **16 kHz**
- Duration: **6.0 seconds**
- Only batch dimension is dynamic

---

## 1.2 Input Value Range

### Image / Video Inputs
- dtype: `uint8`
- range: `[0, 255]`
- GASBench handles: resize → crop → augmentations
- Your model wrapper handles: `[0,255] → [0,1]` + normalization (e.g., ImageNet mean/std)

---

### Audio Inputs
- dtype: `float32`
- shape: `(batch, 96000)`
- range: `[-1, 1]` (standard PCM→float32 conversion, no normalization)

GASBench handles:
1. mono conversion  
2. resample → 16k  
3. crop/pad → 96000 samples  

Your model wrapper handles any normalization (e.g., mean/std) if needed.

> **Note:** No peak/loudness normalization is applied. This preserves loudness and dynamic range cues that may distinguish real from synthetic audio. This matches image/video preprocessing where raw `uint8` values are passed.

---

## 1.3 Output Format (logits)

Models must output logits for:

### Preferred (binary, 2-class)
`[real, synthetic]`  
Shape: `(batch, 2)`

### Supported (3-class)
`[real, semisynthetic, synthetic]`  
Shape: `(batch, 3)`

GASBench always reduces predictions to a **binary** decision.

---

## 2. GASBench Preprocessing

### 2.1 Image / Video
- resize shortest edge
- center crop `(H, W)`
- random augmentations (rotation, jitter, JPEG, etc.)
- batching
- fed to ONNX as `uint8`

### 2.2 Audio
- convert to mono
- resample to 16k
- crop/pad to 6s
- fed to ONNX as `float32` `(batch, 96000)` in `[-1, 1]` (no normalization)

---

## 3. Export Examples

### Image Export

```python
dummy = torch.randint(0,256,(1,3,224,224),dtype=torch.uint8)

torch.onnx.export(
    wrapped,
    dummy,
    "image_detector.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=11,
)
```

### Video Export

```python
dummy = torch.randint(0,256,(1,8,3,224,224),dtype=torch.uint8)

torch.onnx.export(
    wrapped,
    dummy,
    "video_detector.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0:"batch",1:"frames"},"logits":{0:"batch"}},
    opset_version=11,
)
```

### Audio Export

```python
dummy = torch.zeros((1,96000),dtype=torch.float32)

torch.onnx.export(
    wrapped,
    dummy,
    "audio_detector.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input":{0:"batch"},"logits":{0:"batch"}},
    opset_version=11,
)
```

---

## 4. Packaging (SN34 submission only)

Submitted models must be zipped before uploading with [gascli](https://github.com/BitMind-AI/bitmind-subnet/blob/main/docs/Discriminative-Mining.md). Each zip must contain exactly **one** `.onnx` file.

```
zip image_detector.zip image_detector.onnx
zip video_detector.zip video_detector.onnx
zip audio_detector.zip audio_detector.onnx
```

