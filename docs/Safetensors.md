# Safetensors Model Specification

This document defines the requirements for **safetensors** format models used with GASBench on Subnet 34.

> **Note:** ONNX format is no longer accepted for competition submissions. All new models must use safetensors format.

---

## 1. Required Files

Your submission must be a directory (or ZIP archive) containing:

```
my_model/
├── model_config.yaml    # Model metadata and preprocessing config
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights (or *.safetensors)
```

All three files are **required**.

---

## 2. model_config.yaml

The config file defines model metadata and preprocessing settings.

### Image Model Config

```yaml
name: "my-image-detector"
version: "1.0.0"
modality: "image"

preprocessing:
  resize: [224, 224]       # Target [H, W] - must match model input
  normalize:               # Optional - applied after uint8->float conversion
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

model:
  num_classes: 2           # Required: 2 for [real, synthetic]
  weights_file: "model.safetensors"  # Optional, defaults to model.safetensors
```

### Video Model Config

```yaml
name: "my-video-detector"
version: "1.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]       # Target [H, W] for each frame
  max_frames: 16           # Maximum frames to sample

model:
  num_classes: 2
  weights_file: "model.safetensors"
```

### Audio Model Config

```yaml
name: "my-audio-detector"
version: "1.0.0"
modality: "audio"

preprocessing:
  sample_rate: 16000       # Target sample rate (Hz)
  duration_seconds: 6.0    # Target duration (samples = rate * duration)

model:
  num_classes: 2
  weights_file: "model.safetensors"
```

---

## 3. model.py Requirements

Your `model.py` must define a `load_model()` function:

```python
def load_model(weights_path: str, num_classes: int = 2) -> torch.nn.Module:
    """
    Load the model with pretrained weights.
    
    This is the required entry point called by gasbench.
    
    Args:
        weights_path: Path to the .safetensors weights file
        num_classes: Number of output classes (from config)
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    model = YourModel(num_classes=num_classes)
    state_dict = load_file(weights_path)  # from safetensors.torch
    model.load_state_dict(state_dict)
    model.train(False)  # Set to eval mode
    return model
```

---

## 4. Input/Output Specifications

### Image Models

**Input:**
- Shape: `[batch_size, 3, H, W]`
- Data type: `uint8`
- Value range: `[0, 255]`
- Color format: RGB

**Output:**
- Shape: `[batch_size, num_classes]`
- Type: Logits (raw scores, before softmax)
- Classes: `[real, synthetic]` for 2-class

Your model's `forward()` should handle uint8 input and convert to float internally:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, 3, H, W] uint8 [0, 255]
    x = x.float() / 255.0  # Convert to float [0, 1]
    # ... your model logic ...
    return logits  # [B, num_classes]
```

### Video Models

**Input:**
- Shape: `[batch_size, frames, 3, H, W]`
- Data type: `uint8`
- Value range: `[0, 255]`
- Color format: BGR (as from cv2.VideoCapture)

**Output:**
- Shape: `[batch_size, num_classes]`
- Type: Logits

Your model should aggregate temporal information internally:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, 3, H, W] uint8 [0, 255]
    batch_size, num_frames = x.shape[:2]
    x = x.float() / 255.0
    # ... process frames, aggregate temporally ...
    return logits  # [B, num_classes]
```

### Audio Models

**Input:**
- Shape: `[batch_size, 96000]`
- Data type: `float32`
- Value range: `[-1, 1]`
- Sample rate: 16 kHz
- Duration: 6.0 seconds (16000 * 6 = 96000 samples)
- Channels: Mono

**Output:**
- Shape: `[batch_size, num_classes]`
- Type: Logits

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, 96000] float32 [-1, 1]
    # ... your model logic ...
    return logits  # [B, num_classes]
```

---

## 5. Complete Example

### Image Model Example

**model_config.yaml:**
```yaml
name: "simple-image-detector"
version: "1.0.0"
modality: "image"

preprocessing:
  resize: [224, 224]

model:
  num_classes: 2
  weights_file: "model.safetensors"
```

**model.py:**
```python
import torch
import torch.nn as nn
from safetensors.torch import load_file


class SimpleImageDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    model = SimpleImageDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
```

---

## 6. Testing Your Model

Use gasbench to test your model locally before submission:

```bash
# Test image model
gasbench run --image-model ./my_image_model/ --debug

# Test video model
gasbench run --video-model ./my_video_model/ --debug

# Test audio model
gasbench run --audio-model ./my_audio_model/ --debug
```

The `--debug` flag runs a quick test with limited samples.

---

## 7. Creating the Weights File

Use `safetensors.torch.save_file()` to create your weights file:

```python
from safetensors.torch import save_file

model = YourModel()
# ... train your model ...

# Save weights
save_file(model.state_dict(), "model.safetensors")
```

---

## 8. Packaging for Submission

Create a ZIP archive of your model directory:

```bash
cd my_model/
zip -r ../my_model.zip model_config.yaml model.py model.safetensors
```

Then upload using the discriminator push command:

```bash
# Image model
gascli d push --image-model my_model.zip

# Video model  
gascli d push --video-model my_model.zip

# Audio model
gascli d push --audio-model my_model.zip
```

---

## Common Issues

1. **Missing load_model function**: Ensure `model.py` has a `load_model(weights_path, num_classes)` function.

2. **Wrong input dtype**: Models receive `uint8` for image/video, `float32` for audio. Handle conversion in your forward pass.

3. **Wrong output shape**: Output must be `[batch_size, num_classes]` logits.

4. **Mismatched resize dimensions**: Ensure `preprocessing.resize` in config matches your model's expected input size.

5. **ONNX format**: ONNX is no longer accepted. Convert your model to safetensors format.
