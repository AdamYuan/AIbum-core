# AIbum-core
[![Linux GCC](https://github.com/AdamYuan/AIbum-core/actions/workflows/linux.yml/badge.svg)](https://github.com/AdamYuan/AIbum-core/actions/workflows/linux.yml)
[![Windows MSVC](https://github.com/AdamYuan/AIbum-core/actions/workflows/windows-msvc.yml/badge.svg)](https://github.com/AdamYuan/AIbum-core/actions/workflows/windows-msvc.yml)
## Examples
```python
import pyaibum_core as ab
image = ab.Image('image.png')
classifier = ab.ImageNet('path/to/models/')
tags = classifier.getTags(image)
face_recogizer = ab.MTCNNFaceNet('path/to/models/')
faces = face_recogizer.getFaces(image)
```