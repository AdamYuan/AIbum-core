# AIbum-core
## Examples
```python
import pyaibum_core as ab
image = ab.Image('image.png')
classifier = ab.ImageNet('path/to/models/')
tags = classifier.getTags(image)
face_recogizer = ab.MTCNNFaceNet('path/to/models/')
faces = face_recogizer.getFaces(image)
```