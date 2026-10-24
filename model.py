"""Cat vs Dog classifier.

* If `cat_dog_model.keras` exists (made by train.py) it is used.
* Otherwise a pretrained MobileNetV2 (ImageNet) is used and its 1000 classes
  are grouped into cat (281-285) and dog (151-268) so the app works
  immediately, with no training.
"""
import os
from functools import lru_cache

import numpy as np
from PIL import Image

MODEL_PATH = os.getenv("CAT_DOG_MODEL", "cat_dog_model.keras")
IMG_SIZE = (224, 224)
UNSURE_THRESHOLD = 0.60


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf

    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH), "custom-trained"
    base = tf.keras.applications.MobileNetV2(weights="imagenet")
    return base, "imagenet-pretrained"


def _prep(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype="float32"), 0)  # 0-255 range


def predict(img: Image.Image) -> dict:
    import tensorflow as tf

    model, mode = load_model()
    x = _prep(img)

    if mode == "custom-trained":
        # preprocessing is baked into the model; output = P(dog)
        dog = float(model.predict(x, verbose=0)[0][0])
        cat = 1.0 - dog
    else:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs = model.predict(x, verbose=0)[0]
        cat_raw = float(probs[281:286].sum())
        dog_raw = float(probs[151:269].sum())
        total = cat_raw + dog_raw
        if total < 0.05:  # neither looks like a cat or dog
            return {"label": "Not sure", "confidence": 0.0, "cat_prob": 0.0,
                    "dog_prob": 0.0, "mode": mode}
        cat, dog = cat_raw / total, dog_raw / total

    label = "Dog" if dog >= cat else "Cat"
    confidence = max(cat, dog)
    if confidence < UNSURE_THRESHOLD:
        label = "Not sure"
    return {"label": label, "confidence": confidence, "cat_prob": cat,
            "dog_prob": dog, "mode": mode}
