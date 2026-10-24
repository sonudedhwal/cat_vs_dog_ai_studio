"""Train a Cat vs Dog classifier with transfer learning (MobileNetV2).

Dataset layout:
    data/train/cat/*.jpg      data/train/dog/*.jpg
    data/val/cat/*.jpg        data/val/dog/*.jpg     (optional)

Run:  python train.py --data data --epochs 5
"""
import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers

IMG = (224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="cat_dog_model.keras")
    args = ap.parse_args()

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    if os.path.isdir(val_dir):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=IMG, batch_size=args.batch, label_mode="binary")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, image_size=IMG, batch_size=args.batch, label_mode="binary")
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, validation_split=0.2, subset="training", seed=42,
            image_size=IMG, batch_size=args.batch, label_mode="binary")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, validation_split=0.2, subset="validation", seed=42,
            image_size=IMG, batch_size=args.batch, label_mode="binary")

    print("Classes (alphabetical, dog must be 1):", train_ds.class_names)
    assert train_ds.class_names == ["cat", "dog"], "Folders must be named 'cat' and 'dog'"

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=IMG + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # baked into model
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(dog)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # fine-tune top of the backbone
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=max(2, args.epochs // 2))

    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
