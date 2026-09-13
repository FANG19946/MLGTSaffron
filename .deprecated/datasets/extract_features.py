import io
import os
import tarfile

import numpy as np
import torch
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights


# -------------------------
# Configuration
# -------------------------

IMAGE_TAR = "/mnt/Drive1/users/adnan/imagenet_work/ILSVRC2012_img_train.tar"
OUTPUT_PATH = "/mnt/Drive1/users/adnan/imagenet_work/X.npy"
PROGRESS_PATH = "/mnt/Drive1/users/adnan/imagenet_work/extraction_progress.txt"

DEVICE = torch.device("cuda:1")
BATCH_SIZE = 32

NUM_IMAGES = 1281167
FEATURE_DIM = 4096


# -------------------------
# Load VGG19
# -------------------------

print("Loading VGG19...", flush=True)

weights = VGG19_Weights.IMAGENET1K_V1

model = vgg19(weights=weights)
model = model.to(DEVICE)
model.eval()

# Official ImageNet preprocessing
preprocess = weights.transforms()

# VGG19:
# classifier[0] = FC1
# classifier[3] = FC2
# classifier[6] = FC3
#
# classifier[:5]:
# FC1 -> ReLU -> Dropout -> FC2 -> ReLU
#
# Therefore this outputs the 4096-D FC2 feature after ReLU.

feature_extractor = torch.nn.Sequential(
    *list(model.classifier.children())[:5]
)

feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# Convolutional part of VGG19
features = model.features


# -------------------------
# Create / open X.npy
# -------------------------

if os.path.exists(OUTPUT_PATH):

    print("Existing X.npy found. Opening it...", flush=True)

    X = np.lib.format.open_memmap(
        OUTPUT_PATH,
        mode="r+",
    )

    if X.shape != (NUM_IMAGES, FEATURE_DIM):
        raise RuntimeError(
            f"Existing X.npy has wrong shape: {X.shape}. "
            f"Expected: {(NUM_IMAGES, FEATURE_DIM)}"
        )

    if X.dtype != np.float32:
        raise RuntimeError(
            f"Existing X.npy has wrong dtype: {X.dtype}. "
            f"Expected: float32"
        )

else:

    print("Creating X.npy...", flush=True)

    X = np.lib.format.open_memmap(
        OUTPUT_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(NUM_IMAGES, FEATURE_DIM),
    )


print(f"X.npy shape: {X.shape}", flush=True)
print(f"X.npy dtype: {X.dtype}", flush=True)


# -------------------------
# Open ImageNet archive
# -------------------------

print("Opening ImageNet training archive...", flush=True)

with tarfile.open(IMAGE_TAR, mode="r") as outer_tar:

    # Find all 1000 ImageNet class tar files
    class_members = [
        member
        for member in outer_tar
        if member.name.endswith(".tar")
    ]

    print(
        f"Found {len(class_members)} classes",
        flush=True,
    )

    if len(class_members) != 1000:
        raise RuntimeError(
            f"Expected 1000 class archives, "
            f"but found {len(class_members)}"
        )


    # -------------------------
    # Determine where to resume
    # -------------------------

    start_class = 0
    image_index = 0

    if os.path.exists(PROGRESS_PATH):

        with open(PROGRESS_PATH, "r") as f:
            progress = f.read().strip()

        if progress:

            parts = progress.split()

            start_class = int(parts[0])
            image_index = int(parts[1])

            print(
                f"Resuming from class "
                f"{start_class + 1}/1000",
                flush=True,
            )

            print(
                f"Images already written: "
                f"{image_index}/{NUM_IMAGES}",
                flush=True,
            )

    else:

        print(
            "No previous progress found. "
            "Starting from class 1.",
            flush=True,
        )


    # -------------------------
    # Process classes
    # -------------------------

    for class_index in range(
        start_class,
        len(class_members),
    ):

        class_member = class_members[class_index]

        class_number = class_index + 1

        class_name = os.path.splitext(
            os.path.basename(class_member.name)
        )[0]

        print(
            f"\n[{class_number}/1000] "
            f"Processing class: {class_name}",
            flush=True,
        )

        class_file = outer_tar.extractfile(
            class_member
        )

        if class_file is None:
            raise RuntimeError(
                f"Could not extract class archive: "
                f"{class_name}"
            )


        # -------------------------
        # Stream nested class TAR
        # -------------------------

        with tarfile.open(
            fileobj=class_file,
            mode="r|",
        ) as class_tar:

            batch_images = []
            class_image_count = 0

            for image_member in class_tar:

                if not image_member.name.endswith(".JPEG"):
                    continue

                image_file = class_tar.extractfile(
                    image_member
                )

                if image_file is None:
                    continue

                # Read JPEG bytes into RAM
                image_bytes = image_file.read()

                # Decode JPEG
                image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                # Apply VGG19 ImageNet preprocessing
                batch_images.append(
                    preprocess(image)
                )


                # -------------------------
                # Process full batch
                # -------------------------

                if len(batch_images) == BATCH_SIZE:

                    batch = torch.stack(
                        batch_images
                    ).to(DEVICE)

                    with torch.no_grad():

                        x = features(batch)

                        x = torch.flatten(
                            x,
                            1,
                        )

                        x = feature_extractor(x)


                    # Move features to CPU
                    batch_features = (
                        x.cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    batch_size = len(
                        batch_features
                    )


                    # -------------------------
                    # Write directly to X.npy
                    # -------------------------

                    X[
                        image_index:
                        image_index + batch_size
                    ] = batch_features

                    image_index += batch_size
                    class_image_count += batch_size


                    print(
                        f"  Processed "
                        f"{class_image_count} images "
                        f"(total: "
                        f"{image_index}/{NUM_IMAGES})",
                        flush=True,
                    )


                    # -------------------------
                    # Release memory
                    # -------------------------

                    del batch
                    del x
                    del batch_features

                    batch_images = []


            # -------------------------
            # Process final incomplete batch
            # -------------------------

            if batch_images:

                batch = torch.stack(
                    batch_images
                ).to(DEVICE)

                with torch.no_grad():

                    x = features(batch)

                    x = torch.flatten(
                        x,
                        1,
                    )

                    x = feature_extractor(x)


                batch_features = (
                    x.cpu()
                    .numpy()
                    .astype(np.float32)
                )

                batch_size = len(
                    batch_features
                )


                X[
                    image_index:
                    image_index + batch_size
                ] = batch_features

                image_index += batch_size
                class_image_count += batch_size


                print(
                    f"  Processed "
                    f"{class_image_count} images "
                    f"(total: "
                    f"{image_index}/{NUM_IMAGES})",
                    flush=True,
                )


                del batch
                del x
                del batch_features

                batch_images = []


        # -------------------------
        # Class completed
        # -------------------------

        X.flush()


        # Save progress:
        #
        # First number  = number of completed classes
        # Second number = number of images written
        #
        # Example:
        # 1 1300
        #
        # means class 1 is complete and
        # 1300 images have been written.

        completed_classes = class_number

        with open(
            PROGRESS_PATH,
            "w",
        ) as f:

            f.write(
                f"{completed_classes} "
                f"{image_index}"
            )


        print(
            f"Finished class {class_name}. "
            f"Class images: {class_image_count}. "
            f"Total written: "
            f"{image_index}/{NUM_IMAGES}",
            flush=True,
        )


# -------------------------
# Final verification
# -------------------------

X.flush()

print()
print("========================================")
print("Finished!")
print("========================================")
print(
    f"Images written : {image_index}"
)
print(
    f"Expected       : {NUM_IMAGES}"
)
print(
    f"Output         : {OUTPUT_PATH}"
)
print(
    f"Shape          : {X.shape}"
)
print(
    f"Dtype          : {X.dtype}"
)
print("========================================")