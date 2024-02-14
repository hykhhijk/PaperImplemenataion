import albumentations as A

#basic crop size is 224
def get_transform(crop_size):
    train_transform = [
        A.RandomResizedCrop(crop_size, crop_size),
        A.HorizontalFlip(),
        # A.Normalize(),
    ]
    val_transform = [
        A.RandomResizedCrop(crop_size, crop_size),
    ]

    return A.Compose(train_transform), A.Compose(val_transform)