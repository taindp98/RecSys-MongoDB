from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_transform(config, is_train=False):
    if is_train:
        trf_aug = transforms.Compose(
            [
                transforms.Resize(
                    (int(config.DATA.IMG_SIZE), int(config.DATA.IMG_SIZE))
                ),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.6),
                transforms.RandomRotation(60),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        trf_aug = transforms.Compose(
            [
                transforms.Resize(
                    (int(config.DATA.IMG_SIZE), int(config.DATA.IMG_SIZE))
                ),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return trf_aug
