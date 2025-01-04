import torchvision.transforms as T


def config_compose(transforms_list):
    ops = []
    for item in transforms_list:
        t_type = item["type"]

        if t_type == "ColorJitter":
            ops.append(
                T.ColorJitter(
                    brightness=item.get("brightness", 0),
                    contrast=item.get("contrast", 0),
                    saturation=item.get("saturation", 0),
                    hue=item.get("hue", 0),
                )
            )
        elif t_type == "RandomEqualize":
            ops.append(T.RandomEqualize(p=item["p"]))
        elif t_type == "AugMix":
            ops.append(T.AugMix())
        elif t_type == "RandomHorizontalFlip":
            ops.append(T.RandomHorizontalFlip(p=item["p"]))
        elif t_type == "RandomVerticalFlip":
            ops.append(T.RandomVerticalFlip(p=item["p"]))
        elif t_type == "GaussianBlur":
            ops.append(T.GaussianBlur(kernel_size=item["kernel_size"]))
        elif t_type == "RandomRotation":
            ops.append(T.RandomRotation(item["degrees"]))
        elif t_type == "Resize":
            ops.append(T.Resize(item["size"]))
        elif t_type == "ToTensor":
            ops.append(T.ToTensor())
        else:
            raise ValueError(f"Неизвестный тип трансформации: {t_type}")

    return T.Compose(ops)
