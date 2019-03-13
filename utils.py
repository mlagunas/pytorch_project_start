def _torch_to_numpy(tensor):
    """
    Moves a pytorch tensor to numpy
    """
    return tensor.detach().cpu().numpy()


def clip_image(img, min=0, max=1):
    """
    returns an image with values between min and max

    """
    img[img < min] = min
    img[img > max] = max
    return img


def norm01(t):
    """
    normalize t to be between 0-1.
    """
    return (t - t.min()) / (t.max() - t.min())
