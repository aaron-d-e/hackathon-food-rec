import os

import gdown


def download_from_drive(id_or_url, output, md5=None, quiet=False, cache=True):
    if id_or_url.startswith('http') or id_or_url.startswith('https'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    # gdown>=6 removed md5 from download(); cached_download then breaks.
    # Skip re-download when output already exists (same effect when md5 is unused).
    if cache and output and os.path.isfile(output) and os.path.getsize(output) > 0:
        return output

    if output:
        parent = os.path.dirname(os.path.abspath(output))
        if parent:
            os.makedirs(parent, exist_ok=True)

    return gdown.download(url, output, quiet=quiet)

weight_urls = {
    'yolov8s': "1f2kOOyCQ8aHzSHPH8jf9Z6cT4ai-yqmx",
    'yolov5s': "1rISMag8OCM5v99TYuavAobm3LkwjtAi9",
    "yolov5m": "1I649VGqkam_IcCCW8WUA965vPrW_pqDX",
    "yolov5l": "1sBciFcRav2ZE6jzhWnca9uegjQ4860om",
    "yolov5x": "1CRD6T9QtH9XEa-h985_Ho6jgLWu58zn0",
    "effnetb4": "1-K_iDfuhxQFHIF9HTy8SvfnIFwjqxtaX",
    "semantic_seg": "19JRQr9xs2SIeTxX0TQ0k4U9ZnihahvqC"
}


def download_pretrained_weights(name, output=None):
    return download_from_drive(weight_urls[name], output)
