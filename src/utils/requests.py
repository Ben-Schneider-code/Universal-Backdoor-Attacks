import hashlib
import os

import requests
from tqdm import tqdm

from src.global_configs import system_configs

def url_to_local_file(url: str) -> str:
    """ Downloads a file and saves it at the filepath """
    filepath = os.path.join(system_configs.CACHE_DIR, hashlib.md5(url.encode('utf-8')).hexdigest())
    if not os.path.exists(filepath):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get(url, stream=True, headers=headers)
        with open(filepath, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024), desc="Downloading File .."):
                if chunk:
                    f.write(chunk)
    return filepath


