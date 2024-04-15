import os
import subprocess
import re


def trim(s):
    s = s[0:-5]
    return int( s.strip() )

"""
Finds a gpu with empty vram and sets that GPU to be utilized
"""
def gpu_selector(select_gpu=-1):

    if select_gpu > -1:
        print("GPU " + str(select_gpu) + " is explicitly selected")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(select_gpu)
        return

    line = str(subprocess.check_output("nvidia-smi | grep Default", shell=True))
    iter = re.finditer(" [0-9]*MiB /", line)

    for ind, i in enumerate(iter):
        s = i.group()
        usage = trim(s)
        if usage < 10:
            print("GPU " + str(ind) + " is selected")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ind)
            return

    print("No free GPU was found, please advise")
    exit()