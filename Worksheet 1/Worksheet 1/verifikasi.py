import shutil
import subprocess
import soundfile as sf
import cv2
from PIL import Image
import skimage
import matplotlib
import moviepy
import numpy as np
import pandas as pd
import notebook
import imageio_ffmpeg as iio_ffmpeg
import sys
import librosa
import scipy

print("=== Verifikasi Library ===")
print("Python      :", sys.version.split()[0])
print("pysoundfile :", sf.__version__)
print("OpenCV      :", cv2.__version__)
print("Pillow      :", Image.__version__)
print("scikit-image:", skimage.__version__)
print("matplotlib  :", matplotlib.__version__)
print("moviepy     :", moviepy.__version__)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("jupyter/notebook:", notebook.__version__)
print("librosa     :", librosa.__version__)
print("scipy       :", scipy.__version__)

print("\n=== Verifikasi ffmpeg (imageio-ffmpeg) ===")
try:
    print("Path :", iio_ffmpeg.get_ffmpeg_exe())
    print("Info :", iio_ffmpeg.get_ffmpeg_version())
except Exception as e:
    print("Error saat cek imageio-ffmpeg:", e)
