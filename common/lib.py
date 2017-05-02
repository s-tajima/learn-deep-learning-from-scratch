from subprocess import *

import random, math, sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

def plt_show_alt(plt):
    plt.savefig("/tmp/output.png")
    process = Popen(["/usr/local/sbin/imgcat", "/tmp/output.png"])
    plt.clf()
