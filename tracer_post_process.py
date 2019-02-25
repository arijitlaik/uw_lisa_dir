import os
import sys

os.environ["UW_ENABLE_TIMING"] = "1"
import underworld as uw
from underworld import function as fn

from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd

import glucifer

# import colorcet as cc

# import tokyo
import numpy as np

# from colorMaps import coldmorning as coldmorning

import pickle
import json
import datetime


class QuanityEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, u.Quantity):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


#
# Logging, output path and restarting stuff
#


# outputDirName = "dev_py3_TEST_opTe_2x12_512x256"
outputDirName = "4x12_8-00175_hiSpEta"
outputDir = os.path.join(os.path.abspath("."), outputDirName + "/")
if uw.rank() == 0:
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
uw.barrier()
modelHeight = 2880.0 * u.kilometer
# plateHeight = 120. * u.kilometer
refDensity = 3200.0 * u.kilogram / u.meter ** 3
deltaRhoMax = 80.0 * u.kilogram / u.meter ** 3
gravity = 9.8 * u.metre / u.second ** 2
# 1.57e20 * u.pascal * u.second 5.e20 * u.pascal * u.second
refViscosity = 5.0e20 * u.pascal * u.second
bodyForce = deltaRhoMax * gravity

# scaling coefficients
K_eta = refViscosity
KL = modelHeight
K_tau = bodyForce * modelHeight
K_v = K_tau * modelHeight / K_eta
# Kt = KL/K_v
Kt = K_eta / K_tau
KM = K_tau * modelHeight * Kt ** 2

scaling_coefficients = uw.scaling.get_coefficients()

scaling_coefficients["[length]"] = KL.to_base_units()
# scaling_coefficients["[temperature]"] = KT.to_base_units()
scaling_coefficients["[time]"] = Kt.to_base_units()
scaling_coefficients["[mass]"] = KM.to_base_units()

print(json.dumps(scaling_coefficients, cls=QuanityEncoder, indent=2))

vRes = 64
resMult = 8  # 64 being the base vRes
aRatioMesh = 2  # xRes/yRes
aRatioCoor = 4  # Model len ratio
yRes = int(vRes * resMult)
xRes = int(vRes * aRatioMesh * resMult)
refineHoriz = True
refineVert = True
# refineHoriz = False
# refineVert = False
refInt = [0.00125, 0.00125]
refRange = [0.5, -0.25]
sTime = 0
time = nd(sTime * u.megayear)
dt = 0.0
# CFL = 0.1*refInt[1]*yRes
CFL = 1.0


import h5py

import numpy as np

outputDir = "4x12_8-00175_hiSpEta/"
with h5py.File(outputDir + "tswarm-00000.h5", "r") as f:
    t0 = f["data"][()]
with h5py.File(outputDir + "tswarm-00025.h5", "r") as f:
    t1 = f["data"][()]
with h5py.File(outputDir + "tcoords-00000.h5", "r") as f:
    ic0 = f["data"][()]
with h5py.File(outputDir + "tcoords-00025.h5", "r") as f:
    ic1 = f["data"][()]


isNearTrench0 = (ic0[:, 1] == 0) & ((ic0[:, 0] > 1.999) & (ic0[:, 0] < 2.001))
trench0 = t0[isNearTrench0]
isNearTrench1 = (ic1[:, 1] == 0) & ((ic1[:, 0] > 1.999) & (ic1[:, 0] < 2.001))
trench1 = t1[isNearTrench1]
import matplotlib.pyplot as plt

trench0
trench1
ic0[isNearTrench0]
ic1[isNearTrench1]
# %matplotlib

plt.scatter(trench0[:, 0], trench0[:, 1], c=ic0[isNearTrench0][:, 1])
plt.scatter(trench1[:, 0], trench1[:, 1], c=ic1[isNearTrench1][:, 1])
# plt.scatter(ic0['data'][:, 0], ic0['data'][:,1], s=0.1)
# plt.scatter(ic1['data'][:, 0], ic1['data'][:,1], s=0.1)
# PendingDeprecationWarning
