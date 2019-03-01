# import sys
import os
import underworld as uw
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
import matplotlib.pyplot as plt
import h5py
import json
import numpy as np

# import datetime
# import pickle
# import glucifer


outputDirName = "/run/user/1000/gvfs/sftp:host=lisa.surfsara.nl,user=alaik/home/alaik/uw/4x12_8-00175_DrhoLM50/"
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
refViscosity = 5e20 * u.pascal * u.second
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

time = np.genfromtxt(outputDir + "/tcheckpoint.log", delimiter=",")

# time[-1]
trC = []
spC = []
opBaC = []
opFaC = []
for i in time[:, 0]:
    stStr = str(int(i)).zfill(5)
    with h5py.File(outputDir + "tswarm-" + stStr + ".h5", "r") as f:
        tcord = f["data"][()]
    with h5py.File(outputDir + "tcoords-" + stStr + ".h5", "r") as f:
        ic = f["data"][()]
    isNearTrench = (ic[:, 1] == 0) & ((ic[:, 0] > 1.999) & (ic[:, 0] < 2.001))
    isSubductingPlate = (ic[:, 1] == 0) & (ic[:, 0] < 0.701) & (ic[:, 0] > 0.699)
    isOverRidingPlateBA = (ic[:, 1] == 0) & (ic[:, 0] > 2.499) & (ic[:, 0] < 2.501)
    isOverRidingPlateFA = (ic[:, 1] == 0) & (ic[:, 0] > 2.049) & (ic[:, 0] < 2.051)

    # Mask For The Trench
    sp = np.copy(tcord[isSubductingPlate])
    tr = np.copy(tcord[isNearTrench])
    opBA = np.copy(tcord[isOverRidingPlateBA])
    opFA = np.copy(tcord[isOverRidingPlateFA])
    spC.append(np.average(sp[:, 0]))
    opBaC.append(np.average(opBA[:, 0]))
    opFaC.append(np.average(opFA[:, 0]))
    trC.append(np.average(tr[:, 0]))
# %matplotlib
plt.figure()
plt.plot(ic[:, 0], ic[:, 1])
trC = np.array(trC)
spC = np.array(spC)
opBaC = np.array(opBaC)
opFaC = np.array(opFaC)
# %matplotlib
# plt.plot(vt)

trDx = trC[0:-1] - trC[1:]
spDx = spC[1:] - spC[0:-1]
opBaDx = opBaC[0:-1] - opBaC[1:]
opFaDx = opFaC[0:-1] - opFaC[1:]
dt = time[1:, 1] - time[0:-1, 1]
vt = trDx / dt
vsp = spDx / dt
vb = opBaDx / dt
vf = opFaDx / dt
time.shape
# %matplotlib
plt.style.use("seaborn")
plt.plot(dm(time[1:, 1], u.megayear), dm(vt, u.centimeter / u.year), label="$V_t$")
plt.plot(dm(time[1:, 1], u.megayear), dm(vsp, u.centimeter / u.year), label="$V_{sp}$")
plt.plot(
    dm(time[1:, 1], u.megayear), dm(vsp - vt, u.centimeter / u.year), label="$V_c$"
)

plt.plot(dm(time[1:, 1], u.megayear), dm(vb, u.centimeter / u.year), label="$V_ba$")
plt.plot(dm(time[1:, 1], u.megayear), dm(vf, u.centimeter / u.year), label="$V_fa$")
plt.plot(
    dm(time[1:, 1], u.megayear), dm(vf - vb, u.centimeter / u.year), label="$V_{f-b}$"
)
plt.xlabel("Time in megayear")
plt.ylabel("$Vx in centimeters/year")
plt.legend()
