# IMPORTS
import os
import underworld as uw
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
import matplotlib.pyplot as plt
import h5py
import numpy as np

# import sys
# import datetime
# import pickle
# import json
# import glucifer
# from underworld import function as fn


# **SCALING** #
# =========== #

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
scaling_coefficients["[time]"] = Kt.to_base_units()
scaling_coefficients["[mass]"] = KM.to_base_units()
# scaling_coefficients["[temperature]"] = KT.to_base_units()

# **MESH PARAMS** #
# =============== #

vRes = 64
resMult = 8  # 64 being the base vRes
aRatioMesh = 2  # xRes/yRes
aRatioCoor = 4  # Model len ratio
yRes = int(vRes * resMult)
xRes = int(vRes * aRatioMesh * resMult)
refineHoriz = True
refineVert = True
refInt = [0.00125, 0.00125]
refRange = [0.5, -0.25]
sTime = 0
time = sTime
dt = 0.0
CFL = 1.0  # 0.1*refInt[1]*yRes

# **READ/PARSE LOGS** #
# "00",  00_a_indNB  30  30_hiSpEta  50
# =================== #
setDir = "/Users/arijit/Documents/research/lisa/uw/r8_interfaceTest/"
setPostFixes = ["wtR8_Den_VutUc_thDc_eta0_1e3_Lc1e3_nlLM_2dot5e15_sLc-NoSmooth/","wtR8_Den_VutUc_thDc_eta0_1e3_Lc1e3_nlLM_2dot5e15_sLc"]
for setPostFix in setPostFixes:
    outputDirName = setDir + setPostFix
    # outputDirName = "sftp://alaik@lisa.surfsara.nl/home/alaik/uw/4x12_8-00175_DrhoLM00"
    outputDir = os.path.join(os.path.abspath("."), outputDirName + "/")

    time = np.genfromtxt(outputDir + "tcheckpoint.log", delimiter=",")

    time = np.insert(time, 0, [0.0, 0.0], axis=0)
    cPc = np.zeros_like(time[:, 0])

    for index, step in enumerate(time[:, 0]):
        # step = 100
        stStr = str(int(step)).zfill(5)
        # Tracer Coordinated from Tracer Swarm
        with h5py.File(outputDir + "tswarm-" + stStr + ".h5", "r") as f:
            tcord = f["data"][()]
        # with h5py.File(outputDir + "tvel-" + stStr + ".h5", "r") as f:
        #     tvel = f["data"][()]
        # Initial Tracer Coordinated from Tracer Coords Swarm Variable
        with h5py.File(outputDir + "tcoords-" + stStr + ".h5", "r") as f:
            ic = f["data"][()]

        # Masks for Regions

        isCp = ((ic[:, 1] < -0.0103) & (ic[:, 1] > -0.0105)) & (
            (ic[:, 0] > 1.049) & (ic[:, 0] < 1.051)
        )
        cP = np.copy(tcord[isCp])
        print(cP.size)

        cPc[index] = np.average(cP[:, 1])

    opD = "tracerPP/"
    np.save(opD + setPostFix + "time", time)
    np.save(opD + setPostFix + "cdepth", cPc)

plt.style.use("seaborn")
# plt.figure()
# Calcutae Dx,Dt and V #
# trDx = trC[0:-1] - trC[1:]
# spDx = spC[1:] - spC[0:-1]
# opBaDx = opBaC[0:-1] - opBaC[1:]
# opFaDx = opFaC[0:-1] - opFaC[1:]
# dt = time[1:, 1] - time[0:-1, 1]
# vt = trDx / dt
# vsp = spDx / dt
# vb = opBaDx / dt
# vf = opFaDx / dt
# time.shape
# opD = "SET_e_TracerPorcessed/"
# np.save(opD + setPostFix + "trc", trC)
# np.save(opD + setPostFix + "spc", spC)
# np.save(opD + setPostFix + "bac", opBaC)
# np.save(opD + setPostFix + "fac", opFaC)
# np.save(opD + setPostFix + "time", time)
# # %ma tplotlib
plt.style.use("seaborn")
# plt.figure()
# # %matplotlib
# plt.plot(dm(time[1:, 1], u.megayear), dm(vt, u.centimeter / u.year), label="$V_t0$")
# plt.plot(dm(time[:, 1], u.megayear), -dm(vtr, u.centimeter / u.year), label="$V_tE$")
#
# plt.plot(dm(time[:, 1], u.megayear), dm(vs, u.centimeter / u.year), label="$V_{sp}0$")
# plt.plot(
#     dm(time[1:, 1], u.megayear), dm(vsp - vt, u.centimeter / u.year), label="$V_c0$"
# )
# #
# # plt.plot(dm(time[1:, 1], u.megayear), dm(vb, u.centimeter / u.year), label="$V_ba$")
# plt.plot(dm(time[1:, 1], u.megayear), dm(vf, u.centimeter / u.year), label="$V_fa$")
# plt.plot(
#     dm(time[1:, 1], u.megayear), dm(vf - vb, u.centimeter / u.year), label="$V_{f-b}$"
# )
plt.xlabel("Time in megayear")
plt.ylabel("$Vx$ in centimeters/year")
plt.legend()
