import sys
import underworld as uw
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
import matplotlib.pyplot as plt
import h5py
import datetime
import json
import pickle
import numpy as np
import glucifer
import os

os.environ["UW_ENABLE_TIMING"] = "1"


# import colorcet as cc

# import tokyo

# from colorMaps import coldmorning as coldmorning


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


outputDir = "4x12_8-00175_hiSpEta/"

fH = open(outputDir + "/checkpoint.log", "r")

with open(outputDir + "/checkpoint.log", "r") as infile, open(
    outputDir + "/tRcheckpoint.log", "w+"
) as outfile:
    temp = infile.read().replace(";", "")
    outfile.write(temp)
time = np.genfromtxt(outputDir + "/tRcheckpoint.log", delimiter=",")

time[-1]
trC = []
spC = []
for i in time[:, 0]:
    stStr = str(int(i)).zfill(5)
    with h5py.File(outputDir + "tswarm-" + stStr + ".h5", "r") as f:
        tcord = f["data"][()]
    with h5py.File(outputDir + "tcoords-" + stStr + ".h5", "r") as f:
        ic = f["data"][()]
    isNearTrench = (ic[:, 1] == 0) & ((ic[:, 0] > 1.99) & (ic[:, 0] < 2.01))
    isSubductingPlate = (ic[:, 1] == 0) & (ic[:, 0] < 0.701) & (ic[:, 0] > 0.699)
    # Mask For The Trench
    sp = np.copy(tcord[isSubductingPlate])
    tr = np.copy(tcord[isNearTrench])
    spC.append(np.average(sp[:, 0]))
    trC.append(np.average(tr[:, 0]))
# %matplotlib
plt.plot(ic[:, 0], ic[:, 1])
trC = np.array(trC)
spC = np.array(spC)
# %matplotlib
# plt.plot(vt)

trDx = trC[0:-1] - trC[1:]
spDx = spC[1:] - spC[0:-1]
dt = time[1:, 1] - time[0:-1, 1]
vt = trDx / dt
vs = spDx / dt
time.shape

plt.plot(dm(time[1:, 1], u.megayear), dm(vt, u.centimeter / u.year), label="$V_t$")
plt.plot(dm(time[1:, 1], u.megayear), dm(vs, u.centimeter / u.year), label="$V_s$")
plt.legend()
# ! head ./4x12_8-00175_hiSpEta/runLog.log -n 100
mesh = uw.mesh.FeMesh_Cartesian(
    elementType=("Q1/dQ0"),
    elementRes=(xRes, yRes),
    # minCoord=(nd(0.*u.kilometer), nd(-modelHeight+192.*u.kilometer)),
    minCoord=(nd(0.0 * u.kilometer), nd(-modelHeight)),
    # maxCoord=(nd(9600.*u.kilometer), nd(192.*u.kilometer)),
    maxCoord=(aRatioCoor * nd(modelHeight), nd(0.0 * u.kilometer)),
    periodic=[False, False],
)
mesh.load(outputDir + "/mesh.h5")
velocityField = mesh.add_variable(nodeDofCount=mesh.dim)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)
tracerSwarm = uw.swarm.Swarm(mesh=mesh)
tincord = tracerSwarm.add_variable(dataType="double", count=2)

f = glucifer.Figure(figsize=(1300, 444))
f.viewer()


def load_mesh_vars(step):
    if uw.rank() == 0:
        print("Loading Mesh.....")
    # mh =
    mesh.load(outputDir + "/mesh.00000.h5")

    velocityField.load(outputDir + "velocity-" + str(step).zfill(5) + ".h5")
    pressureField.load(outputDir + "pressure-" + str(step).zfill(5) + ".h5")
    if uw.rank() == 0:
        print("Loading Mesh......Done!")
    # return mesh.save(outputDir + "mesh.00000.h5")


# isSubductingPlate = (ic[:, 1] == 0) & (ic[:, 0] < 0.701) & (ic[:, 0] > 0.699)
# tcord[isSubductingPlate]
time
velsp = []
velt = []
for i in np.arange(0, time[-1][0], 50):
    stStr = str(int(i)).zfill(5)
    # tracerSwarm = uw.swarm.Swarm(mesh=mesh)
    # tincord = tracerSwarm.add_variable(dataType="double", count=2)
    # tracerSwarm.load(outputDir + "tswarm-" + stStr + ".h5")
    # tincord.load(outputDir + "tcoords-" + stStr + ".h5")
    # ic = tincord.data
    # tcord = tracerSwarm.data
    with h5py.File(outputDir + "tswarm-" + stStr + ".h5", "r") as f:
        tcord = f["data"][()]
    with h5py.File(outputDir + "tcoords-" + stStr + ".h5", "r") as f:
        ic = f["data"][()]
    velocityField.load(outputDir + "velocity-" + stStr + ".h5")
    # isNearTrench = (ic[:, 1] == 0) & (ic[:, 0] == 0.725)
    # isNearTrench = (ic[:, 1] == 0) & ((ic[:, 0] > 0.99) & (ic[:, 0] < 1.00))
    # tr = tcord[isNearTrench]
    isNearTrench = (ic[:, 1] == 0) & ((ic[:, 0] > 1.99) & (ic[:, 0] < 2.01))
    isSubductingPlate = (ic[:, 1] == 0) & (ic[:, 0] < 0.701) & (ic[:, 0] > 0.699)
    # Mask For The Trench
    sp = np.copy(tcord[isSubductingPlate])
    tr = np.copy(tcord[isNearTrench])
    # print(tr)
    velt.append(velocityField.evaluate(tr))
    velsp.append(velocityField.evaluate(sp))
# vel.shape
# velsp.shape
velt = np.array(velt)
velsp = np.array(velsp)
x = np.arange(0, time[-1][0], 50)
time[-1]
ndT = []
for i in np.arange(0, time[-1][0], 50):
    ndT.append(time[time[:, 0] == i, 1])
ndT = np.array(ndT)
# % matplotlib
plt.plot(
    dm(ndT[1:], u.megayear), dm(-velt[1:, 1, 0], u.centimeter / u.year), label="vTM2"
)
plt.plot(
    dm(ndT[1:], u.megayear), dm(velsp[1:, 2, 0], u.centimeter / u.year), label="vSM-2"
)
plt.plot(dm(time[1:, 1], u.megayear), dm(vt, u.centimeter / u.year), label="$V_t$2")
plt.plot(dm(time[1:, 1], u.megayear), dm(vs, u.centimeter / u.year), label="$V_s$2")
plt.legend()

plt.xlabel("Time in megayear")
plt.ylabel("$Vx_{t}$ and $Vx_{sp}$ in centimeters/year")
# !ls
# plt.plot(dm(np.array(ndT)[1:], u.megayear), dm(vel[:, 0, 0], u.centimeter / u.year))
