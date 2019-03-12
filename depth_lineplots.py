"""
Timeseries plot
================================
"""
import seaborn as sns
import numpy as np
import os
import pandas as pd
import cPickle
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
from underworld import scaling as scaling

# **SCALING** #
# =========== #

modelHeight = 2880.0 * u.kilometer
refDensity = 3200.0 * u.kilogram / u.meter ** 3
deltaRhoMax = 80.0 * u.kilogram / u.meter ** 3
gravity = 9.8 * u.metre / u.second ** 2
refViscosity = 5e20 * u.pascal * u.second
bodyForce = deltaRhoMax * gravity

# scaling coefficients
K_eta = refViscosity
KL = modelHeight
K_tau = bodyForce * modelHeight
K_v = K_tau * modelHeight / K_eta  # or # Kt = KL/K_v
Kt = K_eta / K_tau
KM = K_tau * modelHeight * Kt ** 2

scaling_coefficients = scaling.get_coefficients()

scaling_coefficients["[length]"] = KL.to_base_units()
scaling_coefficients["[time]"] = Kt.to_base_units()
scaling_coefficients["[mass]"] = KM.to_base_units()


dataCols = ["time", "velocity" "Depth" "Exp"]
df = pd.DataFrame(columns=dataCols)
setPostFix = ["00", "00_a_indNB", "30", "30_hiSpEta", "50"]
# setPostFix = ["50", "30", "00", "00_a_indNB", "00_a", "hiSpEta", "LRINDNB"]
opD = "./tracerPP/"
for i in setPostFix:

    #
    time = np.load(opD + i + "time.npy")
    cBaseDepth = np.load(opD + i + "cdepth.npy")
    cBaseDepth = dm(cBaseDepth, u.kilometer)
    t = dm(time[0:, 1], u.megayear)
    trC = np.load(opD + i + "trc.npy")
    trDx = trC[0:-1] - trC[1:]
    dt = time[1:, 1] - time[0:-1, 1]
    trDx = np.insert(trDx, 0, 0)
    dt = np.insert(dt, 0, 1)
    vt = trDx / dt
    vt = dm(vt, u.centimeter / u.year)
    # vsp = np.load(opD + setPrefix[n] + i + "vsp.npy")

    _dft = pd.DataFrame(dict(time=t, velocity=vt, Depth=cBaseDepth, Exp="e0" + i))

    df = df.append(_dft)
# %matplotlib
sns.set()

sns.set_context("poster")
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.05})
ax = sns.lineplot(x="time", y="Depth", hue="Exp", data=df)
ax.set_ylim(-360, -20)
ax.set_xlim(50, 200)
ax.figure.set_size_inches(25, 12)
sns.despine()
ax.set_xlabel("Time (Myr)")
ax.set_ylabel("Depth (Km)")
# ax.figure.set_size_inches(10.5, 10)
ax.yaxis.set_ticks([-30, -105, -136, -165, -336, -360])
ax.xaxis.set_ticks([60, 90, 110, 123, 142, 155, 180, 200])

ax.figure.savefig("tvd.pdf")

ax.grid()

# %matplotlib
# palette="PuBuGn"
# ax.clear()
sns.set()
dfc = df[(df["Exp"] == "e000") | (df["Exp"] == "e050") | (df["Exp"] == "e030")]
dfc = dfc[(dfc["Type"] != "Convergence")]
sns.set_context("poster")
sns.set_style("ticks", {"grid.linestyle": "--", "grid.linewidth": 0.05})

# palette = sns.color_palette("mako_r", 6)
# %matplotlib
sns.set_palette("Set2")
ax = sns.lineplot(x="time", y="velocity", hue="Exp", style="Type", data=dfc)
ax.figure.set_size_inches(10.5, 10)
ax.set_xlim(0, 200)
sns.despine()
# ax.set_ylim(-4.25, 12.5)
ax.set_xlabel("Time (Myr)")
ax.set_ylabel("Velocity (cm/yr)")
# ax.set_yticks(np.arange(-1.5, 4, 0.5))
# ax.set_xticks(np.arange(0, 225, 25))
ax.figure.savefig("tvd.pdf")

ax.clear()
dfc = df[(df["Exp"] == "e000") | (df["Exp"] == "e050") | (df["Exp"] == "e030")]
dfc = dfc[(dfc["Type"] == "Convergence")]
sns.set_context("poster")
sns.set_style("ticks", {"grid.linestyle": "--", "grid.linewidth": 0.05})

# palette = sns.color_palette("mako_r", 6)
# %matplotlib
sns.set_palette("Set2")
ax = sns.lineplot(x="time", y="velocity", hue="Exp", style="Type", data=dfc)
ax.figure.set_size_inches(12.5, 10)
ax.set_xlim(0, 200)
sns.despine()
ax.set_xlabel("Time (Myr)")
ax.set_ylabel("Velocity (cm/yr)")
# ax.set_ylim(-4.25, 12.5)

# ax.set_yticks(np.arange(-1.5, 4, 0.5))
# ax.set_xticks(np.arange(0, 225, 25))
ax.figure.savefig("DrhoCC.pdf")
ax.figure.savefig("DrhoCC.svg")


df.to_csv("df.csv")
ax.figure.show()
# ax.errorbar()
ax.clear()
dfc = df[(df["Exp"] == "e030") | (df["Exp"] == "e030_hiSpEta")]
dfc = dfc[(dfc["Type"] != "Convergence")]
sns.set_palette("Set2")
ax = sns.lineplot(x="time", y="velocity", hue="Exp", style="Type", data=dfc)
ax.figure.set_size_inches(12.5, 10)
ax.set_xlim(0, 200)
sns.despine()
ax.set_xlabel("Time (Myr)")
ax.set_ylabel("Velocity (cm/yr)")
ax.figure.savefig("DETA.pdf")
# ax.set_ylim(-1.5, 4)
# ax.set_yticks(np.arange(-1.5, 4.5, 0.5))
# ax.set_xticks(np.arange(0, 225, 25))
# ax.figure.savefig("DINDNBTRSP.svg")
# ax.grid()
ax.clear()
dfc = df[(df["Exp"] == "e000") | (df["Exp"] == "e000_a_indNB")]
dfc = dfc[(dfc["Type"] == "Trench")]
sns.set_palette("Set2")
ax = sns.lineplot(x="time", y="velocity", hue="Exp", style="Type", data=dfc)
ax.figure.set_size_inches(10.5, 10)
ax.set_xlim(0, 150)
ax.set_ylim(-1.5, 4)
ax.set_yticks(np.arange(-1.5, 4, 0.5))
ax.set_xticks(np.arange(0, 225, 25))
ax.figure.savefig("DINDNBTRSP.svg")
ax.legend()
