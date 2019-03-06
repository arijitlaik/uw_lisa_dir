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


dataCols = ["time", "velocity", "Type", "Exp"]
df = pd.DataFrame(columns=dataCols)
setPostFix = ["50", "30", "00", "00_a_indNB", "00_a", "hiSpEta"]
setPrefix = ["", "", "", "4x12_8-00175_DrhoLM", "4x12_8-00175_DrhoLM", "4x12_8-00175_"]
opD = "./SET_e_TracerPorcessed/"
for n, i in enumerate(setPostFix):

    #
    time = np.load(opD + setPrefix[n] + i + "time.npy")
    trC = np.load(opD + setPrefix[n] + i + "trc.npy")
    spC = np.load(opD + setPrefix[n] + i + "spc.npy")

    trDx = trC[0:-1] - trC[1:]
    spDx = spC[1:] - spC[0:-1]
    # opBaDx = opBaC[0:-1] - opBaC[1:]
    dt = time[1:, 1] - time[0:-1, 1]
    vt = trDx / dt
    vsp = spDx / dt
    t = time[1:, 1]
    # t = dm(time[1:, 1], u.megayear)
    # vt = np.load(opD + setPrefix[n] + i + "vt.npy")
    # vsp = np.load(opD + setPrefix[n] + i + "vsp.npy")
    # vt = dm(vt, u.centimeter / u.year)
    # vsp = dm(vsp, u.centimeter / u.year)
    vc = vsp - vt
    _dft = pd.DataFrame(dict(time=t, velocity=vt, Type="Trench", Exp="e0" + i))
    _dfs = pd.DataFrame(
        dict(time=t, velocity=vsp, Type="SubductingPlate", Exp="e0" + i)
    )
    _dfc = pd.DataFrame(dict(time=t, velocity=vc, Type="Convergence", Exp="e0" + i))
    df = df.append(_dft)
    df = df.append(_dfs)
    df = df.append(_dfc)
# %matplotlibpalette="PuBuGn"
sns.set_style("ticks")
dfc = df[df["Type"] == "Convergence"]
sns.set_context("paper")
palette = sns.color_palette("mako_r", 6)
# %matplotlib

ax = sns.lineplot(x="time", y="velocity", hue="Type", data=df)

# , style="Type", hue="Exp", data=df, palette=palette

sns.despine(trim=True)
ax.set_xticks(np.arange(0, 225, 25))
df.to_csv("df.csv")

# ax.set_yticks(np.arange(-1.5, 4, 0.5))
ax.grid()
ax.set_xlim(0, 200)
ax.legend()
