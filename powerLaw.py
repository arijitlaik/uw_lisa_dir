import numpy as np
import matplotlib.pyplot as plt


import underworld.scaling as scaling
from underworld.scaling import units as u


def powerLaw(eII, eIIref, n):
    return np.power((eII/eIIref), (1.-n)/n)


def powerLawA(eII, A, n):
    return np.power(A, (-1.0/n))*np.power(eII, (1.-n)/n)


dm = scaling.dimensionalise
nd = scaling.non_dimensionalise

modelHeight = 2880.0 * u.kilometer
refDensity = 3200.0 * u.kilogram / u.meter ** 3
deltaRhoMax = 80.0 * u.kilogram / u.meter ** 3
gravity = 9.8 * u.metre / u.second ** 2
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

scaling_coefficients = scaling.get_coefficients()

scaling_coefficients["[length]"] = KL.to_base_units()
# scaling_coefficients["[temperature]"] = KT.to_base_units()
scaling_coefficients["[time]"] = Kt.to_base_units()
scaling_coefficients["[mass]"] = KM.to_base_units()

n = 3.5
eIIs = np.linspace(nd(1e-18*1./u.second), nd(1e-12*1./u.second))


refEII = nd(1e-15*1./u.second)
etas = powerLaw(eIIs, refEII, n)
etas2 = powerLaw(eIIs, 1, n)

plt.loglog(dm(etas, u.pascal * u.second), dm(eIIs, 1./u.second))
plt.loglog(dm(etas2, u.pascal * u.second), dm(eIIs, 1./u.second))
plt.loglog(dm(powerLaw(eIIs+nd(1e-18*1./u.second), refEII, n),u.pascal * u.second), dm(eIIs, 1./u.second))
#nd(1e-18*1./u.second)
plt.loglog(dm(powerLaw(eIIs, nd(1e-13*1./u.second), n),u.pascal * u.second), dm(eIIs, 1./u.second))
