#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import underworld as uw
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
import glucifer
import numpy as np


# In[2]:


#
# Scaling and Units
#
# Dimentional Parameters
modelHeight = 2891 * u.kilometer
earthRadius = 6371 * u.kilometer
modelCartAspect = 4
ThetaRAD = np.rad2deg((modelHeight * modelCartAspect) / earthRadius)

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
time = 0.0
step = 0
maxtimeSteps = 100

# In[29]:


mesh = uw.mesh.FeMesh_Annulus(
    elementRes=(128, 128),
    radialLengths=(nd(earthRadius - modelHeight), nd(earthRadius)),
    angularExtent=((180 - ThetaRAD.magnitude) / 2, 90 + ThetaRAD.magnitude / 2),
)


velocityField = mesh.add_variable(nodeDofCount=2)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)

swarm = uw.swarm.Swarm(mesh, particleEscape=True)
materialVariable = swarm.add_variable(count=1, dataType="int")
layout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, particlesPerCell=10)
swarm.populate_using_layout(layout)
advector = uw.systems.SwarmAdvector(velocityField=velocityField, swarm=swarm)

swarm_popcontrol = uw.swarm.PopulationControl(
    swarm,
    deleteThreshold=0.0025,
    splitThreshold=0.10,
    maxDeletions=0,
    maxSplits=100,
    aggressive=True,
    aggressiveThreshold=0.95,
    particlesPerCell=20,
)
# In[30]:


lower = mesh.specialSets["Bottom_VertexSet"]
upper = mesh.specialSets["Top_VertexSet"]
left = mesh.specialSets["Left_VertexSet"]
right = mesh.specialSets["Right_VertexSet"]
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
# iWalls+jWalls-lower-upper-left-right


# In[31]:


velocityField.data[:] = [0.0, 0.0]
freeslipBC = uw.conditions.RotatedDirichletCondition(
    variable=velocityField,
    indexSetsPerDof=(iWalls, jWalls),
    basis_vectors=(mesh.bnd_vec_normal, mesh.bnd_vec_tangent),
)


# In[32]:


store = None
fig = glucifer.Figure(store=store, figsize=(1200, 600))
# fig.append( glucifer.objects.Mesh( mesh ,nodeNumbers=True))
fig.append(glucifer.objects.Mesh(mesh))
# fig.append( glucifer.objects.Points( swarm,pointsize=4))
fig.show()


# In[33]:


figV = glucifer.Figure(store=store, figsize=(1200, 600))
figV.Surface(mesh, fn.math.dot(velocityField, velocityField), colours="spectral")
figV.save("V" + str(step).zfill(5))


# In[34]:


radialFn = fn.math.sqrt(fn.math.dot(fn.coord(), fn.coord()))
thetaFn = fn.math.atan2(fn.coord()[1], fn.coord()[0])


# In[ ]:


# In[35]:


# def make_slab(startTheta,length,)


# In[10]:


# for index,coord in enumerate(swarm.data):
#     r,theta=cart2pol(coord[0],coord[1])
#     if r>nd(earthRadius-660.*u.kilometer):
#         materialVariable.data[index]=1
sarc = (modelHeight * 1.5 / earthRadius).magnitude
uppermantle = radialFn.evaluate(swarm.data) > nd(earthRadius - 660.0 * u.kilometer)
upslab = (
    (thetaFn.evaluate(swarm.data) > np.radians(90))
    & (thetaFn.evaluate(swarm.data) < np.radians(90) + sarc)
    & (radialFn.evaluate(swarm.data) > nd(earthRadius - 10.0 * u.kilometer))
)
slab = (
    (thetaFn.evaluate(swarm.data) > np.radians(90))
    & (thetaFn.evaluate(swarm.data) < np.radians(90) + sarc)
    & (radialFn.evaluate(swarm.data) > nd(earthRadius - 80.0 * u.kilometer))
)
perturb = (
    (thetaFn.evaluate(swarm.data) > np.radians(90))
    & (thetaFn.evaluate(swarm.data) < np.radians(91))
    & (radialFn.evaluate(swarm.data) > nd(earthRadius - 170.0 * u.kilometer))
)
materialVariable.data[:] = 1
materialVariable.data[uppermantle] = 0
materialVariable.data[slab | perturb] = 2
materialVariable.data[upslab] = 3


# In[11]:


store = None
fig = glucifer.Figure(store=store, figsize=(1200, 600))
fig.append(glucifer.objects.Mesh(mesh))
fig.append(glucifer.objects.Points(swarm, materialVariable, pointsize=4))

fig.save("M" + str(step).zfill(5))


# In[12]:


# strainRate_2ndInvariant = fn.tensor.second_invariant(
#                             fn.tensor.symmetric(
#                             velocityField.fn_gradient ))
# cohesion = 0.06
# vonMises = 0.5 * cohesion / (strainRate_2ndInvariant+1.0e-18)

# slabYieldvisc = fn.exception.SafeMaths( fn.misc.min(vonMises, slabViscosity) )
viscosityMap = {0: 1.0, 1: 100.0, 2: 1000.0, 3: 1.0}
viscosityFn = fn.branching.map(fn_key=materialVariable, mapping=viscosityMap)


# In[13]:


store = None
figVisc = glucifer.Figure(store=store, figsize=(1200, 600))
figVisc.append(glucifer.objects.Points(swarm, viscosityFn, pointsize=4, logScale=True))

figVisc.save("Eta" + str(step).zfill(5))


# In[14]:


densityMap = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0}
densityFn = fn.branching.map(fn_key=materialVariable, mapping=densityMap)

bodyForceFn = -1.0 * densityFn * mesh.unitvec_r_Fn
# bodyForceProj=


# In[17]:


figDensity = glucifer.Figure(store=store, figsize=(1200, 600))
figDensity.append(glucifer.objects.Points(swarm, viscosityFn, pointsize=4))
figDensity.append(glucifer.objects.VectorArrows(mesh, -1.0 * mesh.unitvec_r_Fn))
figDensity.show()


# In[18]:


stokesSLE = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    conditions=freeslipBC,
    fn_viscosity=viscosityFn,
    fn_bodyforce=bodyForceFn,
    _removeBCs=False,
)  # _removeBC is required


# In[20]:


solver = uw.systems.Solver(stokesSLE)

solver.options.A11.ksp_type = "fgmres"
solver.options.scr.ksp_rtol = 1.0e-4
solver.options.scr.ksp_max_it = 100
solver.options.scr.ksp_monitor = "ascii"
solver.options.scr.ksp_type = "fgmres"
solver.options.A11.ksp_rtol = 1.0e-5
solver.options.A11.ksp_monitor = "ascii"

solver.set_penalty(10.0)
solver.options.main.restore_K = True
solver.options.main.force_correction = True
solver.options.main.Q22_pc_type = "gkgdiag"

solver.options.mg_accel.mg_accelerating_smoothing = True
solver.options.mg_accel.mg_smooths_to_start = 3
solver.options.mg_accel.mg_smooths_max = 10
solver.options.mg.mg_levels_ksp_convergence_test = "skip"
solver.options.mg.mg_levels_ksp_norm_type = "none"
solver.options.mg.mg_levels_ksp_max_it = 5
solver.options.mg.mg_levels_ksp_type = "chebyshev"
solver.options.mg.mg_coarse_pc_type = "lu"
solver.options.mg.mg_coarse_pc_factor_mat_solver_package = "mumps"


def postSolve():
    # realign solution
    uw.libUnderworld.Underworld.AXequalsX(
        stokesSLE._rot._cself, stokesSLE._velocitySol._cself, False
    )
    # remove null space
    uw.libUnderworld.StgFEM.SolutionVector_RemoveVectorSpace(
        stokesSLE._velocitySol._cself, stokesSLE._vnsVec._cself
    )


advector = uw.systems.SwarmAdvector(velocityField=velocityField, swarm=swarm)


# In[24]:


while step < maxtimeSteps:
    solver.solve(print_stats=True, reinitialise=True, callback_post_solve=postSolve)
    uw.libUnderworld.Underworld.AXequalsX(
        stokesSLE._rot._cself, stokesSLE._velocitySol._cself, False
    )
    dt = advector.get_max_dt()
    advector.integrate(dt)
    uw.mpi.barrier()
    swarm_popcontrol.repopulate()
    step = step + 1
    time = time + dt
    if step % 2 == 0:
        figV.save("V" + str(step).zfill(5))
        figVisc.save("Eta" + str(step).zfill(5))


# In[ ]:


# figV.Surface(mesh,fn.math.dot(velocityField,velocityField),colours='spectral')
figV.show()


# In[95]:


figVisc.show()
