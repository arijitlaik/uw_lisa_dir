#!/usr/bin/env python
# coding: utf-8


import os
import sys

import underworld as uw
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise as dm, non_dimensionalise as nd
import glucifer
import numpy as np
import collections


#
# Scaling and Units
#
# Dimentional Parameters
wholeMantleFlag = False
airLayer = True
airHeight = 50.0 * u.kilometer if airLayer else 0.0 * u.kilometer
modelHeight = 2891 * u.kilometer if wholeMantleFlag else 1000.0 * u.kilometer
earthRadius = 6371 * u.kilometer
modelCartAspect = 2 if wholeMantleFlag else 4
ThetaRAD = np.rad2deg((modelHeight * modelCartAspect) / earthRadius)

# plateHeight = 120. * u.kilometer
refDensity = 3200.0 * u.kilogram / u.meter ** 3
deltaRhoMax = 100.0 * u.kilogram / u.meter ** 3
gravity = 9.8 * u.metre / u.second ** 2
# 1.57e20 * u.pascal * u.second 5.e20 * u.pascal * u.second
refViscosity = 5.0e20 * u.pascal * u.second
bodyForce = refDensity * gravity

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


step = 0
time = 0.0
maxSteps = 1
outputDirName = "AnnSubBenchmark_SA_Hi"
outputDir = os.path.join(os.path.abspath("."), outputDirName + "/")
if uw.mpi.rank == 0:
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
try:
    tflag = os.environ["UW_ENABLE_TIMING"]
except KeyError:
    tflag = False
else:
    if tflag == "1":
        tflag = True
    else:
        tflag = False
uw.mpi.barrier()
if tflag:
    uw.timing.start()
mesh = uw.mesh.FeMesh_Annulus(
    elementRes=(256, 512),
    radialLengths=(nd(earthRadius - modelHeight + airHeight), nd(earthRadius)),
    angularExtent=((180 - ThetaRAD.magnitude) / 2, 90 + ThetaRAD.magnitude / 2),
    periodic=[False, False],
)


velocityField = mesh.add_variable(nodeDofCount=mesh.dim)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)
lower = mesh.specialSets["MinI_VertexSet"]
upper = mesh.specialSets["MaxI_VertexSet"]
swarm = uw.swarm.Swarm(mesh, particleEscape=True)
materialVariable = swarm.add_variable(count=1, dataType="int")
layout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, particlesPerCell=20)
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


store = glucifer.Store(outputDir + "/gfxstore")
if store is not None:
    store.step = 0

fig = glucifer.Figure(store=store, figsize=(1200, 450))
# fig.append( glucifer.objects.Mesh( mesh ,nodeNumbers=True))
fig.append(glucifer.objects.Mesh(mesh))
# fig.append( glucifer.objects.Points( swarm,pointsize=4))
fig.save(outputDir + "/mesh")


radialFn = fn.math.sqrt(fn.math.dot(fn.coord(), fn.coord()))
thetaFn = fn.math.atan2(fn.coord()[1], fn.coord()[0])


# thetaFn.evaluate()


air = mesh.unit_heightFn.evaluate(swarm.data) > nd(modelHeight - airHeight)
lowermantle = mesh.unit_heightFn.evaluate(swarm.data) < nd(
    modelHeight - airHeight - 660.0 * u.kilometer
)
slab = (
    (mesh.unit_heightFn.evaluate(swarm.data) < nd(modelHeight - airHeight))
    & (
        mesh.unit_heightFn.evaluate(swarm.data)
        > nd(modelHeight - airHeight - 100.0 * u.kilometer)
    )
    & (mesh.thetaFn.evaluate(swarm.data) <= np.pi / 2)
)
perturb = (
    (mesh.unit_heightFn.evaluate(swarm.data) < nd(modelHeight - airHeight))
    & (
        mesh.unit_heightFn.evaluate(swarm.data)
        > nd(modelHeight - airHeight - 200.0 * u.kilometer)
    )
    & (mesh.thetaFn.evaluate(swarm.data) <= np.pi / 2)
    & (
        mesh.thetaFn.evaluate(swarm.data)
        >= np.pi / 2 - (100.0 * u.kilometer / earthRadius).magnitude
    )
)
materialVariable.data[:] = 1
materialVariable.data[air] = 0
materialVariable.data[lowermantle] = 2
materialVariable.data[slab | perturb] = 3

figP = glucifer.Figure(store=store, figsize=(1200, 450))
# fig.append( glucifer.objects.Mesh( mesh))
figP.append(
    glucifer.objects.Points(
        swarm,
        materialVariable,
        colours="#00CED1 #F0E68C #FFA500 #2F4F4F",
        pointsize=2,
        discrete=True,
    )
)
figP.save(outputDir + "/Par")


def checkpoint(
    mesh,
    fieldDict,
    swarm,
    swarmDict,
    index,
    modeltime=None,
    meshName="mesh",
    swarmName="swarm",
    prefix="./",
    enable_xdmf=True,
):
    # Check the prefix is valid
    if prefix is not None:
        if not prefix.endswith("/"):
            prefix += "/"  # add a backslash
        if not os.path.exists(prefix) and uw.mpi.rank == 0:
            print("Creating directory: ", prefix)
            os.makedirs(prefix)

    if not isinstance(index, int):
        raise TypeError("'index' is not of type int")
    if modeltime is not None:
        time = modeltime
    else:
        time = index

    ii = str(index).zfill(5)

    if mesh is not None:

        # Error check the mesh and fields
        if not isinstance(mesh, uw.mesh.FeMesh):
            raise TypeError("'mesh' is not of type uw.mesh.FeMesh")
        if not isinstance(fieldDict, dict):
            raise TypeError("'fieldDict' is not of type dict")
        for key, value in fieldDict.items():
            if not isinstance(value, uw.mesh.MeshVariable):
                raise TypeError(
                    "'fieldDict' must contain uw.mesh.MeshVariable elements"
                )

        # see if we have already saved the mesh. It only needs to be saved once
        if not hasattr(checkpoint, "mH"):
            checkpoint.mH = mesh.save(prefix + meshName + ".h5")
        mh = checkpoint.mH

        for key, value in fieldDict.items():
            filename = prefix + key + "-" + ii
            handle = value.save(filename + ".h5")
            if enable_xdmf:
                value.xdmf(filename, handle, key, mh, meshName, modeltime=time)

    # is there a swarm
    if swarm is not None:

        # Error check the swarms
        if not isinstance(swarm, uw.swarm.Swarm):
            raise TypeError("'swarm' is not of type uw.swarm.Swarm")
        if not isinstance(swarmDict, dict):
            raise TypeError("'swarmDict' is not of type dict")
        for key, value in swarmDict.items():
            if not isinstance(value, uw.swarm.SwarmVariable):
                raise TypeError(
                    "'fieldDict' must contain uw.swarm.SwarmVariable elements"
                )
        sH = swarm.save(prefix + swarmName + "-" + ii + ".h5")
        for key, value in swarmDict.items():
            filename = prefix + key + "-" + ii
            handle = value.save(filename + ".h5")
            if enable_xdmf:
                value.xdmf(filename, handle, key, sH, swarmName, modeltime=time)


# I = mesh.specialSets["inner"]
# O = mesh.specialSets["outer"]
# W = mesh.specialSets["Left_VertexSet"]
# E = mesh.specialSets["Right_VertexSet"]
# S = mesh.specialSets["MinJ_VertexSet"]
# N = mesh.specialSets["MaxJ_VertexSet"]

# allWalls = mesh.specialSets["AllWalls_VertexSet"]
# NS0 = N+S-(E+W)
# # build corner edges node indexset
# cEdge = (N&W)+(N&E)+(S&E)+(S&W)

upperSurf = mesh.specialSets["upper_surface_VertexSet"]
lowerSurf = mesh.specialSets["lower_surface_VertexSet"]
rightSide = mesh.specialSets["MinJ_VertexSet"]
leftSide = mesh.specialSets["MaxJ_VertexSet"]
cEdge = (
    (upper & rightSide)
    + (upper & leftSide)
    + (lowerSurf & rightSide)
    + (lowerSurf & leftSide)
)


def setVbc():
    velocityField.data[...] = 0.0
    velocityField.data[upperSurf.data, :] = (0.0, 0.0)
    velocityField.data[lowerSurf.data, :] = (0.0, 0.0)
    velocityField.data[leftSide.data, :] = (0.0, 0.0)
    velocityField.data[rightSide.data, :] = (0.0, 0.0)


setVbc()


freeSlipAn = uw.conditions.RotatedDirichletCondition(
    variable=velocityField,
    indexSetsPerDof=(upperSurf + lowerSurf + cEdge, rightSide + leftSide + cEdge),
    basis_vectors=(mesh.bnd_vec_normal, mesh.bnd_vec_tangent),
)

figVdot = glucifer.Figure(store=store, figsize=(1200, 450))
# fig.append( glucifer.objects.Mesh( mesh))
figVdot.append(
    glucifer.objects.Surface(
        mesh,
        fn.math.dot(velocityField, velocityField),
        # logScale=True,
        colours=glucifer.lavavu.matplotlib_colourmap("viridis"),
    )
)
figVdot.save(outputDir + "/Vdot" + str(step).zfill(5))

airDensity = 0.01
mantleDensity = 0.0
slabDensity = 1.0
densityMap = {0: airDensity, 1: mantleDensity, 2: mantleDensity, 3: slabDensity}
densityFn = fn.branching.map(fn_key=materialVariable, mapping=densityMap)
buoyancyFn = -1.0 * densityFn * mesh.unitvec_r_Fn


figD = glucifer.Figure(store=store, figsize=(1200, 450))
# fig.append( glucifer.objects.Mesh( mesh))
figD.append(glucifer.objects.Points(swarm, densityFn, colours="spectral", pointsize=2))
figD.save(outputDir + "/Den")


airViscosity = 1.0
mantleViscosity = 1.0
lowermantleViscosity = 1e2
slabViscosity = 1e2
viscosityMap = {
    0: airViscosity,
    1: mantleViscosity,
    2: lowermantleViscosity,
    3: slabViscosity,
}
swarmviscosityFn = fn.branching.map(fn_key=materialVariable, mapping=viscosityMap)

viscosityFn = mesh.add_variable(1)
projVisMesh = uw.utils.MeshVariable_Projection(viscosityFn, swarmviscosityFn, type=0)
projVisMesh.solve()

figV = glucifer.Figure(store=store, figsize=(1200, 450))
# fig.append( glucifer.objects.Mesh( mesh))
figV.append(
    glucifer.objects.Surface(
        mesh,
        viscosityFn,
        logScale=True,
        colours=glucifer.lavavu.matplotlib_colourmap("inferno_r"),
    )
)
figV.save(outputDir + "/eta" + str(step).zfill(5))


fieldDict = collections.OrderedDict()
fieldDict["velocity"] = velocityField
fieldDict["pressure"] = pressureField
fieldDict["meshViscosity"] = viscosityFn

# swarmDict = {"materials": materialVariable}
swarmDict = collections.OrderedDict()
swarmDict["materials"] = materialVariable
# traceDict = {"tcoords": tincord, "tvel": tracerVelocity}
uw.mpi.barrier()

checkpoint(
    mesh,
    fieldDict,
    swarm,
    swarmDict,
    index=step,
    modeltime=dm(time, 1.0 * u.megayear).magnitude,
    prefix=outputDir,
)


stokesSLE = uw.systems.Stokes(
    velocityField,
    pressureField,
    fn_viscosity=viscosityFn,
    fn_bodyforce=buoyancyFn,
    conditions=freeSlipAn,
    _removeBCs=False,
)


stokesSolverAN = uw.systems.Solver(stokesSLE)

# stokesSolverAN.set_inner_method = "mumps"
stokesSolverAN.options.A11.ksp_type = "fgmres"
stokesSolverAN.options.scr.ksp_rtol = 1.0e-4
stokesSolverAN.options.scr.ksp_max_it = 100
stokesSolverAN.options.scr.ksp_monitor = "ascii"
stokesSolverAN.options.scr.ksp_type = "fgmres"
stokesSolverAN.options.A11.ksp_rtol = 1.0e-5
stokesSolverAN.options.A11.ksp_monitor = "ascii"

stokesSolverAN.set_penalty(100.0)
stokesSolverAN.options.main.restore_K = True
stokesSolverAN.options.main.force_correction = True
stokesSolverAN.options.main.Q22_pc_type = "gkgdiag"

stokesSolverAN.options.mg_accel.mg_accelerating_smoothing = True
stokesSolverAN.options.mg_accel.mg_smooths_to_start = 3
stokesSolverAN.options.mg_accel.mg_smooths_max = 10
stokesSolverAN.options.mg.mg_levels_ksp_convergence_test = "skip"
stokesSolverAN.options.mg.mg_levels_ksp_norm_type = "none"
stokesSolverAN.options.mg.mg_levels_ksp_max_it = 5
stokesSolverAN.options.mg.mg_levels_ksp_type = "chebyshev"
stokesSolverAN.options.mg.mg_coarse_pc_type = "lu"
stokesSolverAN.options.mg.mg_coarse_pc_factor_mat_solver_package = "mumps"


# stokesSolverAN.solve(print_stats=True, reinitialise=True)


# uw.libUnderworld.Underworld.AXequalsX( stokesSLE_AN._rot._cself, stokesSLE_AN._velocitySol._cself, False)
top = upperSurf
surfaceArea = uw.utils.Integral(
    fn=1.0, mesh=mesh, integrationType="surface", surfaceIndexSet=top
)
surfacePressureIntegral = uw.utils.Integral(
    fn=pressureField, mesh=mesh, integrationType="surface", surfaceIndexSet=top
)

# a callback function to calibrate the pressure - will pass to solver later
NodePressure = uw.mesh.MeshVariable(mesh, nodeDofCount=1)
Cell2Nodes = uw.utils.MeshVariable_Projection(NodePressure, pressureField, type=0)
Nodes2Cell = uw.utils.MeshVariable_Projection(pressureField, NodePressure, type=0)


NodePressure = uw.mesh.MeshVariable(mesh, nodeDofCount=1)
Cell2Nodes = uw.utils.MeshVariable_Projection(NodePressure, pressureField, type=0)
Nodes2Cell = uw.utils.MeshVariable_Projection(pressureField, NodePressure, type=0)


def smooth_pressure():
    if uw.mpi.rank == 0:
        print("Smoothing Pressure...")
    Cell2Nodes.solve()
    Nodes2Cell.solve()


def pressure_calibrate():
    if uw.mpi.rank == 0:
        print("Calibration and Smoothing of PressureField... ")
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    offset = p0 / area
    if uw.mpi.rank == 0:
        print("Zeroing pressure using mean upper surface pressure {}".format(offset))
    pressureField.data[:] -= offset
    smooth_pressure()
    if uw.mpi.rank == 0:
        print("Calibration and Smoothing of PressureField... Done")
    sys.stdout.flush()


def postSolve():
    pressure_calibrate()
    # realign solution
    uw.libUnderworld.Underworld.AXequalsX(
        stokesSLE._rot._cself, stokesSLE._velocitySol._cself, False
    )
    # remove null space
    uw.libUnderworld.StgFEM.SolutionVector_RemoveVectorSpace(
        stokesSLE._velocitySol._cself, stokesSLE._vnsVec._cself
    )


# stokesSolverAN.solve(print_stats=True, reinitialise=True)

while step < maxSteps:
    setVbc()  # seems important
    stokesSolverAN.solve(callback_post_solve=postSolve)
    # uw.libUnderworld.Underworld.AXequalsX(
    #     stokesSLE._rot._cself, stokesSLE._velocitySol._cself, False
    # )
    dt = advector.get_max_dt()
    advector.integrate(dt)
    swarm_popcontrol.repopulate()
    step += 1
    time += dt
    projVisMesh.solve()
    if step % 5 == 0 or step == 1 or step == 2:
        store.step = step
        figV.save(outputDir + "/eta" + str(step).zfill(5))
        figVdot.save(outputDir + "/Vdot" + str(step).zfill(5))
    if step % 10 == 0 or step == 1 or step == 2:
        checkpoint(
            mesh,
            fieldDict,
            swarm,
            swarmDict,
            index=step,
            modeltime=dm(time, 1.0 * u.megayear).magnitude,
            prefix=outputDir,
        )

if tflag:
    uw.timing.print_table(
        output_file=outputDir + "/uwTimer.log", display_fraction=0.999
    )
