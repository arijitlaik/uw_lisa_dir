import os
import underworld as uw

mesh = uw.mesh.FeMesh_Cartesian(elementRes=(128, 128))
swarm = uw.swarm.Swarm(mesh=mesh)
velocityField = mesh.add_variable(nodeDofCount=mesh.dim)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)
projVisc = mesh.add_variable(1)
materialVariable = swarm.add_variable(dataType="int", count=1)
tracerSwarm = uw.swarm.Swarm(mesh=mesh)
tincord = tracerSwarm.add_variable(dataType="double", count=2)


def checkpoint(
    mesh,
    fieldDict,
    swarm,
    swarmDict,
    index,
    meshName="mesh",
    swarmName="swarm",
    prefix="./",
    enable_xdmf=True,
):
    # Check the prefix is valid
    if prefix is not None:
        if not prefix.endswith("/"):
            prefix += "/"  # add a backslash
        if not os.path.exists(prefix) and uw.rank() == 0:
            print("Creating directory: ", prefix)
            os.makedirs(prefix)

    uw.barrier()

    if not isinstance(index, int):
        raise TypeError("'index' is not of type int")
    ii = str(index).zfill(5)

    if mesh is not None:

        # Error check the mesh and fields
        if not isinstance(mesh, uw.mesh.FeMesh):
            raise TypeError("'mesh' is not of type uw.mesh.FeMesh")
        if not isinstance(fieldDict, dict):
            raise TypeError("'fieldDict' is not of type dict")
        for key, value in fieldDict.iteritems():
            if not isinstance(value, uw.mesh.MeshVariable):
                raise TypeError(
                    "'fieldDict' must contain uw.mesh.MeshVariable elements"
                )

        # see if we have already saved the mesh. It only needs to be saved once
        if not hasattr(checkpoint, "mH"):
            checkpoint.mH = mesh.save(prefix + meshName + ".h5")
        mh = checkpoint.mH

        for key, value in fieldDict.iteritems():
            filename = prefix + key + "-" + ii
            handle = value.save(filename + ".h5")
            if enable_xdmf:
                value.xdmf(filename, handle, key, mh, meshName)

    # is there a swarm
    if swarm is not None:

        # Error check the swarms
        if not isinstance(swarm, uw.swarm.Swarm):
            raise TypeError("'swarm' is not of type uw.swarm.Swarm")
        if not isinstance(swarmDict, dict):
            raise TypeError("'swarmDict' is not of type dict")
        for key, value in swarmDict.iteritems():
            if not isinstance(value, uw.swarm.SwarmVariable):
                raise TypeError(
                    "'fieldDict' must contain uw.swarm.SwarmVariable elements"
                )

        sH = swarm.save(prefix + swarmName + "-" + ii + ".h5")
        for key, value in swarmDict.iteritems():
            filename = prefix + key + "-" + ii
            handle = value.save(filename + ".h5")
            if enable_xdmf:
                value.xdmf(filename, handle, key, sH, swarmName)


fieldDict = {
    "velocity": velocityField,
    "pressure": pressureField,
    "meshViscosity": projVisc,
}

swarmDict = {"materials": materialVariable}
traceDict = {"tcoords": tincord}

checkpoint(mesh, fieldDict, swarm, swarmDict, index=1, prefix="checkpointTest")
checkpoint(None, None, tracerSwarm, traceDict, index=1, prefix="checkpointTest")
