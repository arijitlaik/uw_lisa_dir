import underworld as uw
import glucifer
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


fn = uw.function

mesh = uw.mesh.FeMesh_Cartesian(elementRes=(64, 64))
fig = glucifer.Figure()
fig.Mesh(mesh)
fig.show()

lonelySwarm = uw.swarm.Swarm(mesh)
lonelySwarm.add_particles_with_coordinates(np.array([[0, 1], [1, 0], [0, 0]]))
fig.Points(lonelySwarm, pointsize=10)

fig.show()

get_coords = fn.coord()
coordsGlobal = None
print("before run:, ", uw.rank(), coordsGlobal)
uw.barrier()

coordsLocal = (
    get_coords.evaluate(lonelySwarm)[0].tolist()
    if get_coords.evaluate(lonelySwarm) is not None
    else None
)
uw.barrier()
# coordsLocal.shape[0]
print(uw.rank(), coordsLocal, lonelySwarm.particleCoordinates.data.shape)

coordsGlobal = np.array(comm.gather(coordsLocal, root=0))
# coordsGlobal = comm.gather(coordsLocal, root=0)
uw.barrier()


# print("After run:, ", uw.rank(), coordsGlobal)
if uw.rank() == 0:
    print("After run: ", uw.rank(), coordsGlobal)
    coordsGlobal = np.delete(coordsGlobal, np.where(coordsGlobal == None))
    coordsGlobal = np.array(list(coordsGlobal), dtype=np.float)
    print("-------------------------")
    print("After fliter: Rank {0},\nArray:{1} ".format(uw.rank(), coordsGlobal))
    print("-------------------------")
    np.savetxt("coordsGlobal.dat", coordsGlobal, delimiter=';')
    np.save("coordsGlobal.npy", coordsGlobal)

uw.barrier()
#
# lonelySwarm.particleCoordinates.data[0]
# try:
#     lonelySwarm.particleCoordinates.data[0]
# except IndexError:
#     print("Lonely swarm is not at Processor-{0:d}".format(uw.rank()))
# else:
#     print(
#         "The lonely swarm is at Processor {0:d} and is sitting at {1}".format(
#             uw.rank(), lonelySwarm.particleCoordinates.data[0]
#         )
#     )
