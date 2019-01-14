mantleShape = make_layer2d(
    startX=0.,
    topY=0.,
    length=nd(modelHeight*aRatioCoor),
    thicknessArray=[nd(modelHeight)],
    # thicknessArray=[nd(660.*u.kilometer), nd(modelHeight-660.*u.kilometer)]
    )

slabshapes = make_slab2d(
    topX=nd(0.725*modelHeight), topY=0.,
    length=nd(1.275*modelHeight),
    taper=15, dip=29,
    depth=nd(120.*u.kilometer),
    thicknessArray=[nd(10.*u.kilometer), nd(20.*u.kilometer),
                    nd(20.*u.kilometer), nd(30.*u.kilometer)]
    )  # 10 20 20 30

indentorshapes = make_Indentor2d(
    startX=nd(0.3*modelHeight),
    topY=nd(0.*u.kilometer),
    length=nd(.85*modelHeight),
    taper=18,
    thicknessArray=[nd(10.*u.kilometer), nd(20.*u.kilometer),
                    nd(20.*u.kilometer), nd(50.*u.kilometer)],
    taper2=12)

overRidingShapesForeArc = make_overRidingPlate2d(
    topX=mesh.maxCoord[0]-mesh.minCoord[0],
    topY=nd(0.*u.kilometer),
    length=-nd(2.*modelHeight),
    taper=15,
    dip=29,
    thicknessArray=[nd(40.*u.kilometer), nd(80.*u.kilometer)]
    )

overRidingShapes = make_overRidingPlate2d(
    topX=mesh.maxCoord[0]-mesh.minCoord[0],
    topY=nd(0.*u.kilometer),
    length=nd(-1.875*modelHeight),
    taper=90, dip=90,
    thicknessArray=[nd(40.*u.kilometer), nd(80.*u.kilometer)]
    )
# define the viscosity Range
viscRange = [1., 1e5]
defaultSRInv = nd(1e-18 / u.second)

modelMaterials = [
    # {"name": "Air",
    #  "shape": mantleandAirShape[0],
    #  "viscosity":0.1*refViscosity,
    #  "density":0.*u.kilogram / u.meter**3},

    {"name": 'Mantle',
     "shape": mantleShape[0],
     "viscosity":"deptDependent",
     "eta0":refViscosity,
     "eta1":1e2*refViscosity,
     "etaChangeDepth":660.*u.kilometer,
     "density":"deptDependent",
     "rho0":3200.*u.kilogram / u.meter**3,
     "rho1":3230.*u.kilogram / u.meter**3,
     "rhoChangeDepth":660.*u.kilometer},
    # {"name": 'Upper Mantle',
    #  "shape": mantleShape[0],
    #  "viscosity":1*refViscosity,
    #  "density":3200.*u.kilogram / u.meter**3},
    # {"name": 'Lower Mantle',
    #  "shape": mantleShape[1],
    #  "viscosity":1e2*refViscosity,
    #  "density":3240.*u.kilogram / u.meter**3},



    # Indo-Australian Plate `4 plates`
    {"name": 'Uppper Crust Indo-Australian Plate',
     "shape": slabshapes[0],
     "viscosity": 1e2*refViscosity,
     "cohesion": 06.*u.megapascal,
     # "viscosity":"deptDependent",
     # "eta0":yield_visc(nd(06.*u.megapascal), nd(1e2*refViscosity)),
     # "eta1":yield_visc(nd(30.*u.megapascal), nd(5e1*refViscosity)),  # 5e1
     # "etaChangeDepth":150.*u.kilometer,
     "density":3280.*u.kilogram / u.meter**3, },
    {"name": 'Lower Crust Indo-Australian Plate',
     "shape": slabshapes[1],
     "viscosity":1e3*refViscosity,
     "cohesion":30.*u.megapascal,
     "density":3280.*u.kilogram / u.meter**3, },  # 5.*u.megapascal,
    {"name": 'Lithospheric Mantle Crust Indo-Australian Plate',
     "shape": slabshapes[2],
     "viscosity":1e5*refViscosity,
     "cohesion":350.*u.megapascal,
     "density":3280.*u.kilogram / u.meter**3},
    {"name": 'Lithospheric Mantle Indo-Australian Plate',
     "shape": slabshapes[3],
     "viscosity":5e2*refViscosity,
     "density":3280.*u.kilogram / u.meter**3,
     "cohesion":30.*u.megapascal},


    # Indian Indentor
    {"name": 'Upper Crust Indian Indentor',
     "shape": indentorshapes[0],
     # "viscosity":"deptDependent",
     # "eta0":yield_visc(nd(06.*u.megapascal), nd(1e2*refViscosity)),  # 1e2
     # "eta1":yield_visc(nd(30.*u.megapascal), nd(5e1*refViscosity)),  # 5e1
     # "etaChangeDepth":150.*u.kilometer,
     "viscosity": 1e2*refViscosity,
     "cohesion": 06.*u.megapascal,
     "density":2800.*u.kilogram / u.meter**3,
     # "density":"deptDependent",
     # "rho0":2800.*u.kilogram / u.meter**3,
     # "rho1":3280.*u.kilogram / u.meter**3,
     # "rhoChangeDepth":150.*u.kilometer
     },
    {"name": 'Lower Crust Indian Indentor',
     "shape": indentorshapes[1],
     "viscosity":1e2*refViscosity,
     "cohesion":30.*u.megapascal,
     "density":2800.*u.kilogram / u.meter**3,
     # "density":"deptDependent",
     # "rho0":2800.*u.kilogram / u.meter**3,
     # "rho1":3280.*u.kilogram / u.meter**3,
     # "rhoChangeDepth":150.*u.kilometer
     },
    {"name": 'Lithospheric Mantle Indian Indentor',
     "shape": indentorshapes[2],
     "viscosity":1e5*refViscosity,
     "cohesion":350.*u.megapascal,
     "density":3200.*u.kilogram / u.meter**3,
     # "density":"deptDependent",
     # "rho0":3200.*u.kilogram / u.meter**3,
     # "rho1":3280.*u.kilogram / u.meter**3,
     # "rhoChangeDepth":150.*u.kilometer
     },
    {"name": 'Lithospheric Mantle Indian Indentor',
     "shape": indentorshapes[3],
     "viscosity":5e4*refViscosity,
     "cohesion":30.*u.megapascal,
     "density":3220.*u.kilogram / u.meter**3
     # "density":"deptDependent",
     # "rho0":3220.*u.kilogram / u.meter**3,
     # "rho1":3280.*u.kilogram / u.meter**3,
     # "rhoChangeDepth":150.*u.
     },


    # Eurasian Plate
    {"name": "Crust Eurasian Plate ForeArc",
     "shape": overRidingShapesForeArc[0],
     "viscosity":1e3*refViscosity,
     "density":3200.*u.kilogram / u.meter**3},
    {"name": "Lithospheric Mantle Eurasian Plate ForeArc",
     "shape": overRidingShapesForeArc[1],
     "viscosity":5e2*refViscosity,
     "density":3200.*u.kilogram / u.meter**3},
    {"name": "Crust Eurasian Plate",
     "shape": overRidingShapes[0],
     "viscosity":5e2*refViscosity,
     "density":3200.*u.kilogram / u.meter**3},
    {"name": "Lithospheric Mantle Eurasian Plate",
     "shape": overRidingShapes[1],
     "viscosity":2e2*refViscosity,
     "density":3200.*u.kilogram / u.meter**3}

     ]
