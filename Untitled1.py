
# coding: utf-8

# In[243]:



step = maxSteps+-1

while step < maxSteps:

   if uw.rank() == 0:
       print("Stokes Solver Started...")
   ntol = 1e-2 if step == 0 else 1e-2
   solver.solve(
       nonLinearIterate=True,
       nonLinearTolerance=ntol,
       callback_post_solve=pressure_calibrate,
   )

   Vrms = np.sqrt(mesh.integrate(vdotv)[0] / mesh.integrate(1.0)[0])
   # update
   time, step, dt = model_update()
   dmTime = dm(time, 1.0 * u.megayear).magnitude
   if uw.rank() == 0:
       logFile = open(outputDir + "/runLog.log", "a")

       stepLog = "step = {0:6d}; dt = {1:.3e} Nd; time = {2:.3e} Ma, Vrms = {3:5e}\n".format(
           step, dt, dmTime, Vrms)
       # print step
       # print dt
       # print time
       # uw.timing.print_table(output_file=outputDir+'/uwTimer.log')
       # uw.timing.start()

       print(stepLog)
       logFile.write(stepLog)
       logFile.close()
   if step % 100 == 0 or step == 1:

       checkpoint(step=step, time=dmTime)
   sys.stdout.flush()

   uw.barrier()


# uw.timing.print_table(output_file=outputDir+'/Time.Log')
# Testing Block


# In[192]:


indentorshapes = make_Indentor2d(
    startX=nd(0.3 * modelHeight),
    topY=nd(0.0 * u.kilometer),
    length=nd(0.85 * modelHeight),
    taper=18,
    thicknessArray=[
        nd(15.0 * u.kilometer),
        nd(15.0 * u.kilometer),
        nd(30.0 * u.kilometer),
        nd(50.0 * u.kilometer),
    ],  # UL
    taper2=12,
)


# In[193]:


slabshapes = make_slab2d(
    topX=nd(0.725 * modelHeight),
    topY=0.0,
    length=nd(1.275 * modelHeight),
    taper=15,
    dip=29,
    depth=nd(120.0 * u.kilometer),
    thicknessArray=[
        nd(15.0 * u.kilometer),
        nd(15.0 * u.kilometer),
        nd(30.0 * u.kilometer),
        nd(30.0 * u.kilometer),
    ],  # thic
)


# In[204]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = [18, 4]


# In[205]:


plt.rcParams['figure.figsize'] = [18, 4]
slabCoords = np.unique(np.concatenate(np.array(slabshapes)[:, :, :]), axis=0)
indCoords = np.unique(np.concatenate(np.array(indentorshapes)[:, :, :]), axis=0)
spPoints=np.concatenate((slabCoords,indCoords),axis=0)


# In[215]:





# In[207]:


from scipy.spatial import ConvexHull,Delaunay


# In[208]:


points=spPoints


# In[209]:


opHull = ConvexHull(slabCoords)


# In[210]:


plt.plot(slabCoords[:,0], slabCoords[:,1], 'o')
for simplex in opHull.simplices:
    plt.plot(slabCoords[simplex,0], slabCoords[simplex,1], 'k-')


# In[211]:


# opHull.add_points([ 2., -0.0312])


# In[212]:


plt.triplot(points[:,0], points[:,1], tri.vertices)
plt.plot(points[:,0], points[:,1], 'o')


# In[218]:



plt.scatter(slabCoords[:, 0], slabCoords[:, 1])
# plt.plot(slabCoords[:, 0], slabCoords[:, 1])
# plt.scatter(indCoords[:, 0], indCoords[:, 1])
# plt.plot(slabCoords[1:][:, 0],slabCoords[1:][:, 1])
# plt.plot(indCoords[:, 0], indCoords[:, 1])
# plt.plot(xS,yS)
# for l,r in zip(inEp[0],np.flip(inEp[1],0)):
#     xyS=np.stack((np.linspace(l[0],r[0],100),np.full((100,),l[1])), axis=-1)
#     plt.plot(xyS[:,0],xyS[:,1])
spPoints=np.concatenate((slabCoords,indCoords),axis=0)
plt.tight_layout()


# In[187]:


inEp = np.split(indCoords, 2)


# In[188]:


opTx=[]
for l,r in zip(inEp[0],np.flip(inEp[1],0)):
    xyS=np.stack((np.linspace(l[0],r[0],100),np.full((100,),l[1])), axis=-1)
    plt.plot(xyS[:,0],xyS[:,1])



# In[189]:


import networkx as nx


# In[173]:


import sys
get_ipython().system(u'{sys.executable} -m pip install --user networkx')


# In[174]:


import sys
get_ipython().system(u'{sys.executable} -m ')


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(slabshapes[:][0], slabshapes[:][1])


slabCoords = np.unique(np.concatenate(np.array(slabshapes)[:, :, :]), axis=0)
indCoords = np.unique(np.concatenate(np.array(indentorshapes)[:, :, :]), axis=0)
get_ipython().magic(u'matplotlib')
plt.scatter(slabCoords[:, 0], slabCoords[:, 1])
plt.plot(slabCoords[:, 0], slabCoords[:, 1])
plt.scatter(indCoords[:, 0], indCoords[:, 1])
plt.plot(indCoords[:, 0], indCoords[:, 1])
plt.axis('equal')
inEp = np.split(indCoords, 2)
inEp
# inR = np.flip(inR)
for l,r in zip(inEp[0],np.flip(inEp[1],0)):
    print(l,r)



plt.plot(inR[:, 0], inR[:, 1])
plt.plot(inL[:, 0], inL[:, 1])

plt.plot(opRL[0:, 0])
plt.plot(opRL[1])


# In[217]:


tracerSwarm=uw.swarm.Swarm(mesh=mesh)


# In[221]:


tracerSwarm.add_particles_with_coordinates(spPoints)


# In[246]:


tracerAdv=uw.systems.SwarmAdvector(
    swarm=tracerSwarm, velocityField=velocityField, order=2)


# In[238]:


figParticleT = glucifer.Figure(
    store, figsize=tuple(map(lambda x: x/3,figSize)), quality=3, name="Tr", boundingBox=bBox)
figParticleT.Points(tracerSwarm,pointsize=10)
figParticleT.save("tr.png")


# In[248]:


dt = advector.get_max_dt()
tDt = tracerAdv.get_max_dt()


# In[252]:


vo=velocityField.evaluate(tracerSwarm)


# In[253]:


tracerAdv.integrate(dt)


# In[254]:


v1=velocityField.evaluate(tracerSwarm)


# In[256]:


v1-vo


# In[263]:


getCoors=fn.coord()


# In[265]:


getCoors.evaluate(tracerSwarm)
