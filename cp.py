
restartFlag=False
try:
    fH=open("checkpoint.log","r")
except IOError:
    restartFlag=False
else:
    restartFlag=True
    lnF=fH.readlines()[-1]
    lC=lnF.split(",")
    rstep=float(lC[0])
    rtimeDm=float(lC[-1].split(";")[0])
    fH.close()
