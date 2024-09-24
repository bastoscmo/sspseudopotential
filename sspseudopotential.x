#!/usr/bin/env python3
########## Pseudopotential creator ##############################
#                                                               #
# Dr. Carlos Maciel de Oliveira Bastos                          #
#                                                               #
# Institute of Physics - University of Brasilia-DF- Brazil      #
#                                                               #
#    versão 1.0      date 20/09/2024                            #
#################################################################

#### LIBRARY
import numpy as np
import matplotlib.pyplot as plt
import mendeleev as mdl
import sys
import os
import scipy as sc


################# DICTIONARY #################
class_fxc={
          "wi":"LDA.Winger",
          "wis":"LDA.Winger-sp",
          "wir":"LDA.Winger-rel",
          "hl":"LDA.HL",
          "hls":"LDA.HL-sp",
          "hlr":"LDA.HL-rel",
          "gl":"LDA.GL",
          "gls":"LDA.GL-sp",
          "glr":"LDA.GL-rel",
          "bh":"LDA.BH",
          "bhs":"LDA.BH-sp",
          "bhr":"LDA.BH-rel",
          "ca":"LDA.CA",
          "cas":"LDA.CA-sp",
          "car":"LDA.CA-rel",
          "pw":"LDA.PW92",
          "pws":"LDA.PW92-sp",
          "pwr":"LDA.PW92-rel",
          "pb":"GGA.PBE",
          "pbs":"GGA.PBE-sp",
          "pbr":"GGA.PBE-rel",
          "wp":"GGA.PW91",
          "wps":"GGA.PW91-sp",
          "wpr":"GGA.PW91-rel",
          "rp":"GGA.RPBE",
          "rps":"GGA.RPBE-sp",
          "rpr":"GGA.RPBE-rel",
          "rv":"GGA.revPBE",
          "rvs":"GGA.revPBE-sp",
          "rvr":"GGA.revPBE-rel",
          "ps":"GGA.PBEsol",
          "pss":"GGA.PBEsol-sp",
          "psr":"GGA.PBEsol-rel",
          "wc":"GGA.WC",
          "wcs":"GGA.WC-sp",
          "wcr":"GGA.WC-rel",
          "jo":"GGA.PBEJsJrLO",
          "jos":"GGA.PBEJsJrLO-sp",
          "jor":"GGA.PBEJsJrLO-rel",
          "jh":"GGA.PBEJsJrHEG",
          "jhs":"GGA.PBEJsJrHEG-sp",
          "jhr":"GGA.PBEJsJrHEG-rel",
          "go":"GGA.PBEGcGxLO",
          "gos":"GGA.PBEGcGxLO-sp",
          "gor":"GGA.PBEGcGxLO-rel",
          "gh":"GGA.PBEGcGxHEG",
          "ghs":"GGA.PBEGcGxHEG-sp",
          "ghr":"GGA.PBEGcGxHEG-rel",
          "bl":"GGA.BLYP",
          "bls":"GGA.BLYP-sp",
          "blr":"GGA.BLYP-rel",
          "vw":"VdW.DRSLL",
          "vws":"VdW.DRSLL-sp",
          "vwr":"VdW.DRSLL-rel",
          "vf":"VdW.DRSLL",
          "vfs":"VdW.DRSLL-sp",
          "vfr":"VdW.DRSLL-rel",
          "vl":"VdW.LMKLL",
          "vls":"VdW.LMKLL-sp",
          "vlr":"VdW.LMKLL-rel",
          "vk":"VdW.KBM",
          "vks":"VdW.KBM-sp",
          "vkr":"VdW.KBM-rel",
          "vc":"VdW.C09",
          "vcs":"VdW.C09-sp",
          "vcr":"VdW.C09-rel",
          "vb":"VdW.BH",
          "vbs":"VdW.BH-sp",
          "vbr":"VdW.BH-rel",
          "vv":"VdW.VV",
          "vvs":"VdW.VV-sp",
          "vvr":"VdW.VV-rel"
          }

################ FUNCTIONS ########################

# Read sspseudopotential.config
def read_config():
  dirssp = os.path.dirname(os.path.abspath(__file__))
  pathssp = os.path.join(dirssp, "sspseudo.config")
  configfile = np.loadtxt(pathssp,str)
  atompath=configfile[0]
  workpath=configfile[1]
  return atompath,workpath

# Order the valence distribution and electrons
def de_format(element):
  de_valence =str( mdl.element(element).ec.get_valence())
  de = de_valence.split()
  nmax=mdl.element(element).ec.max_n()
# Select in the valence states all states with high level number
# and incomplete sublevel. Another not are select as valence sublevel.
  selsub=[]
  for i in range(len(de)):
    if int(de[i][0])==nmax:
      selsub.append(de[i])
    else:
      if str(de[i][1])=="s" and int(de[i][2:])!=2:
        selsub.append(de[i])
      if str(de[i][1])=="p" and int(de[i][2:])!=6:
        selsub.append(de[i])
      if str(de[i][1])=="d" and int(de[i][2:])!=10:
        selsub.append(de[i])
      if str(de[i][1])=="f" and int(de[i][2:])!=14:
        selsub.append(de[i])
  # substitute s,p,d and f for numbers 0,1,2 and 3
  de_num=[]
  de_slipt=[]
  for i in range(len(selsub)):
    if str(selsub[i][1])=="s":
      l='0'
    elif str(selsub[i][1])=='p':
      l='1'
    elif str(selsub[i][1])=='d':
      l='2'
    elif str(selsub[i][1])=='f':
      l='3'
    # Divide electron in spin up and down
    spinup =  int(selsub[i][2:])// 2
    spindown = int(selsub[i][2:]) -  spinup
    de_num.append([selsub[i][0],l,spinup,spindown])
  return np.array(de_num,str),selsub

### Create file input for ae
def inputfileatom_ae(element,fxc):
  #organize the electronic distribution
  de_electrons,selsub = de_format(element)
  #Variables define
  ncore=len(mdl.element(element).ec.spin_occupations())-len(de_electrons)
  nval=len(de_electrons)
  #write the input atom
  inputatom=[]
  inputatom.append("   ae  "+element+" All electron - valence: " + str( " ".join(selsub) ) )
  if len(element)==1:
    inputatom.append("   "+element+"    "+fxc+" ")
  if len(element)==2:
    inputatom.append("   "+element+"   "+fxc+" ")
  inputatom.append("       0.0")
  if len(str(ncore))==1:
    inputatom.append("    "+str(ncore)+"    "+str(nval))
  if len(str(ncore))==2:
    inputatom.append("   "+str(ncore)+"    "+str(nval))
  for i in range(nval):
    inputatom.append("    "+de_electrons[i][0]+"    "+de_electrons[i][1]+"      "+str(de_electrons[i][2])+".00   "+str(de_electrons[i][3])+".00")
  return inputatom

### Create file input for pseudopotential
def inputfileatom_pg(element,fxc,rc):
  #organize the electronic distribution
  de_electrons,selsub = de_format(element)
  #Variables define
  ncore=len(mdl.element(element).ec.spin_occupations())-len(de_electrons)
  nval=len(de_electrons)
  #write the input atom
  inputatom=[]
  inputatom.append("   pg  "+element+" All electron - valence: " + str( " ".join(selsub) ) )
  inputatom.append("        "+mpseudo+"     3.0")
  if len(element)==1:
    inputatom.append("   "+element+"    "+fxc+" ")
  if len(element)==2:
    inputatom.append("   "+element+"   "+fxc+" ")
  inputatom.append("       0.0")
  if len(str(ncore))==1:
    inputatom.append("    "+str(ncore)+"    "+str(nval))
  if len(str(ncore))==2:
    inputatom.append("   "+str(ncore)+"    "+str(nval))
  for i in range(nval):
    inputatom.append("    "+de_electrons[i][0]+"    "+de_electrons[i][1]+"      "+str(de_electrons[i][2])+".00   "+str(de_electrons[i][3])+".00")
  inputatom.append("      "+"      ".join(rc))
  return inputatom

# Found the recommend values for rc
def rc_range(element,pathfolder_ae):
  lname=["s","p","d","f"]
  min_max_wf=[]
  limit=0.005
  de_electrons,selsub = de_format(element)
  rc=[]
  wf_ae=[]
  for i in range(len(selsub)):
    if selsub[i][1]=="s":
      lind=0
    elif selsub[i][1]=="p":
      lind=1
    elif selsub[i][1]=="d":
      lind=2
    elif selsub[i][1]=="f":
      lind=3
    wf_ae.append(np.loadtxt(pathfolder_ae+"/AEWFNR"+str(lind),dtype=float))
  # found the max and minimum of wavefunctions
  for i in range(len(selsub)):
    max_y = max(y for x, y in wf_ae[i][:,:2])
    min_y = min(y for x, y in wf_ae[i][:,:2])
    wfmax= [[x, y] for x, y in wf_ae[i][:,:2] if y == max_y]
    wfmin= [[x, y] for x, y in wf_ae[i][:,:2] if y == min_y]
    wf_zeros = [[x, y] for x, y in wf_ae[i][:,:2] if abs(y) < limit and wfmin[0][0]<x<wfmax[0][0]]
    #If the limit is not sufficient   
    while not  wf_zeros:
      limit=limit+0.005
      wf_zeros = [[x, y] for x, y in wf_ae[i][:,:2] if abs(y) < limit and wfmin[0][0]<x<wfmax[0][0]]
    min_max_wf.append([wf_zeros[-1],wfmax])
    #Write in screen
    print("---")
    print("Wave function recommend rc range for "+selsub[i][1]+":")
    print("rc_min:" + str(min_max_wf[i][0][0])+ " (last zero of wf)")
    print("rc_max:" + str(min_max_wf[i][1][0][0])+ " (maximum of wf)")
    print("---\n\n")
  return min_max_wf

# Out plots of wavefunctions and charge density
def outplot(element,pathfolder_sr):
  #Read Files and identification spdf
  lname=["s","p","d","f"]
  de_electrons,selsub = de_format(element)
  rc=[]
  wf_ae=[]
  wf_ps=[]
  ch_ae=[]
  ch_ps=[]
  for i in range(len(selsub)):
    if selsub[i][1]=="s":
      lind=0
    elif selsub[i][1]=="p":
      lind=1
    elif selsub[i][1]=="d":
      lind=2
    elif selsub[i][1]=="f":
      lind=3
    wf_ae.append(np.loadtxt(pathfolder_sr+"/AEWFNR"+str(lind),dtype=float))
    wf_ps.append(np.loadtxt(pathfolder_sr+"/PSWFNR"+str(lind),dtype=float))
    #plot wavefunctions
    plt.plot(wf_ae[i][:,0],wf_ae[i][:,1],label=selsub[i][1]+" - AE")
    plt.plot(wf_ps[i][:,0],wf_ps[i][:,1],label=selsub[i][1]+" - PS")
    plt.legend()
    plt.xlim(0,7)
    plt.title(element+ " - wavefunction - el. config: "+" ".join(selsub))
    plt.show()
  #Read Charges charge
  ch_ae.append(np.loadtxt(pathfolder_sr+"/CHARGE",dtype=float))
  ch_ps.append(np.loadtxt(pathfolder_sr+"/PSCHARGE",dtype=float))
  #plot Charges
  plt.plot(ch_ae[0][:,0],ch_ae[0][:,1]+ch_ae[0][:,2]-ch_ae[0][:,3],label="AE")
  plt.plot(ch_ps[0][:,0],ch_ps[0][:,1]+ch_ps[0][:,2]-ch_ps[0][:,3],label="PS")
  plt.xlim(0,7)
  plt.title(element+ " - valence electrons charge - el. config: "+" ".join(selsub))
  plt.legend()
  plt.show()
  return


# write information on screen part 1
def swrite_init(element):
  de_electrons,selsub = de_format(element)
  print("------------------------")
  print("SSPSEUDOPOTENTIAL CREATOR")
  print("--------v. 1.0 ---------\n")
  print("---")
  print("Atom identified: " + str(element))
  print("Valence Electronic configuration: "+" ".join(selsub) )
  print("Exchange correlation: " + str(fxc))
  print("Pseudopotential method used: " + str(mpseudo))
  print("---\n")
  print("--- All Electron Calculation ---")
  print("Creating folder...")
  print("Creating INP file...")
  print("Calling atom code...")
  return

# write information on screen part 2
def swrite_pseudo():
  print("--- Pseudopotential Calculation ---")
  print("Creating folder...")
  print("Creating INP file...")
  print("Calling atom code...")



######### Main Code ##################
### Read Config File
pathatom,pathwork=read_config()

### define flags
mode = str(sys.argv[2])
element=str(sys.argv[4])
fxc=str(sys.argv[6])
mpseudo=str(sys.argv[8])

### Define paths
pathfolder_pp=pathwork+"/"+"pseudopotentials/"+class_fxc[fxc]
pathfolder_sr=pathwork+"/"+"source/"+class_fxc[fxc]+"/"+element+".pg."+mpseudo
pathfolder_ae=pathwork+"/"+"source/"+class_fxc[fxc]+"/"+element+".ae."+mpseudo


#create folder
if not os.path.exists(pathfolder_pp):
  os.makedirs(pathfolder_pp)  
if not os.path.exists(pathfolder_sr):
  os.makedirs(pathfolder_sr)
if not os.path.exists(pathfolder_ae):
  os.makedirs(pathfolder_ae)


#all electron mode
if mode == "ae":
  #create folder
  np.savetxt(pathfolder_ae+"/INP",inputfileatom_ae(element,fxc),fmt='%s')
  #call atom code
  os.system("cd "+pathfolder_ae+"; "+pathatom)

#Pseudopotential mode
elif mode == "pg":
  swrite_init(element)
  #create all electron calculation to determine the cutoff radii
  np.savetxt(pathfolder_ae+"/INP",inputfileatom_ae(element,fxc),fmt='%s')
  print("----- ATOM WARNING -----\n")
  #call atom code
  os.system("cd "+pathfolder_ae+"; "+pathatom)
  print("\n----- ATOM WARNING -----\n")

  print("finished atom calculation...")
  print("Calculating the radii cutoff...\n\n")
  rc_rg=rc_range(element,pathfolder_ae)
  rc=[]
  lname=["s","p","d","f"]
  for i in range(len(rc_rg)):
    rc.append(str((rc_rg[i][0][0]+rc_rg[i][1][0][0])/2.)[:4])
    print("---")
    print(" avg rc used for "+lname[i]+ ": "+ rc[i])
  print("---\n\n")
  #create the psedudopotential
  swrite_pseudo()
  np.savetxt(pathfolder_sr+"/INP",inputfileatom_pg(element,fxc,rc),fmt='%s')
  print("----- ATOM WARNING -----\n")
  #call atom code
  os.system("cd "+pathfolder_sr+"; "+pathatom)
  print("\n----- ATOM WARNING -----\n\n")
  print("finished pseudopotential calculation...")
  print("Plotting results...\n\n")
  #plotting graphics
  outplot(element,pathfolder_sr)
  print("Creating a pseudopotential folder...\n\n")
  #organize the pseudopotential
  os.system("cp "+pathfolder_sr+"/VPSOUT "+pathfolder_pp+"/"+element+"."+mpseudo+".vps")
  print(" --------------")
  print("|  Finished!   |")
  print(" --------------")





