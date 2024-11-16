1) Configure the sspseudo.config with Atom code path and your work folder to save the pseudopotentials
2) Depend on the python library: Numpy, MatlibPlot, Mendeleev and Scipy
3) to use the example
   
   $ sspseudopotential.x -md pg -el Si -fxc pbr -mps ker

 md: calculation mode - all electron (ae) or pseudopotential (ps)
 el: chemical element - atom element to calculate (atomic symbol)
 fxc: exchange-correlation functional - same used in Atom code manual
 mps: pseudopotential method -  Kerker (ker), Troullier-Martins (tm2), Hamann-Schluter-Chiang (bsc)
