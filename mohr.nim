import algorithm, os, math, strutils
import morpheus


# From: https://csmbrannon.net/2011/04/25/stress-state-analysis-python-script/

# ------------------------------------------------------------
# Helper functions for uniform printing throughout the script.
# ------------------------------------------------------------

proc headerprint(s: string) =
   # Prints a centered string to divide output sections.
   const mywidth = 64
   const mychar = "="
   let numspaces = mywidth - len(s)
   let before = int(ceil(numspaces / 2))
   let after = int(floor(numspaces / 2))
   echo("\n" & mychar.repeat(before) & s & mychar.repeat(after) & "\n")

proc valprint(s: string, value: float) =
   # Ensure uniform formatting of scalar value outputs.
   echo(s.align(30), ": ", value.formatEng)

proc matprint(s: string, value: Matrix) =
   # Ensure uniform formatting of matrix value outputs.
   echo(s, ":\n", value)

proc usage() =
   # When the user needs help, print the script usage.
   headerprint(" Analyze Stress State ")
   let appname = getAppFilename().extractFilename()
   let s = """ For a given stress state this script computes many
 useful quantities that help to analyze the stress state.

 Currently, the following values are output:
   Isotropic Matrix
   Deviatoric Matrix
   Principal Stresses
   Maximum Shear
   Mean Stress
   Equivalent Stress
   Invariant I1
   Invariant J2
   Invariant J3
   Lode Coordinates
   Triaxiality

 Command line syntax option 1:

     > ./$1 sig11 sig22 sig33

 Command line syntax option 2:

     > ./$1 sig11 sig22 sig33 sig12 sig13 sig23""" % appname
   quit(s)

# ------------------------------
# The main section of the script
# ------------------------------

proc main() =
   # If the script is run with an incorrect number of arguments
   # or if the user is asking for help, print the usage information.
   let params = commandLineParams()
   if "--help" in params or "-h" in params or
      params.len != 3 and params.len != 6:
      usage()

   # load stress components from the command line in a temporary
   # container
   var dum = newSeq[float64](6)
   for idx in 0 ..< params.len:
      try:
         dum[idx] = parseFloat(params[idx])
      except ValueError:
         quit("Argument '$1' is not a valid float" % params[idx])

   # load the stresses into our matrix and compute the
   # deviatoric and isotropic stress matricies
   let
      sigma = matrix(@[
         @[dum[0], dum[3], dum[4]],
         @[dum[3], dum[1], dum[5]],
         @[dum[4], dum[5], dum[2]]
      ])
      sigma_iso = 1.0/3.0*trace(sigma)*identity(sigma.m)
      sigma_dev = sigma - sigma_iso

   # compute principal stresses
   var eigvals = eig(sigma).getRealEigenvalues
   eigvals.sort(cmp, Descending)

   # compute max shear stress
   let maxshear = (eigvals.max-eigvals.min)/2.0

   # compute the stress invariants
   let
      I1 = trace(sigma)
      J2 = 1.0/2.0*trace(sigma_dev*sigma_dev)
      J3 = 1.0/3.0*trace(sigma_dev*sigma_dev*sigma_dev)

   # compute other common stress measures
   let
      mean_stress = 1.0/3.0*I1
      eqv_stress  = sqrt(3.0*J2)

   # compute lode coordinates
   let
      lode_r = sqrt(2.0*J2)
      lode_z = I1/sqrt(3.0)

      temp = 3.0*sqrt(6.0)*det(sigma_dev/lode_r)
      lode_theta = 1.0/3.0*arcsin(temp)

   # compute the stress triaxiality
   let triaxiality = mean_stress/eqv_stress

   # Print out what we've found
   headerprint(" Stress State Analysis ")
   matprint("Input Stress", sigma)
   headerprint(" Component Matricies ")
   matprint("Isotropic Stress", sigma_iso)
   matprint("Deviatoric Stress", sigma_dev)
   headerprint(" Scalar Values ")
   valprint("P1", eigvals[0])
   valprint("P2", eigvals[1])
   valprint("P3", eigvals[2])
   valprint("Max Shear", maxshear)
   valprint("Mean Stress", mean_stress)
   valprint("Equivalent Stress", eqv_stress)
   valprint("I1", I1)
   valprint("J2", J2)
   valprint("J3", J3)
   valprint("Lode z", lode_z)
   valprint("Lode r", lode_r)
   valprint("Lode theta (rad)", lode_theta)
   valprint("Lode theta (deg)", radToDeg(lode_theta))
   valprint("Triaxiality", triaxiality)
   headerprint(" End Output ")


main()

# End of script
