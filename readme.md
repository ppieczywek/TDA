# Tensile data analysis (TDA)

Repository contains a set of scripts designed to analyze data obtained from uniaxial mechanical tests (cyclic or extension to break). This set of functions allows extraction of such data as force/maximum stress, force/stress at rupture, Young's modulus/stiffness (by searching for linear segments on the force-strain curve and calculating their slopes).

The tda class has an additional auxiliary function dedicated to upload data from Deben Microtest miniature testing machine output files *.mtr" (see https://deben.co.uk/support/microtest-tensile-stages/). 

### tda.py
A script containing all the functions needed to analyze mechanical test data.

### example.py
Examples of data analysis from the "DATA" directory.