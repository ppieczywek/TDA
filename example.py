# import pandas as pd

from tda import tda

(force, elongation) = tda.read_mtr("DATA\\NANOCEL_KW.CHAL_450PPM_15.MTR")
tensile_test_results = tda.process_tensile_test(force, elongation)
print(tensile_test_results)
print('\n')
(force, elongation) = tda.read_mtr("DATA\\BC_0_W_R.013.MTR")
cyclic_test_results = tda.process_cyclic_test(force, elongation)
print(cyclic_test_results)
