import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from mtr import mtr

file_name = "./DATA/NANO_K_4.mtr"
header_length = 19
test_data = pd.read_csv(file_name, sep=",", skiprows=header_length, decimal=".")
test_data = test_data[['Elongation', 'Force']]
print(test_data)

test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()

test_data = test_data.iloc[0:test_data['Force'].argmax(), :]

init_pos = mtr.get_init_point(test_data["Force"].values)

fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(test_data['Elongation'], test_data['Force'], '-')

x = test_data["Elongation"][init_pos]
y = test_data["Force"][init_pos]
ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
ax1.plot(x, y, 'ro')
#
# ax2.plot(sig, '-')

plt.show()
