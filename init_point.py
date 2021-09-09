import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


file_name = "./DATA/NANO_K_11.mtr"
header_length = 19
test_data = pd.read_csv(file_name, sep=",", skiprows=header_length, decimal=".")
test_data = test_data[['Elongation', 'Force']]
# print(test_data)

test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()

test_data = test_data.iloc[0:test_data['Force'].argmax(), :]

test_data["Force"] = test_data["Force"] - test_data["Force"].min()
test_data["Force"] = test_data["Force"] / test_data["Force"].max()

wnd_half_width = 41
window = np.concatenate((np.ones(wnd_half_width)/wnd_half_width,
                         np.zeros(wnd_half_width)), axis=None)

EY2 = signal.convolve(test_data["Force"].values**2, window, mode='valid')
E2Y = signal.convolve(test_data["Force"].values, window, mode='valid')**2

EY2_REV = signal.convolve(test_data["Force"].values**2, np.flip(window, axis=0), mode='valid')
E2Y_REV = signal.convolve(test_data["Force"].values, np.flip(window, axis=0), mode='valid')**2

VAR_BACK = EY2 - E2Y
VAR_AHEAD = EY2_REV - E2Y_REV

# VAR_BACK[VAR_BACK < 0.00001] = 0.00001
VAR_AHEAD[VAR_AHEAD <= 0.0] = 0.000001

force = test_data["Force"][wnd_half_width:(len(test_data)-wnd_half_width)]

sig = VAR_BACK/VAR_AHEAD
init_pos = sig.argmax() + np.floor(0.5*wnd_half_width)

fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(test_data['Elongation'], test_data['Force'], '-')

x = test_data["Elongation"][init_pos]
y = test_data["Force"][init_pos]
ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
ax1.plot(x, y, 'ro')

ax2.plot(sig, '-')

plt.show()
