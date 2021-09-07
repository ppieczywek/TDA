import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np


file_name = "./DATA/NANO_K_5.mtr"
header_length = 19
test_data = pd.read_csv(file_name, sep=",", skiprows=header_length, decimal=".")
test_data = test_data[['Elongation', 'Force']]
# print(test_data)

test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()

test_data = test_data.iloc[0:test_data['Force'].argmax(), :]

test_data["dF"] = scipy.signal.savgol_filter(test_data["Force"].values, 31, 1,
                                             deriv=1, delta=1.0, axis=- 1,
                                             mode='interp', cval=0.0)

test_data["dF"] = test_data["dF"] - test_data["dF"].min()
test_data["dF"] = test_data["dF"] / test_data["dF"].max()
test_data["Elongation"] = test_data["Elongation"] / test_data["Elongation"].max()

test_data["dF"] = scipy.signal.savgol_filter(test_data["dF"].values, 51, 1)

half_window = 50

slopes = np.zeros(len(test_data["dF"]))
for i in range(half_window, len(test_data["dF"]) - half_window):
    a = np.polyfit(test_data.Elongation[i-half_window:i+half_window],
                   test_data.dF[i-half_window:i+half_window], 1)[0]
    # print(a)
    slopes[i] = a

test_data["slopes"] = slopes

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(test_data['Elongation'], test_data['Force'], '-')
ax2.plot(test_data['Elongation'], test_data['dF'], '-')
ax3.plot(test_data['Elongation'], test_data['slopes'], '-')
plt.show()


# print(test_data.columns)
