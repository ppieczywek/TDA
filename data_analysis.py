import pandas as pd
import matplotlib.pyplot as plt
from mtr import mtr

file_name = "./DATA/NANOCEL_KW.CHAL_900PPM_3.mtr"
header_length = 19
test_data = pd.read_csv(file_name, sep=",", skiprows=header_length, decimal=".")
test_data = test_data[['Elongation', 'Force']]
test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()


init_pos = mtr.get_init_point(test_data["Force"].values)
break_point = mtr.get_break_point(test_data["Force"].values)
limits = mtr.get_slopes_limits(test_data['Force'].values,
                               test_data['Elongation'].values, init_pos)
values = mtr.get_slopes_values(test_data['Force'].values,
                               test_data['Elongation'].values,
                               init_pos,
                               limits)

print(values)

#
#
# test_data = test_data.iloc[0:test_data['Force'].argmax(), :]
# test_data["dF"] = scipy.signal.savgol_filter(test_data["Force"].values, 25, 1,
#                                              deriv=1, delta=1.0, axis=- 1,
#                                              mode='interp', cval=0.0)
#
# test_data["dF"] = (test_data["dF"] - test_data["dF"].min()) / (test_data["dF"].max()-test_data["dF"].min())
# test_data["Elongation"] = test_data["Elongation"] / test_data["Elongation"].max()
#
# half_window = 30
#
# slopes = np.zeros(len(test_data["dF"]))
# for i in range(half_window, len(test_data["dF"]) - half_window):
#     a = np.polyfit(test_data.Elongation[i-half_window:i+half_window],
#                    test_data.dF[i-half_window:i+half_window], 1)[0]
#     slopes[i] = a
#
# slopes[range(len(slopes) - half_window, len(slopes))] = slopes[len(slopes) - (half_window+1)]
# slopes[range(0, half_window)] = slopes[half_window]
#
# test_data["slopes"] = slopes
# test_data["selection"] = ((test_data["slopes"].abs() < 0.5*test_data["slopes"].max()).astype(int))
#
# split_points = np.where(abs(np.diff(test_data["selection"].values, n=1, axis=-1)) == 1)[0]+1
# groups = np.split(test_data["selection"].values, split_points, axis=0)
#
# limits = np.column_stack((np.concatenate(([0], split_points), axis=None),
#                           np.concatenate((split_points-1, [len(test_data["selection"].values)-1]), axis=None)))
#
# limits = limits[np.where(np.asarray(list(map(np.sum, groups))) > 0)[0], :]
# limits = limits[np.where((limits < init_pos).sum(axis=1) == 0)[0], :]


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

x = test_data["Elongation"][init_pos]
y = test_data["Force"][init_pos]

ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
ax1.plot(x, y, 'ro')

x = test_data["Elongation"][break_point]
y = test_data["Force"][break_point]

ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
ax1.plot(x, y, 'ko')

for i in range(limits.shape[0]):
    ax1.plot(test_data["Elongation"][limits[i, 0]:limits[i, 1]],
             test_data["Force"][limits[i, 0]:limits[i, 1]], 'yo')


# ax2.plot(test_data['Elongation'], test_data['dF'], '-')
# ax3.plot(test_data['Elongation'], test_data['slopes'], '-')
plt.show()


# print(test_data.columns)
