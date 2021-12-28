# import pandas as pd

from mtr import mtr
import wx

app = wx.App()
frame = wx.Frame(None, -1, 'MicrotestData')
frame.SetSize(0, 0, 200, 50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Open", "", "",
                               "Select MTR output data files (*.mtr)|*.mtr",
                               wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

openFileDialog.ShowModal()
files = openFileDialog.GetPaths()
openFileDialog.Destroy()

tt = mtr.process_mtr_file(files, preview=True)
tt.to_csv('out.csv', index=False)
print(tt)
input("Press Enter to continue...")
# file_name = "./DATA/NANOCEL_KW.GAL_900PPM_4.MTR"
# header_length = 19
# test_data = pd.read_csv(file_name, sep=",", skiprows=header_length, decimal=".")
# test_data = test_data[['Elongation', 'Force']]
# test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()
#
#
# init_pos = mtr.get_init_point(test_data["Force"].values, wnd_half_width=41)
# break_point = mtr.get_break_point(test_data["Force"].values, force_drop_thr=0.25)
# limits = mtr.get_slopes_limits(test_data['Force'].values,
#                                test_data['Elongation'].values, init_pos,
#                                slope_thr=0.4,
#                                slope_wnd=40)
# values = mtr.get_slopes_values(test_data['Force'].values,
#                                test_data['Elongation'].values,
#                                init_pos,
#                                limits)
# #
# flt = mtr.remove_spike(test_data["Force"].values)
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#
# x = test_data["Elongation"][init_pos]
# y = test_data["Force"][init_pos]
#
# ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
# ax1.plot(x, y, 'ro')
#
# x = test_data["Elongation"][break_point]
# y = test_data["Force"][break_point]
#
# ax1.plot(test_data["Elongation"], test_data["Force"], 'b-')
# ax1.plot(x, y, 'ko')
#
# ax2.plot(test_data["Elongation"], flt, 'b-')
# ax2.plot(x, y, 'ko')
#
#
# for i in range(limits.shape[0]):
#     ax1.plot(test_data["Elongation"][limits[i, 0]:limits[i, 1]],
#              test_data["Force"][limits[i, 0]:limits[i, 1]], 'yo')
#
#
# plt.show()
#
#
# print(test_data.columns)
