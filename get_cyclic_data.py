from mtr import mtr
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import wx
from scipy import signal
# import matplotlib.pyplot as plt


results = []

pp = wx.App()
frame = wx.Frame(None, -1, 'RheoData.py')
frame.SetSize(0, 0, 200, 50)
window = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Open", "", "",
                               "Deben microstester data files (*.mtr)|*.mtr",
                               wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)  #

openFileDialog.ShowModal()
file_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

files_to_open = len(file_list)

if __name__ == '__main__':

    for count, file in enumerate(file_list):
        data_file = file.split('\\')[-1]
        print(f"Processing data from file: {data_file}")

        mtr_header_length = 19
        data = pd.read_csv(file, sep=",", skiprows=mtr_header_length, encoding='unicode_escape', decimal=".")
        data = data[['Elongation', 'Force', 'Samplerate']]
        force = data["Force"].values
        force = mtr.remove_spike(force)\

        # convolution filter to smooth data
        force = np.abs(signal.convolve(force, window, mode='same'))

        # calculating time scale
        time = np.cumsum(data["Samplerate"].values) / 1000.0 / 60.0

        # searching for positive and negative force peaks
        max_peaks, _ = find_peaks(force, distance=30, prominence=0.4)
        min_peaks, _ = find_peaks(np.abs(force-np.max(force)), distance=30, prominence=0.4)
        # looking for the last positive peak
        if (min_peaks[-1] > max_peaks[-1]):
            max_pos = np.argmax(force[min_peaks[-1]: (len(force)-1)])
            max_peaks = np.append(max_peaks, max_pos+min_peaks[-1])

        # validate peaks
        valid_peaks = []
        for index in range(1, len(max_peaks)):
            current_max = max_peaks[index]
            peaks = min_peaks[min_peaks < current_max]
            min_index = np.argmin(np.abs(peaks - current_max))
            valid_peaks.append(peaks[min_index])
        min_peaks = valid_peaks

        # label extension cycles
        cycles = np.zeros(len(force), np.int32)
        for cycle in range(len(min_peaks)):
            if cycle == 0:
                cycles[0:min_peaks[0]] = cycle
            else:
                cycles[min_peaks[cycle-1]:min_peaks[cycle]] = cycle
        cycles[min_peaks[-1]:len(cycles)] = len(min_peaks)

        # plt.plot(time, force, 'b-')
        # plt.plot(time[max_peaks], force[max_peaks], 'ro')
        # plt.plot(time[min_peaks], force[min_peaks], 'go')
        # plt.plot(time, cycles, 'b-')
        # plt.show()

        # label extension direction
        direction = np.zeros(len(force), np.int32)
        for index in range(len(force)):
            cycle = cycles[index]
            if index <= max_peaks[cycle]:
                direction[index] = 0
            else:
                direction[index] = 1
        direction[max_peaks[-1]:-1] = 1

        # build data frame
        data['Time'] = np.cumsum(data["Samplerate"].values) / 1000.0 / 60.0
        data['Cycle'] = cycles
        data['Direction'] = direction
        data.loc[data['Direction'] == 0, ['Direction']] = "U"
        data.loc[data['Direction'] == 1, ['Direction']] = "D"

        # analyse each cycle
        for cycle_id in (data['Cycle'].unique()):
            if cycle_id > 0:
                cycle_data = data.loc[data['Cycle'] == cycle_id, ]
                cycle_data = cycle_data.loc[cycle_data['Elongation'] > 0.0, ]
                cycle_data = cycle_data.loc[cycle_data['Elongation'].diff().fillna(0).abs() < 0.06, ]
                cycle_data = cycle_data.groupby(['Direction', 'Elongation'])[['Force', 'Time']].mean().reset_index()
                cycle_data = cycle_data.sort_values(by=['Time'], ascending=True)

                up_cycle_data = cycle_data.loc[cycle_data['Direction'] == "U", ]
                rows = up_cycle_data.shape[0]
                middle_row = rows//2
                fit_span = int(np.ceil(middle_row * 0.6))

                a = np.polyfit(up_cycle_data['Elongation'].values[(middle_row-fit_span):(middle_row+fit_span)],
                               up_cycle_data['Force'].values[(middle_row-fit_span):(middle_row+fit_span)], 1)[0]

                mean_step = (up_cycle_data['Elongation'].diff().mean())
                upward_area = np.sum(up_cycle_data['Force'].values * mean_step)

                elongation_at_start = up_cycle_data.loc[up_cycle_data['Force'].abs().idxmin(), 'Elongation']
                elongation_at_max = cycle_data.loc[cycle_data['Force'].idxmax(), 'Elongation']
                force_max = cycle_data['Force'].max()

                down_cycle_data = cycle_data.loc[cycle_data['Direction'] == "D", ]
                elongation_at_end = down_cycle_data.loc[down_cycle_data['Force'].abs().idxmin(), 'Elongation']

                mean_step = (down_cycle_data['Elongation'].diff().abs().mean())
                downward_area = np.sum(down_cycle_data['Force'].values * mean_step)

                cycle_results = {"file_name": data_file,
                                 "cycle": cycle_id,
                                 "modulus": a,
                                 "elongation_at_start": elongation_at_start,
                                 "elongation_at_max": elongation_at_max,
                                 "force_max": force_max,
                                 "elongation_at_end": elongation_at_end,
                                 "upward_area": upward_area,
                                 "downward_area": downward_area}
                results.append(cycle_results)

if len(results) > 0:
    outpu_file_name = "test_data.csv"
    export_data = pd.DataFrame(results)
    export_data.to_csv(outpu_file_name, index=False, line_terminator='\n')
