from scipy import signal
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class mtr():
    """microtest tensile stage data analysis class"""

    @staticmethod
    def read_mtr(file_path):
        """Gets force and elongation from MTR file, a native data format of
           Deben Microtest miniature testing machine.

        Parameters
        ----------
        file_path : str
            The file location

        Returns
        -------
        (force, elongation)
            A touple of two Numpy representations of the DataFrame columns with
            force and elongation data.
        """
        header_length = 19
        test_data = pd.read_csv(file_path, sep=",", skiprows=header_length, decimal=".")
        test_data = test_data[['Elongation', 'Force']]
        test_data = test_data.groupby(['Elongation'])['Force'].mean().reset_index()
        return (test_data["Force"].values, test_data["Elongation"].values)

    @staticmethod
    def get_init_point(force, wnd_half_width=41):
        """Sets the starting point of tensile test.

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        wnd_half_width: float number
            Half size of algorithm search window.
        Returns
        -------
        init_pos
            A integer number indicating the vector index position of tensile
            test starting point.
        """
        force = force[0:force.argmax()]
        force = force - force.min()
        force = force / force.max()

        xx = np.linspace(0, 7, 8)
        yy = force[0:8]
        p = np.polyfit(xx, yy, 1)

        xx = np.linspace(-wnd_half_width, 0, wnd_half_width)
        yy = p[1]*xx + p[0]

        window = np.concatenate((np.ones(wnd_half_width)/wnd_half_width,
                                 np.zeros(wnd_half_width)), axis=None)
        force = np.concatenate((yy, force), axis=None)

        ey2_backward = signal.convolve(force**2, window, mode='valid')
        e2y_backward = signal.convolve(force, window, mode='valid')**2

        ey2_foreward = signal.convolve(force**2, np.flip(window, axis=0), mode='valid')
        e2y_foreward = signal.convolve(force, np.flip(window, axis=0), mode='valid')**2

        var_foreward = ey2_foreward - e2y_foreward
        var_backward = ey2_backward - e2y_backward

        var_foreward[var_foreward <= 0.0001] = 0.0001
        var_backward[var_backward <= 0.0001] = 0.0001

        var_change = var_backward / var_foreward
        init_pos = var_change.argmax() + np.floor(0.5*wnd_half_width)
        init_pos -= wnd_half_width

        if init_pos < 0:
            init_pos = 0

        return int(init_pos)

    @staticmethod
    def get_break_point(force, force_drop_thr=0.25, df_step=3):
        """Gets the specimen break point. Break point is idicated by sudden
           drop in force value.

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        force_drop_thr: float number from 0.0 to 1.0
            Threshold value of force drop above which the point is identified
            as the rupture point of the specimen. Express the percentage of
            maximal force during tensile test.

        df_step: positive integer number
            The size of the step between the measuring points between which
            the force drop is checked.

        Returns
        -------
        brea_point
            A integer number indicating the vector index position of tensile
            test break point.
        """
        force = signal.savgol_filter(force, 5, 3, deriv=0, delta=1.0, axis=- 1,
                                     mode='interp', cval=0.0)
        break_point = np.where((abs(np.diff(force, n=df_step, axis=0)) > (force_drop_thr*force.max())) == 1)[0]
        if len(break_point) > 0:
            return break_point[0]
        else:
            return -1

    @staticmethod
    def get_slopes_limits(force, elongation, init_pos, slope_thr=0.5, slope_wnd=30):
        """Gets linear sections of force-elongation curve.

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        elongation : vector which is numpy.ndarray
            Elongation data from tensile test.

        init_pos : positive integer
            A integer number indicating the vector index position of tensile
            test starting point.

        slope_thr: float number from 0.0 to 1.0
            Threshold value for "linearity" of force-elongation curve sections.

        slope_wnd: positive integer number
            The size of the algorithm search window.

        Returns
        -------
        limits
            N by 2 numpy.ndarray of integer numbers indicatin initial (first column)
            and final (second column) indexes of detected linear sections of
            force-elongation curve; N is the number of detected linear sections;
        """
        max_force_loc = force.argmax()
        force = force[0:max_force_loc]
        elongation = elongation[0:max_force_loc]

        sav_gol_wnd = np.floor(slope_wnd*0.8)
        if (sav_gol_wnd % 2 == 0):
            sav_gol_wnd += 1

        force_df = signal.savgol_filter(force, int(sav_gol_wnd), 1, deriv=1, delta=1.0, axis=- 1,
                                        mode='interp', cval=0.0)

        force_df = (force_df - force_df.min()) / (force_df.max() - force_df.min())
        elongation = elongation / elongation.max()

        half_window = slope_wnd
        slopes = np.zeros(len(force_df))

        for i in range(half_window, len(force_df) - half_window):
            a = np.polyfit(elongation[i-half_window:i+half_window],
                           force_df[i-half_window:i+half_window], 1)[0]
            slopes[i] = a

        slopes[range(len(slopes) - half_window, len(slopes))] = slopes[len(slopes) - (half_window+1)]
        slopes[range(0, half_window)] = slopes[half_window]

        selection = ((abs(slopes) < slope_thr*slopes.max()).astype(int))

        split_points = np.where(abs(np.diff(selection, n=1, axis=-1)) == 1)[0]+1
        groups = np.split(selection, split_points, axis=0)

        limits = np.column_stack((np.concatenate(([0], split_points), axis=None),
                                  np.concatenate((split_points-1, [len(selection)-1]), axis=None)))

        limits = limits[np.where(np.asarray(list(map(np.sum, groups))) > 0)[0], :]
        limits = limits[np.where((limits < init_pos).sum(axis=1) == 0)[0], :]

        section_lengths = limits[:, 1] - limits[:, 0]
        # print(section_lengths)

        limits = limits[section_lengths >= 10]
        return limits

    def get_slopes_values(force, elongation, init_pos, limits):
        """Gets slopes of linear sections of force-elongation curve.

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        elongation : vector which is numpy.ndarray
            Elongation data from tensile test.

        init_pos : positive integer
            A integer number indicating the vector index position of tensile
            test starting point.

        limits:
            N by 2 numpy.ndarray of integer numbers indicatin initial (first column)
            and final (second column) indexes of detected linear sections of
            force-elongation curve; N is the number of detected linear sections;

        Returns
        -------
        slopes_values
            list of floating point values indicating slopes of detected
            linear sections of force-elongation curve.
        """
        force = force - force[int(init_pos)]
        elongation = elongation - elongation[int(init_pos)]

        slopes_values = []
        for i in range(limits.shape[0]):
            a = np.polyfit(elongation[limits[i, 0]:limits[i, 1]],
                           force[limits[i, 0]:limits[i, 1]], 1)[0]
            slopes_values.append(a)

        return slopes_values

    def remove_spike(force):
        """Removes data spikes

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        Returns
        -------
        force
            Vector which is numpy.ndarray, with filtered force data from
            tensile test.
        """
        window = np.array([1.0, 0, -1.0])
        filtered = np.abs(signal.convolve(force, window, mode='same'))

        filtered[filtered < 3] = 0
        window = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        filtered = np.abs(signal.convolve(filtered, window, mode='same'))

        force[filtered > 0] = np.nan
        nans, x = np.isnan(force), lambda z: z.nonzero()[0]
        force[nans] = np.interp(x(nans), x(~nans), force[~nans])

        return force

    @staticmethod
    def process_tensile_test(force, elongation, preview=False):
        """Calculates the mechanical parametes from tensile test
            force-deformation curve

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        elongation : vector which is numpy.ndarray
            Elongation data from tensile test.

        -------
        dictionary
            dictionary holding the output parametes calculated from tensile
            test force-deformation curve
        """
        init_pos = mtr.get_init_point(force, wnd_half_width=41)
        break_point = mtr.get_break_point(force, force_drop_thr=0.25)
        limits = mtr.get_slopes_limits(force,
                                       elongation, init_pos,
                                       slope_thr=0.4,
                                       slope_wnd=40)

        values = mtr.get_slopes_values(force, elongation, init_pos, limits)

        if preview is True:
            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(elongation, force, 'b-')
            for i in range(limits.shape[0]):
                ax1.plot(elongation[limits[i, 0]:limits[i, 1]],
                         force[limits[i, 0]:limits[i, 1]], 'yo')

            x = elongation[init_pos]
            y = force[init_pos]
            ax1.plot(elongation, force, 'b-')
            ax1.plot(x, y, 'ro')

            x = elongation[break_point]
            y = force[break_point]
            ax1.plot(elongation, force, 'b-')
            ax1.plot(x, y, 'ko')

            plt.show()

        if values:
            if len(values) == 1:
                slope_1 = values[0]
                slope_2 = -1

                # slope_2.append(None)

            if len(values) == 2:
                slope_1 = values[0]
                slope_2 = values[1]

            result = {"elongation_at_start": elongation[init_pos],
                      "slope_1": slope_1,
                      "slope_2": slope_2,
                      "elongation_at_break": elongation[break_point],
                      "force_at_break": force[break_point],
                      "elastic_elongation_limit": elongation[limits[0, 1]],
                      "elastic_force_limit": force[limits[0, 1]],
                      "toughness": np.trapz(y=force, x=elongation)}
            return result
        else:
            result = {"elongation_at_start": elongation[init_pos],
                      "slope_1": -1,
                      "slope_2": -1,
                      "elongation_at_break": elongation[break_point],
                      "force_at_break": force[break_point],
                      "elastic_elongation_limit": -1,
                      "elastic_force_limit": -1,
                      "toughness": np.trapz(y=force, x=elongation)}
            return result

    @staticmethod
    def process_cyclic_test(force, elongation):
        """Calculates the mechanical parametes from cyclic test
            force-deformation curve

        Parameters
        ----------
        force : vector which is numpy.ndarray
            Force data from tensile test.

        elongation : vector which is numpy.ndarray
            Elongation data from tensile test.

        Returns
        -------
        results
            dictionary holding the output parametes calculated from cyclic
            test force-deformation curve
        """
        window = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        force = mtr.remove_spike(force)
        force = np.abs(signal.convolve(force, window, mode='same'))
        force = signal.savgol_filter(force, 31, 0, deriv=0, delta=1.0, axis=- 1,
                                     mode='interp', cval=0.0)

        # searching for positive and negative force peaks
        max_peaks, _ = find_peaks(force, distance=35, prominence=0.2)
        min_peaks, _ = find_peaks(np.abs(force-np.max(force)), distance=35, prominence=0.18)

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

        # label extension direction
        direction = np.zeros(len(force), np.int32)
        for index in range(len(force)):
            cycle = cycles[index]
            if index <= max_peaks[cycle]:
                direction[index] = 0
            else:
                direction[index] = 1
        direction[max_peaks[-1]:-1] = 1

        d = {'Force': force, 'Elongation': elongation}
        data = pd.DataFrame(data=d)
        data['Cycle'] = cycles
        data['Direction'] = direction
        data.loc[data['Direction'] == 0, ['Direction']] = "U"
        data.loc[data['Direction'] == 1, ['Direction']] = "D"

        results = []
        # analyse each cycle
        for cycle_id in (data['Cycle'].unique()):
            if cycle_id > 0:
                cycle_data = data.loc[data['Cycle'] == cycle_id, ]
                cycle_data = cycle_data.loc[cycle_data['Elongation'] > 0.0, ]
                cycle_data = cycle_data.loc[cycle_data['Elongation'].diff().fillna(0).abs() < 0.06, ]
                cycle_data = cycle_data.groupby(['Direction', 'Elongation'])[['Force']].mean().reset_index()
                # cycle_data = cycle_data.sort_values(by=['Time'], ascending=True)

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

                cycle_results = {"cycle": cycle_id,
                                 "modulus": a,
                                 "elongation_at_start": elongation_at_start,
                                 "elongation_at_max": elongation_at_max,
                                 "force_max": force_max,
                                 "elongation_at_end": elongation_at_end,
                                 "upward_area": upward_area,
                                 "downward_area": downward_area}
                results.append(cycle_results)
        return results
