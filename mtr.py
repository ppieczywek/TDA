from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class mtr():
    """microtest tensile stage data analysis class"""

    @staticmethod
    def get_init_point(force, wnd_half_width=41):

        force = force[0:force.argmax()]
        force = force - force.min()
        force = force / force.max()

        window = np.concatenate((np.ones(wnd_half_width)/wnd_half_width,
                                 np.zeros(wnd_half_width)), axis=None)

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

        return init_pos

    @staticmethod
    def get_break_point(force, force_drop_thr=0.25, df_step=3):
        force = signal.savgol_filter(force, 5, 3, deriv=0, delta=1.0, axis=- 1,
                                     mode='interp', cval=0.0)
        break_point = np.where((abs(np.diff(force, n=df_step, axis=0)) > (force_drop_thr*force.max())) == 1)[0]
        if len(break_point) > 0:
            return break_point[0]
        else:
            return -1

    @staticmethod
    def get_slopes_limits(force, displacement, init_pos, slope_thr=0.5, slope_wnd=30):

        max_force_loc = force.argmax()
        force = force[0:max_force_loc]
        displacement = displacement[0:max_force_loc]
        #
        # force_df = signal.savgol_filter(force, 25, 1, deriv=1, delta=1.0, axis=- 1,
        #                                 mode='interp', cval=0.0)
        sav_gol_wnd = np.floor(slope_wnd*0.8)
        if (sav_gol_wnd % 2 == 0):
            sav_gol_wnd += 1

        force_df = signal.savgol_filter(force, int(sav_gol_wnd), 1, deriv=1, delta=1.0, axis=- 1,
                                        mode='interp', cval=0.0)

        force_df = (force_df - force_df.min()) / (force_df.max() - force_df.min())
        displacement = displacement / displacement.max()

        half_window = slope_wnd
        slopes = np.zeros(len(force_df))

        for i in range(half_window, len(force_df) - half_window):
            a = np.polyfit(displacement[i-half_window:i+half_window],
                           force_df[i-half_window:i+half_window], 1)[0]
            slopes[i] = a

        # sprint(slopes)

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

    def get_slopes_values(force, displacement, init_pos, limits):
        force = force - force[int(init_pos)]
        displacement = displacement - displacement[int(init_pos)]

        slopes_values = []
        for i in range(limits.shape[0]):
            a = np.polyfit(displacement[limits[i, 0]:limits[i, 1]],
                           force[limits[i, 0]:limits[i, 1]], 1)[0]
            slopes_values.append(a)

        return slopes_values

    def remove_spike(force):
        window = np.array([1.0, -1])
        filtered = signal.convolve(force, window, mode='same')
        filtered[filtered < 5] = 0
        smooth = signal.savgol_filter(force, 35, 1, deriv=0, delta=1.0, axis=- 1,
                                      mode='interp', cval=0.0)
        force[filtered > 0] = smooth[filtered > 0]
        return force

    def process_mtr_file(data_file, init_wnd=30, thr=0.50, wnd=30, preview=False):
        mtr_header_length = 19
        max_force = -1
        force_at_break = -1
        init_pos = -1
        break_point = -1
        elongation_at_start = -1
        elongation_at_max_force = -1
        elongation_at_break = -1
        elastic_elongation_limit = -1
        elastic_force_limit = -1
        toughness = -1
        slope_1 = -1
        slope_2 = -1
        limits = None

        file_name = data_file.split('\\')[-1]
        data = pd.read_csv(data_file, sep=",", skiprows=mtr_header_length, decimal=".")
        data = data[['Elongation', 'Force']]
        data = data.groupby(['Elongation'])['Force'].mean().reset_index()
        force = data["Force"].values

        print('Processing data from file: ' + file_name)

        if (len(force) <= (3*wnd)+1):

            print('Error: the file contains too little data for analysis.')
            res = {"file_name": file_name,
                   "elongation_at_start": elongation_at_start,
                   "slope_1": slope_1,
                   "slope_2": slope_2,
                   "elongation_at_max_force": elongation_at_max_force,
                   "max_force": max_force,
                   "elongation_at_break": elongation_at_break,
                   "force_at_break": force_at_break,
                   "elastic_elongation_limit": elastic_elongation_limit,
                   "elastic_force_limit": elastic_force_limit,
                   "toughness": toughness}
            print(res)
        else:
            force = mtr.remove_spike(force)
            elongation = data['Elongation'].values

            init_pos = mtr.get_init_point(force, wnd_half_width=init_wnd)
            break_point = mtr.get_break_point(force, force_drop_thr=0.25, df_step=3)
            limits = mtr.get_slopes_limits(force, elongation, init_pos,
                                           slope_thr=thr, slope_wnd=wnd)
            values = mtr.get_slopes_values(force, elongation, init_pos, limits)

            if values:
                if len(values) == 1:
                    slope_1 = values[0]

                    # slope_2.append(None)

                if len(values) == 2:
                    slope_1 = values[0]
                    slope_2 = values[1]

                elongation_at_start = elongation[int(init_pos)]
                force -= force[int(init_pos)]
                elongation -= elongation[int(init_pos)]
                max_force = np.max(force)
                elongation_at_max_force = elongation[np.argmax(force)]

                # if (limits[0, 1] - int(init_pos)) > 0:
                elastic_elongation_limit = elongation[limits[0, 1]]
                elastic_force_limit = force[limits[0, 1]]

                toughness = np.trapz(y=force, x=elongation)

            if break_point > -1:
                force_at_break = (force[int(break_point)])
                elongation_at_break = (elongation[int(break_point)])

            res = {"file_name": file_name,
                   "elongation_at_start": elongation_at_start,
                   "slope_1": slope_1,
                   "slope_2": slope_2,
                   "elongation_at_max_force": elongation_at_max_force,
                   "max_force": max_force,
                   "elongation_at_break": elongation_at_break,
                   "force_at_break": force_at_break,
                   "elastic_elongation_limit": elastic_elongation_limit,
                   "elastic_force_limit": elastic_force_limit,
                   "toughness": toughness}
            print(res)

        if preview is True:
            # pass
            fig, ax1 = plt.subplots(1, 1)
            # plt.show(block=False)
            #
            if init_pos >= 0:
                x = data["Elongation"][init_pos]
                y = data["Force"][init_pos]
                ax1.plot(x, y, 'ro')

            ax1.plot(data["Elongation"], data["Force"], 'b-')
            if break_point > -1:
                x = data["Elongation"][break_point]
                y = data["Force"][break_point]
                ax1.plot(x, y, 'ko')

            if limits is not None:
                for i in range(limits.shape[0]):
                    ax1.plot(data["Elongation"][limits[i, 0]:limits[i, 1]],
                             data["Force"][limits[i, 0]:limits[i, 1]], 'yo')

            plt.show()

        return res
