from scipy import signal
import numpy as np
import pandas as pd


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
    def get_break_point(force, force_drop_thr=0.25):
        force = signal.savgol_filter(force, 5, 3, deriv=0, delta=1.0, axis=- 1,
                                     mode='interp', cval=0.0)
        break_point = np.where((abs(np.diff(force, n=1, axis=0)) > (0.25*force.max())) == 1)[0]
        if len(break_point) > 0:
            return break_point[0]
        else:
            return -1

    @staticmethod
    def get_slopes_limits(force, displacement, init_pos):

        max_force_loc = force.argmax()
        force = force[0:max_force_loc]
        displacement = displacement[0:max_force_loc]

        force_df = signal.savgol_filter(force, 25, 1, deriv=1, delta=1.0, axis=- 1,
                                        mode='interp', cval=0.0)

        force_df = (force_df - force_df.min()) / (force_df.max() - force_df.min())
        displacement = displacement / displacement.max()

        half_window = 30
        slopes = np.zeros(len(force_df))

        for i in range(half_window, len(force_df) - half_window):
            a = np.polyfit(displacement[i-half_window:i+half_window],
                           force_df[i-half_window:i+half_window], 1)[0]
            slopes[i] = a

        slopes[range(len(slopes) - half_window, len(slopes))] = slopes[len(slopes) - (half_window+1)]
        slopes[range(0, half_window)] = slopes[half_window]

        selection = ((abs(slopes) < 0.5*slopes.max()).astype(int))

        split_points = np.where(abs(np.diff(selection, n=1, axis=-1)) == 1)[0]+1
        groups = np.split(selection, split_points, axis=0)

        limits = np.column_stack((np.concatenate(([0], split_points), axis=None),
                                  np.concatenate((split_points-1, [len(selection)-1]), axis=None)))

        limits = limits[np.where(np.asarray(list(map(np.sum, groups))) > 0)[0], :]
        limits = limits[np.where((limits < init_pos).sum(axis=1) == 0)[0], :]
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

    def process_mtr_file(data_files):
        mtr_header_length = 19

        file_names = []
        max_force = []
        force_at_break = []
        elongation_at_start = []
        elongation_at_max_force = []
        elongation_at_break = []
        slope_1 = []
        slope_2 = []

        for data_file in data_files:

            file_name = data_file.split('\\')[-1]
            file_names.append(file_name)
            data = pd.read_csv(data_file, sep=",", skiprows=mtr_header_length, decimal=".")
            data = data[['Elongation', 'Force']]
            data = data.groupby(['Elongation'])['Force'].mean().reset_index()
            force = data["Force"].values

            force = mtr.remove_spike(force)
            elongation = data['Elongation'].values

            init_pos = mtr.get_init_point(force)
            break_point = mtr.get_break_point(force)

            limits = mtr.get_slopes_limits(force, elongation, init_pos)
            values = mtr.get_slopes_values(force, elongation, init_pos, limits)

            if values:
                if len(values) == 1:
                    slope_1.append(values[0])
                    slope_2.append(None)

                if len(values) == 2:
                    slope_1.append(values[0])
                    slope_2.append(values[1])

            elongation_at_start.append(elongation[int(init_pos)])

            force -= force[int(init_pos)]
            elongation -= elongation[int(init_pos)]

            max_force.append(np.max(force))
            elongation_at_max_force.append(elongation[np.argmax(force)])

            if break_point > -1:
                force_at_break.append(force[int(break_point)])
                elongation_at_break.append(elongation[int(break_point)])
            else:
                force_at_break.append(None)
                elongation_at_break.append(None)

        df = pd.DataFrame(list(zip(file_names,
                                   elongation_at_start,
                                   slope_1,
                                   slope_2,
                                   elongation_at_max_force,
                                   max_force,
                                   elongation_at_break,
                                   force_at_break)),
                          columns=['file_name',
                                   'elongation_at_start',
                                   'slope_1',
                                   'slope_2',
                                   'elongation_at_max_force',
                                   'max_force',
                                   'elongation_at_break',
                                   'force_at_break'])
        return df
