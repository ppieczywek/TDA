from scipy import signal
import numpy as np


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
