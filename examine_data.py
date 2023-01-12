# import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from mtr import mtr
import wx
import pandas as pd
import matplotlib
import numpy as np
# import matplotlib.pyplot as plt

matplotlib.use('WXAgg')


class Example(wx.Frame):
    window_size = 40
    init_window_size = 31
    threshold = 0.4
    files = []
    current_record_idx = 0
    current_record = None
    data = []

    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)

        self.InitUI()
        self.Centre()
        self.Show()

    def InitUI(self):

        panel = wx.Panel(self)
        self.SetTitle('yourtitle')
        sizer = wx.GridBagSizer(0, 0)

        init_lbl = wx.StaticText(panel, label="Init point search window size:")
        sizer.Add(init_lbl, pos=(0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=5)

        self.initpoint_slider = wx.Slider(panel, value=31, minValue=1, maxValue=100,
                                          style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.initpoint_slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        sizer.Add(self.initpoint_slider, pos=(1, 0), span=(1, 2), flag=wx.EXPAND, border=10)

        section_label = wx.StaticText(panel, label="Linear section search window:")
        sizer.Add(section_label, pos=(2, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=5)
        self.section_slider = wx.Slider(panel, value=31, minValue=1, maxValue=100,
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.section_slider.Bind(wx.EVT_SLIDER, self.OnSectionScroll)
        sizer.Add(self.section_slider, pos=(3, 0), span=(1, 2), flag=wx.EXPAND, border=10)

        threshold_label = wx.StaticText(panel, label="Section search threshold:")
        sizer.Add(threshold_label, pos=(4, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=5)
        self.threshold_slider = wx.Slider(panel, value=31, minValue=1, maxValue=100,
                                          style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.threshold_slider.Bind(wx.EVT_SLIDER, self.OnThresholdScroll)
        sizer.Add(self.threshold_slider, pos=(5, 0), span=(1, 2), flag=wx.EXPAND, border=10)

        # self.thr = wx.TextCtrl(panel)
        # self.thr.SetValue(str(self.threshold))
        # sizer.Add(self.thr, pos=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        accept_btn = wx.Button(panel, label="Accept")
        accept_btn.Bind(wx.EVT_BUTTON, self.onAccept)
        sizer.Add(accept_btn, pos=(6, 0), span=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        # redo_btn = wx.Button(panel, label="Redo")
        # redo_btn.Bind(wx.EVT_BUTTON, self.onRedo)
        # sizer.Add(redo_btn, pos=(3, 1), flag=wx.EXPAND | wx.ALL, border=5)

        skip_btn = wx.Button(panel, label="Skip")
        skip_btn.Bind(wx.EVT_BUTTON, self.onSkip)
        sizer.Add(skip_btn, pos=(6, 1), flag=wx.EXPAND | wx.ALL, border=5)

        cancel_btn = wx.Button(panel, label="Cancel")
        cancel_btn.Bind(wx.EVT_BUTTON, self.onCancel)
        sizer.Add(cancel_btn, pos=(7, 1), flag=wx.EXPAND | wx.ALL, border=5)

        select_btn = wx.Button(panel, label="Select data", size=(80, 30))
        select_btn.Bind(wx.EVT_BUTTON, self.onSelect)
        sizer.Add(select_btn, pos=(8, 0), flag=wx.EXPAND | wx.ALL, border=5)

        export_btn = wx.Button(panel, label="Export data", size=(80, 30))
        export_btn.Bind(wx.EVT_BUTTON, self.onExport)
        sizer.Add(export_btn, pos=(8, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.current_file_name = wx.StaticText(panel, label="Current file:")
        sizer.Add(self.current_file_name, pos=(9, 0), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.progress_info = wx.StaticText(panel, label="Processed:")
        sizer.Add(self.progress_info, pos=(10, 0), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        # INFO
        self.elongation_at_start_info = wx.StaticText(panel, label="El. at start:")
        sizer.Add(self.elongation_at_start_info, pos=(0, 4), flag=wx.EXPAND | wx.ALL, border=5)
        self.elongation_at_max_force_info = wx.StaticText(panel, label="El. at max force:")
        sizer.Add(self.elongation_at_max_force_info, pos=(1, 4), flag=wx.EXPAND | wx.ALL, border=5)

        self.slope1_info = wx.StaticText(panel, label="Slope 1:")
        sizer.Add(self.slope1_info, pos=(2, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.slope2_info = wx.StaticText(panel, label="Slope 2:")
        sizer.Add(self.slope2_info, pos=(3, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.max_force_info = wx.StaticText(panel, label="Max force:")
        sizer.Add(self.max_force_info, pos=(4, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.force_at_break_info = wx.StaticText(panel, label="Force at break:")
        sizer.Add(self.force_at_break_info, pos=(5, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.elongation_at_break_info = wx.StaticText(panel, label="Elongation at break:")
        sizer.Add(self.elongation_at_break_info, pos=(6, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.elastic_force_limit_info = wx.StaticText(panel, label="Elastic force limit:")
        sizer.Add(self.elastic_force_limit_info, pos=(7, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.elastic_elongation_limit_info = wx.StaticText(panel, label="Elastic elongation lim.:")
        sizer.Add(self.elastic_elongation_limit_info, pos=(8, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.toughness_info = wx.StaticText(panel, label="Toughness:")
        sizer.Add(self.toughness_info, pos=(9, 4), span=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.figure = Figure()
        self.figure.patch.set_facecolor((.94, .94, .94))
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        sizer.Add(self.canvas, pos=(0, 3), span=(11, 1), flag=wx.LEFT | wx.TOP | wx.GROW, border=5)

        self.border = wx.BoxSizer()
        self.border.Add(sizer, 1, wx.ALL | wx.EXPAND, 20)

        panel.SetSizerAndFit(self.border)
        self.Fit()

    def GetThreshold(self):
        return self.threshold

    def GetWindow(self):
        return self.window_size

    def GetInitWindow(self):
        return self.init_window_size

    def onAccept(self, event):
        if not self.current_record:
            wx.MessageDialog(self, 'No data to save', 'Test', wx.OK | wx.ICON_INFORMATION).ShowModal()
        else:

            if len(self.data) != (self.current_record_idx+1):
                self.data.append(self.current_record)

            if (self.current_record_idx+1) < len(self.files):
                # print(self.current_record)
                # print(self.data)
                self.current_record_idx += 1
                self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                            init_wnd=self.init_window_size,
                                                            thr=self.threshold,
                                                            wnd=self.window_size,
                                                            preview=True)
                self.canvas.draw()

                file_name = "Current file:" + self.files[self.current_record_idx].split('\\')[-1]
                self.current_file_name.SetLabel(file_name)
                self.progress_info.SetLabel(f'Processing: {self.current_record_idx+1} out of {len(self.files)}')

            else:
                wx.MessageDialog(self, 'End of file list reached', 'Test', wx.OK | wx.ICON_INFORMATION).ShowModal()

    def onSkip(self, event):
        if (self.current_record_idx+1) < len(self.files):
            self.current_record_idx += 1
            self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                        init_wnd=self.init_window_size,
                                                        thr=self.threshold,
                                                        wnd=self.window_size,
                                                        preview=True)
            self.canvas.draw()

            file_name = "Current file:" + self.files[self.current_record_idx].split('\\')[-1]
            self.current_file_name.SetLabel(file_name)
            self.progress_info.SetLabel(f'Processing: {self.current_record_idx+1} out of {len(self.files)}')
        else:
            wx.MessageDialog(self, 'End of file list reached', 'Test', wx.OK | wx.ICON_INFORMATION).ShowModal()

    def onSelect(self, event):
        with wx.FileDialog(self, "Open", "", "",
                           "Select MTR output data files (*.mtr)|*.mtr",
                           wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as openFileDialog:

            if openFileDialog.ShowModal() == wx.ID_CANCEL:
                return

            self.files = openFileDialog.GetPaths()
            # openFileDialog.Destroy()
            self.current_record_idx = 0
            self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                        init_wnd=self.init_window_size,
                                                        thr=self.threshold,
                                                        wnd=self.window_size,
                                                        preview=True)
            self.canvas.draw()

            file_name = "Current file:" + self.files[self.current_record_idx].split('\\')[-1]
            self.current_file_name.SetLabel(file_name)
            self.progress_info.SetLabel(f'Processing: {self.current_record_idx+1} out of {len(self.files)}')

    # def onRedo(self, event):
    #     self.window_size = int(self.wnd_size.GetValue())
    #     self.init_window_size = int(self.init_wnd.GetValue())
    #     self.threshold = float(self.thr.GetValue())
    #     self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
    #                                                 init_wnd=self.init_window_size,
    #                                                 thr=self.threshold,
    #                                                 wnd=self.window_size,
    #                                                 preview=True)
    #     self.canvas.draw()

    def OnSliderScroll(self, event):
        if self.files:
            obj = event.GetEventObject()
            self.init_window_size = obj.GetValue()
            self.window_size = self.section_slider.GetValue()
            self.threshold = self.threshold_slider.GetValue() / 100
            self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                        init_wnd=self.init_window_size,
                                                        thr=self.threshold,
                                                        wnd=self.window_size,
                                                        preview=True)
            self.canvas.draw()

    def OnSectionScroll(self, event):
        if self.files:
            obj = event.GetEventObject()
            self.window_size = obj.GetValue()
            self.init_window_size = self.initpoint_slider.GetValue()
            self.threshold = self.threshold_slider.GetValue() / 100
            self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                        init_wnd=self.init_window_size,
                                                        thr=self.threshold,
                                                        wnd=self.window_size,
                                                        preview=True)
            self.canvas.draw()

    def OnThresholdScroll(self, event):
        if self.files:
            obj = event.GetEventObject()
            self.threshold = obj.GetValue() / 100
            self.window_size = self.section_slider.GetValue()
            self.init_window_size = self.initpoint_slider.GetValue()
            self.current_record = self.process_mtr_file(self.files[self.current_record_idx],
                                                        init_wnd=self.init_window_size,
                                                        thr=self.threshold,
                                                        wnd=self.window_size,
                                                        preview=True)
            self.canvas.draw()

    def onCancel(self, event):
        self.Destroy()

    def onExport(self, event):
        with wx.FileDialog(self, "Save csv file", wildcard="CSV files (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'w') as file:
                    if len(self.data) > 0:
                        export_data = pd.DataFrame(self.data)
                        print(export_data)
                        export_data.to_csv(file, index=False, line_terminator='\n')
                    # self.doSaveData(file)
            except IOError:
                wx.LogError("Cannot save current data in file '%s'." % pathname)

    def process_mtr_file(self, data_file, init_wnd=30, thr=0.50, wnd=30, preview=False):
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
        # encoding='unicode_escape',
        file_name = data_file.split('\\')[-1]
        data = pd.read_csv(data_file, sep=",", skiprows=mtr_header_length,  decimal=".")
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

            self.elongation_at_start_info.SetLabel(f'El. at start: {elongation_at_start:.2f}')
            self.elongation_at_max_force_info.SetLabel(f'El. at max force: {elongation_at_max_force:.2f}')

            self.slope1_info.SetLabel(f'Slope 1: {slope_1:.2f}')
            self.slope2_info.SetLabel(f'Slope 2: {slope_2:.2f}')

            self.max_force_info.SetLabel(f'Max force: {max_force:.2f}')
            self.force_at_break_info.SetLabel(f'Force at break: {force_at_break:.2f}')

            self.elongation_at_break_info.SetLabel(f'Elongation at break: {elongation_at_break:.2f}')
            self.elastic_force_limit_info.SetLabel(f'El. force linit: {elastic_force_limit:.2f}')

            self.elastic_elongation_limit_info.SetLabel(f'Elastic elongation lim.: {elastic_elongation_limit:.2f}')
            self.toughness_info.SetLabel(f'Toughness: {toughness:.2f}')

            # print(res)

        self.axes.clear()

        if preview is True:

            if init_pos >= 0:
                x = data["Elongation"][init_pos]
                y = data["Force"][init_pos]
                self.axes.plot(x, y, 'ro')

            self.axes.plot(data["Elongation"], data["Force"], 'b-')
            if break_point > -1:
                x = data["Elongation"][break_point]
                y = data["Force"][break_point]
                self.axes.plot(x, y, 'ko')

            if limits is not None:
                for i in range(limits.shape[0]):
                    self.axes.plot(data["Elongation"][limits[i, 0]:limits[i, 1]],
                                   data["Force"][limits[i, 0]:limits[i, 1]], 'yo')

            # plt.show()

        return res


if __name__ == '__main__':
    app = wx.App()
    ex = Example(None)
    app.MainLoop()

    # input("Press Enter to continue...")
