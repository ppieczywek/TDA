# import pandas as pd

from mtr import mtr
import wx
import pandas as pd


class Example(wx.Dialog):
    window_size = 40
    init_window_size = 31
    threshold = 0.4

    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)

        self.InitUI()

    def InitUI(self):

        panel = wx.Panel(self)

        wnd_sizer = wx.BoxSizer(wx.HORIZONTAL)
        init_lbl = wx.StaticText(panel, label="Init window size:")
        wnd_sizer.Add(init_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.init_wnd = wx.TextCtrl(panel)
        self.init_wnd.SetValue(str(self.init_window_size))
        wnd_sizer.Add(self.init_wnd, 0, wx.ALL, 5)

        user_sizer = wx.BoxSizer(wx.HORIZONTAL)
        user_lbl = wx.StaticText(panel, label="Window size:")
        user_sizer.Add(user_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.user = wx.TextCtrl(panel)
        self.user.SetValue(str(self.window_size))
        user_sizer.Add(self.user, 0, wx.ALL, 5)

        # pass info
        p_sizer = wx.BoxSizer(wx.HORIZONTAL)

        p_lbl = wx.StaticText(panel, label="Threshold:")
        p_sizer.Add(p_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.password = wx.TextCtrl(panel)
        self.password.SetValue(str(self.threshold))
        p_sizer.Add(self.password, 0, wx.ALL, 5)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        accept_btn = wx.Button(panel, label="Accept")
        accept_btn.Bind(wx.EVT_BUTTON, self.onAccept)
        button_sizer.Add(accept_btn, 0, wx.ALL | wx.CENTER, 5)

        redo_btn = wx.Button(panel, label="Redo")
        button_sizer.Add(redo_btn, 0, wx.ALL | wx.CENTER, 5)
        redo_btn.Bind(wx.EVT_BUTTON, self.onRedo)

        skip_btn = wx.Button(panel, label="Skip")
        button_sizer.Add(skip_btn, 0, wx.ALL | wx.CENTER, 5)
        skip_btn.Bind(wx.EVT_BUTTON, self.onSkip)

        cancel_btn = wx.Button(panel, label="Cancel")
        button_sizer.Add(cancel_btn, 0, wx.ALL | wx.CENTER, 5)
        cancel_btn.Bind(wx.EVT_BUTTON, self.onCancel)

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        main_sizer.Add(wnd_sizer, 0, wx.ALL, 5)
        main_sizer.Add(user_sizer, 0, wx.ALL, 5)
        main_sizer.Add(p_sizer, 0, wx.ALL, 5)
        main_sizer.Add(button_sizer, 0, wx.ALL, 5)

        panel.SetSizer(main_sizer)

    def GetThreshold(self):
        return self.threshold

    def GetWindow(self):
        return self.window_size

    def GetInitWindow(self):
        return self.init_window_size

    def onAccept(self, event):
        self.EndModal(wx.ID_OK)

    def onSkip(self, event):
        self.EndModal(wx.ID_IGNORE)

    def onRedo(self, event):
        self.window_size = int(self.user.GetValue())
        self.init_window_size = int(self.init_wnd.GetValue())
        self.threshold = float(self.password.GetValue())

        self.EndModal(wx.ID_REDO)

    def onCancel(self, event):
        self.EndModal(wx.ID_CANCEL)


def main():

    app = wx.App()
    frame = wx.Frame(None, -1, 'MicrotestData')
    # frame.SetSize(0, 0, 200, 50)
    # Create open file dialog
    openFileDialog = wx.FileDialog(frame, "Open", "", "",
                                   "Select MTR output data files (*.mtr)|*.mtr",
                                   wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)
    openFileDialog.ShowModal()
    files = openFileDialog.GetPaths()
    openFileDialog.Destroy()

    ex = Example(None)
    data = []

    for file in files:

        stop = False
        ret = []

        while stop is False:
            window = ex.GetWindow()
            window2 = ex.GetInitWindow()
            threshold = ex.GetThreshold()

            print(window)
            print(threshold)
            tt = mtr.process_mtr_file(file, init_wnd=window2, thr=threshold, wnd=window, preview=True)
            ret = ex.ShowModal()

            if ret == wx.ID_OK:
                data.append(tt)
                stop = True

            elif ret == wx.ID_CANCEL:
                stop = True

            elif ret == wx.ID_REDO:
                pass

            elif ret == wx.ID_IGNORE:
                stop = True

            else:
                stop = True

        if ret == wx.ID_CANCEL:

            break

    ex.Destroy()
    if len(data) > 0:
        data_frame = pd.DataFrame(data)
        data_frame.to_csv('tough_2.csv', index=False)


if __name__ == '__main__':
    main()
    input("Press Enter to continue...")

# app = wx.App()

#
# for file in files:
#     tt = mtr.process_mtr_file(file, preview=True)
#
#
# #tt.to_csv('out.csv', index=False)
# print(tt)
#
