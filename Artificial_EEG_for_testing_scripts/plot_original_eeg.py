import numpy as np
import random

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import filedialog

class eeg_plotter():

    def __init__(self):
        self.labels_ave = ['Fp1','F7','T3','T5','Fp2','F8','T4','T6','F3','C3','P3','O1','F4','C4','P4','O2','A1','A2','FZ','CZ','PZ']
        self.out_ch = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20]
        self.index = 0
        self.amp = 1
        self.sampling_frequency = 256
    
    def open_subject(self):
        self.directory = filedialog.askdirectory()
        self.index = 0
        self.eeg = np.load(self.directory + '/eeg_0.npy', allow_pickle = True)
        self.plot_eeg()
        self.count_var.set(self.index + 1)

    def load_new(self):
        try:
            self.eeg = np.load(self.directory + '/eeg_' + str(self.index) + '.npy', allow_pickle = True)
        except:
            print('--- End of EEG ---')

    def plot_eeg(self):
        self.eeg_plot.ax.cla()
        spacing = 50
        t = np.arange(self.sampling_frequency*10)
        for idx in range(21):
            self.eeg_plot.ax.plot(t/self.sampling_frequency,self.amp*self.eeg[idx]+spacing*idx,color='black',linewidth=0.5)
        self.eeg_plot.ax.set_xticks(np.arange(0,np.round(len(t)/self.sampling_frequency)+1,1))
        self.eeg_plot.ax.set_yticks(np.arange(0,spacing*21,spacing))
        self.eeg_plot.ax.set_yticklabels(self.labels_ave)
        self.eeg_plot.ax.set_ylim(-spacing,21*spacing)
        self.eeg_plot.ax.invert_yaxis()
        self.eeg_plot.canvas.draw()

    def next(self):
        self.index += 1
        self.load_new()
        self.plot_eeg()
        self.count_var.set(self.index + 1)

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.load_new()
            self.plot_eeg()
            self.count_var.set(self.index + 1)

    def amp_up(self):
        self.amp += 0.2
        self.plot_eeg()
    
    def amp_down(self):
        if self.amp > 0.3:
            self.amp -= 0.2
            self.plot_eeg()

class figureFrame(object):

    def __init__(self,frame,figure_size,side,hide_axes,color):

        self.fig = Figure(figsize=figure_size, dpi=100,tight_layout=True,facecolor=color)
        self.ax = self.fig.add_subplot(111)
        if hide_axes:
            self.ax.tick_params(right=False,left=False,top=False,bottom=False,labelleft=False,labelbottom=False)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_visible(False)
        self.canvas = FigureCanvasTkAgg(self.fig,frame)
        self.canvas.draw()
        self.canvas._tkcanvas.pack(side=side,fill=BOTH,padx=5,pady=5)

# ***** Init Generator *****
G = eeg_plotter()
# ***** Init GUI *****
W = 15
color = '#dadada'   # Use '#eaeaea' for Mac
G.root = Tk()
G.root.configure(background=color)
G.root.title('EEG viewer')
mainFrame = Frame(G.root,background=color)
mainFrame.pack(fill=BOTH)
topFrame = Frame(mainFrame,background=color)
topFrame.pack(fill=BOTH,padx=5,pady=5)
bottomFrame = Frame(mainFrame,background=color)
bottomFrame.pack(side=BOTTOM,padx=5,pady=5)

# ***** EEG *****
G.eeg_plot = figureFrame(topFrame,(14,8),'left',False,color)

# ***** Settings *****

Button(bottomFrame,text='Quit',width=W,command=G.root.quit,background=color,highlightbackground=color).grid(row=0,column=0,padx=5,pady=5)
Button(bottomFrame,text='Previous',width=W,command=G.previous,background=color,highlightbackground=color).grid(row=0,column=1,padx=5,pady=5)
Button(bottomFrame,text='Next',width=W,command=G.next,background=color,highlightbackground=color).grid(row=0,column=2,padx=5,pady=5)
Button(bottomFrame,text='-',width=W,command=G.amp_down,background=color,highlightbackground=color).grid(row=0,column=3,padx=5,pady=5)
Button(bottomFrame,text='+',width=W,command=G.amp_up,background=color,highlightbackground=color).grid(row=0,column=4,padx=5,pady=5)
Button(bottomFrame,text='Open directory',width=W,command=G.open_subject,background=color,highlightbackground=color).grid(row=0,column=5,padx=5,pady=5)
G.count_var = IntVar()
G.count_var.set(0)
G.counter = Label(bottomFrame,textvariable=G.count_var,relief=SUNKEN).grid(row=0,column=6,padx=5,pady=5)

# ***** Mainloop *****
G.root.mainloop()
