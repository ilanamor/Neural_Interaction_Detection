import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from NID import NID

class main:
    file_path = ''
    out_path = ''


    def start(self):
        units_list = list(map(int, self.units_arr_entry.get().split(',')))
        nid = NID(int(self.use_main_effect_nets.get()), int(self.is_index_col.get()), int(self.is_header.get()), self.file_path, self.out_path, units_list, self.is_classification_col.get())
        nid.run()
        print('\nend')
        sys.exit()

    def __init__(self,master):
        self.master = master
        master.geometry("450x300")
        master.title("NID framework")


        self.labelTitle = tk.Label(master, text="Fill Params And Paths:")
        self.labelTitle.place(x=140, y=10)

        #path to file
        self.labelPath = tk.Label ( master,  text="File Path:" )
        self.entryPath = tk.Entry(master, width=40)
        self.labelPath.place(x=10, y=40)
        self.entryPath.place(x=100, y=40)
        self.browse_button = tk.Button(master, text="Browse", width=10, command=self.choosefile)
        self.browse_button.pack()
        self.browse_button.place(x=350, y=40)

        # path to output
        self.labelOutPath = tk.Label(master, text="Output Path:")
        self.entryOutPath = tk.Entry(master, width=40)
        self.labelOutPath.place(x=10, y=80)
        self.entryOutPath.place(x=100, y=80)
        self.browse_out_button = tk.Button(master, text="Browse", width=10, command=self.chooseOutFolder)
        self.browse_out_button.pack()
        self.browse_out_button.place(x=350, y=80)

        #learning rate
        # self.learning_rate_label = tk.Label ( master,  text="learning rate:" )
        # self.learning_rate = tk.Entry(master, width=40)
        # self.learning_rate_label.place(x=10, y=80)
        # self.learning_rate.place(x=100, y=80)

       #num_epochs
        # self.num_epochs_label = tk.Label(master, text="num epochs:")
        # self.num_epochs = tk.Entry(master, width=40)
        # self.num_epochs_label.place(x=10, y=120)
        # self.num_epochs.place(x=100, y=120)

        #batch_size
        # self.batch_size_label = tk.Label(master, text="batch size:")
        # self.batch_size= tk.Entry(master, width=40)
        # self.batch_size_label.place(x=10, y=160)
        # self.batch_size.place(x=100, y=160)

        #number_of_hide_unit
        # self.number_of_hide_unit_label = tk.Label(master, text="number of hide units:")
        # self.number_of_hide_unit = tk.Entry(master, width=40)
        # self.number_of_hide_unit_label.place(x=10, y=200)
        # self.number_of_hide_unit.place(x=150, y=200)

        #units_arr
        self.units_arr_label = tk.Label(master, text="Num of units in hidden layers\n(example: 140,100,60,20)")
        self.units_arr_entry = tk.Entry(master, width=40)
        self.units_arr_label.place(x=10, y=110)
        self.units_arr_entry.place(x=190, y=120)
        self.units_arr_entry.insert(0, '140,100,60,20')


        #number_of_hide_unit
        # self.number_of_hide_unit_label = tk.Label(master, text="number of hide unit")
        # self.number_of_hide_unit = tk.Entry(master, width=40)
        # self.number_of_hide_unit_label.place(x=10, y=240)
        # self.number_of_hide_unit.place(x=150, y=240)

        #class_index
        # self.class_index_label = tk.Label(master, text="class index:")
        # self.class_index = tk.Entry(master, width=40)
        # self.class_index_label.place(x=10, y=280)
        # self.class_index.place(x=100, y=280)

        # use_main_effect_nets
        self.use_main_effect_nets = tk.IntVar()
        self.use_main_effect_nets_entry = tk.Checkbutton(master, text="use main effects", variable=self.use_main_effect_nets, width=15)
        self.use_main_effect_nets_entry.pack()
        self.use_main_effect_nets_entry.place(x=10, y=160)

        # has header
        self.is_header = tk.IntVar()
        self.is_header_entry = tk.Checkbutton(master, text="header", variable=self.is_header, width=10)
        self.is_header_entry.pack()
        self.is_header_entry.place(x=160, y=160)

        # index column
        self.is_index_col = tk.IntVar()
        self.is_index_col_entry = tk.Checkbutton(master, text="index", variable=self.is_index_col, width=10)
        self.is_index_col_entry.pack()
        self.is_index_col_entry.place(x=310, y=160)

        # regression
        self.is_regression_col = tk.IntVar()
        self.is_regression_entry = tk.Checkbutton(master, text="regression", variable=self.is_regression_col, width=10, command=self.regression_clicked)
        self.is_regression_entry.pack()
        self.is_regression_entry.place(x=10, y=200)

        # classification
        self.is_classification_col = tk.IntVar()
        self.is_classification_entry = tk.Checkbutton(master, text="classification", variable=self.is_classification_col, width=10, command=self.classification_clicked)
        self.is_classification_entry.pack()
        self.is_classification_entry.place(x=160, y=200)


        #start
        self.submit_button = tk.Button(master, text="start", width=20, command=self.start)
        self.submit_button.pack()
        self.submit_button.place(x=150, y=240)


    def regression_clicked(self):
        if self.is_regression_col.get() == 0:
            self.is_classification_entry.config(state="normal")
        else:
            self.is_classification_entry.config(state="disable")

    def classification_clicked(self):
        if self.is_classification_col.get() == 0:
            self.is_regression_entry.config(state="normal")
        else:
            self.is_regression_entry.config(state="disable")

    def choosefile(self):
        self.file_path = tk.filedialog.askopenfilename()
        self.entryPath.delete(0, tk.END)
        self.entryPath.insert(0, self.file_path)

        if (not self.file_path):
            tk.messagebox .showinfo("fill params", "insert a file path")
            return

        if (not (self.file_path[-4:] == ".csv")):
            tk.messagebox .showerror("fill params", "insert an csv file")
            return

        # if self.df.empty:
        #     tk.messagebox .showerror("fill params", "invalid file!")
        #     return

    def chooseOutFolder(self):
        self.out_path = tk.filedialog.askdirectory()
        self.entryOutPath.delete(0, tk.END)
        self.entryOutPath.insert(0, self.out_path)


# def on_closing():
#     if messagebox.askokcancel("Quit", "Do you want to quit?"):
#         root.destroy()


root = tk.Tk()
my_gui = main(root)
# root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()










