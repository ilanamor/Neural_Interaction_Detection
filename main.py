import numbers
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from NID import NID

class main:
    file_path = ''
    out_path = ''

    def check_is_number(self,entry,entry_name):
        try:
            value = int(entry.get())
            if(value<2):
                tk.messagebox.showerror("wrong params", entry_name + " must be bigger then 1!")
                return 1;
        except ValueError:
            tk.messagebox.showerror("wrong params", entry_name+" must be number!")
            return 1;

    def check_empty_string(self, entry, entry_name):
        if (entry.get() == ""):
            tk.messagebox.showinfo("fill params", "insert " + entry_name)
            return 1;

    def start(self):
        if (not self.file_path):
            tk.messagebox .showinfo("fill params", "insert a file path")
            return
        if (not self.out_path):
            tk.messagebox .showinfo("fill params", "insert a directory path to output file")
            return
        if (self.check_empty_string(self.units_arr_entry,'a network structure') ==1):
            return
        if (self.check_empty_string(self.k_fold_entry, 'k-fold number') == 1 or self.check_is_number(self.k_fold_entry, 'k-fold ')==1):
            return
        if (self.check_empty_string(self.num_epochs_entry, 'num epoch') == 1 or self.check_is_number(self.num_epochs_entry,'num epoch')==1):
            return
        if (self.check_empty_string(self.batch_size_entry, 'batch size') == 1 or self.check_is_number(self.batch_size_entry, 'batch size')==1):
            return
        if (self.is_regression_col.get() == 0 and self.is_classification_col.get() == 0):
            tk.messagebox.showinfo("fill params", "choose regression or classification")
            return

        try:
            units_list = list(map(int, self.units_arr_entry.get().split(',')))
        except ValueError:
            tk.messagebox.showerror("wrong params", "network structure must be numbers separated by comma")
            return
        nid = NID(int(self.use_main_effect_nets.get()), int(self.is_index_col.get()), int(self.is_header.get()), self.file_path, self.out_path, units_list, self.is_classification_col.get(),int(self.k_fold_entry.get()),int(self.num_epochs_entry.get()), int(self.batch_size_entry.get()))
        nid.run()
        print('\nend')
        sys.exit()


    def __init__(self,master):
        self.master = master
        master.geometry("450x370")
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

        # units_arr
        self.units_arr_label = tk.Label(master, text="Network architecture:")
        self.units_arr_entry = tk.Entry(master, width=34)
        self.units_arr_label.place(x=10, y=120)
        self.units_arr_entry.place(x=135, y=120)
        self.units_arr_entry.insert(0, '140,100,60,20')

        #k-fold
        self.k_fold_label = tk.Label ( master,  text="k-fold:" )
        self.k_fold_entry = tk.Entry(master, width=5)
        self.k_fold_label.place(x=10, y=160)
        self.k_fold_entry.place(x=55, y=160)
        self.k_fold_entry.insert(0, '5')

        # num_epochs
        self.num_epochs_label = tk.Label(master, text="num epochs:")
        self.num_epochs_entry = tk.Entry(master, width=5)
        self.num_epochs_label.place(x=140, y=160)
        self.num_epochs_entry.place(x=220, y=160)
        self.num_epochs_entry.insert(0, '200')

        # batch_size
        self.batch_size_label = tk.Label(master, text="batch size:")
        self.batch_size_entry= tk.Entry(master, width=5)
        self.batch_size_label.place(x=300, y=160)
        self.batch_size_entry.place(x=370, y=160)
        self.batch_size_entry.insert(0, '100')

        # use_main_effect_nets
        self.use_main_effect_nets = tk.IntVar()
        self.use_main_effect_nets_entry = tk.Checkbutton(master, text="use main effects",
                                                         variable=self.use_main_effect_nets, width=15)
        self.use_main_effect_nets_entry.pack()
        self.use_main_effect_nets_entry.place(x=10, y=200)

        # has header
        self.is_header = tk.IntVar()
        self.is_header_entry = tk.Checkbutton(master, text="header", variable=self.is_header, width=10)
        self.is_header_entry.pack()
        self.is_header_entry.place(x=160, y=200)

        # index column
        self.is_index_col = tk.IntVar()
        self.is_index_col_entry = tk.Checkbutton(master, text="index", variable=self.is_index_col, width=10)
        self.is_index_col_entry.pack()
        self.is_index_col_entry.place(x=310, y=200)

        # regression
        self.is_regression_col = tk.IntVar()
        self.is_regression_entry = tk.Checkbutton(master, text="regression", variable=self.is_regression_col, width=10,
                                                  command=self.regression_clicked)
        self.is_regression_entry.pack()
        self.is_regression_entry.place(x=10, y=240)

        # classification
        self.is_classification_col = tk.IntVar()
        self.is_classification_entry = tk.Checkbutton(master, text="classification",
                                                      variable=self.is_classification_col, width=10,
                                                      command=self.classification_clicked)
        self.is_classification_entry.pack()
        self.is_classification_entry.place(x=160, y=240)

        # start
        self.submit_button = tk.Button(master, text="start", width=20, command=self.start)
        self.submit_button.pack()
        self.submit_button.place(x=150, y=300)

        #number_of_hide_unit
        # self.number_of_hide_unit_label = tk.Label(master, text="number of hide units:")
        # self.number_of_hide_unit = tk.Entry(master, width=40)
        # self.number_of_hide_unit_label.place(x=10, y=200)
        # self.number_of_hide_unit.place(x=150, y=200)



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
        if (not self.out_path):
            tk.messagebox.showinfo("fill params", "insert a directory path to output file")
            return


# def on_closing():
#     if messagebox.askokcancel("Quit", "Do you want to quit?"):
#         root.destroy()


root = tk.Tk()
my_gui = main(root)
# root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()










