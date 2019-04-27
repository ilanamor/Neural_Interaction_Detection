import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from NID import NID


class GUI:
    file_path = ''
    out_path = ''

    def check_is_number(self,entry,entry_name):
        try:
            value = int(entry.get())
            if(value<2):
                tk.messagebox.showerror("wrong params", entry_name + " must be greater than 1!")
                return 1;
        except ValueError:
            tk.messagebox.showerror("wrong params", entry_name +" must be a number!")
            return 1;

    def check_empty_string(self, entry, entry_name):
        if (entry.get() == ""):
            tk.messagebox.showinfo("fill params", "Insert " + entry_name)
            return 1;


    def start(self):

        if (not self.file_path):
            tk.messagebox.showinfo("fill params", "Insert a file path")
            return
        if(not os.path.getsize(self.file_path) > 0):
            tk.messagebox.showinfo("wrong params", "Selected file is empty, please choose again")
            return
        if (not self.out_path):
            tk.messagebox.showinfo("fill params", "Insert a directory path to output file")
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
            tk.messagebox.showinfo("fill params", "Choose regression or classification")
            return
        if (self.mlp.get() == 0 and self.mlpm.get() == 0 and self.mlp_cutoff.get() == 0):
            tk.messagebox.showinfo("fill params", "Choose MLP or MLP-M or MLP-Cutoff")
            return

        try:
            units_list = list(map(int, self.units_arr_entry.get().split(',')))
        except ValueError:
            tk.messagebox.showinfo("wrong params", "network structure must be numbers separated by comma")
            return

        try:
            nid = NID(self.use_main_effect_nets, self.use_cutoff, int(self.is_index_col.get()), int(self.is_header.get()),
                      self.file_path, self.out_path, units_list, self.is_classification_col.get(),
                      int(self.k_fold_entry.get()), int(self.num_epochs_entry.get()), int(self.batch_size_entry.get()))
        except ValueError as err:
            if err.args[0] == 'Empty file input':
                tk.messagebox.showinfo("wrong params", "Selected file is empty, please choose again")
            elif err.args[0] == 'Mismatch between K-folds and dataset':
                tk.messagebox.showinfo("wrong params", "Mismatch between K-folds and dataset")
            elif err.args[0] == 'Incompatible dataset type and target type':
                tk.messagebox.showinfo("wrong params", "Incompatible dataset type and target type")
            else:
                tk.messagebox.showinfo("error", "Something went wrong, please try again")
            return

        self.status.config(text='Process Started...')
        self.status.update()
        nid.run()
        self.status.config(text='Process Completed')
        self.status.update()
        messagebox.showinfo("Info", "NID Process Completed successfully!")

        print('\nend')

    def exit(self):
        # sys.exit()
        if tk.messagebox.askokcancel("Close", "Are you sure you want to exit?"):
            sys.exit()


    def __init__(self,master):
        self.master = master
        self.master.geometry("480x410")
        self.master.title("NID framework")

        self.labelTitle = tk.Label(self.master, text="Fill Params And Paths:")
        self.labelTitle.place(x=140, y=10)

        #path to file
        self.labelPath = tk.Label (self.master,  text="File Path:" )
        self.entryPath = tk.Entry(self.master, width=40)
        self.labelPath.place(x=10, y=40)
        self.entryPath.place(x=100, y=40)
        self.browse_button = tk.Button(self.master, text="Browse", width=10, command=self.choosefile)
        self.browse_button.pack()
        self.browse_button.place(x=350, y=40)

        # path to output
        self.labelOutPath = tk.Label(self.master, text="Output Path:")
        self.entryOutPath = tk.Entry(self.master, width=40)
        self.labelOutPath.place(x=10, y=80)
        self.entryOutPath.place(x=100, y=80)
        self.browse_out_button = tk.Button(self.master, text="Browse", width=10, command=self.chooseOutFolder)
        self.browse_out_button.pack()
        self.browse_out_button.place(x=350, y=80)

        # units_arr
        self.units_arr_label = tk.Label(self.master, text="Network architecture:")
        self.units_arr_entry = tk.Entry(self.master, width=34)
        self.units_arr_label.place(x=10, y=120)
        self.units_arr_entry.place(x=135, y=120)
        self.units_arr_entry.insert(0, '140,100,60,20')

        #k-fold
        self.k_fold_label = tk.Label(self.master,  text="k-fold:" )
        self.k_fold_entry = tk.Entry(self.master, width=5)
        self.k_fold_label.place(x=10, y=160)
        self.k_fold_entry.place(x=55, y=160)
        self.k_fold_entry.insert(0, '5')

        # num_epochs
        self.num_epochs_label = tk.Label(self.master, text="num epochs:")
        self.num_epochs_entry = tk.Entry(self.master, width=5)
        self.num_epochs_label.place(x=140, y=160)
        self.num_epochs_entry.place(x=220, y=160)
        self.num_epochs_entry.insert(0, '200')

        # batch_size
        self.batch_size_label = tk.Label(self.master, text="batch size:")
        self.batch_size_entry= tk.Entry(self.master, width=5)
        self.batch_size_label.place(x=300, y=160)
        self.batch_size_entry.place(x=370, y=160)
        self.batch_size_entry.insert(0, '100')

        # has header
        self.is_header = tk.IntVar()
        self.is_header_entry = tk.Checkbutton(self.master, text="header", variable=self.is_header, width=10)
        self.is_header_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_header_entry.place(x=10, y=200)

        # index column
        self.is_index_col = tk.IntVar()
        self.is_index_col_entry = tk.Checkbutton(self.master, text="index", variable=self.is_index_col, width=10)
        self.is_index_col_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_index_col_entry.place(x=160, y=200)

        # regression
        self.is_regression_col = tk.IntVar()
        self.is_regression_entry = tk.Checkbutton(self.master, text="regression", variable=self.is_regression_col, width=10,
                                                  command=self.regression_clicked)
        self.is_regression_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_regression_entry.place(x=10, y=240)

        # classification
        self.is_classification_col = tk.IntVar()
        self.is_classification_entry = tk.Checkbutton(self.master, text="classification",
                                                      variable=self.is_classification_col, width=10,
                                                      command=self.classification_clicked)
        self.is_classification_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_classification_entry.place(x=160, y=240)

        # MLP
        self.mlp = tk.IntVar()
        self.mlp_entry = tk.Checkbutton(self.master, text="MLP",variable=self.mlp, width=10, command=self.mlp_clicked)
        self.mlp_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlp_entry.place(x=10, y=280)

        # MLP-M
        self.mlpm = tk.IntVar()
        self.mlpm_entry = tk.Checkbutton(self.master, text="MLP-M",
                                                         variable=self.mlpm, width=10, command=self.mlpm_clicked)
        self.mlpm_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlpm_entry.place(x=160, y=280)

        # MLP-Cutoff
        self.mlp_cutoff = tk.IntVar()
        self.mlp_cutoff_entry = tk.Checkbutton(self.master, text="MLP-Cutoff",
                                                         variable=self.mlp_cutoff, width=10, command=self.mlp_cutoff_clicked)
        self.mlp_cutoff_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlp_cutoff_entry.place(x=310, y=280)

        # start
        self.submit_button = tk.Button(self.master, text="Start", width=15, command=self.start)
        self.submit_button.pack()
        self.submit_button.place(x=50, y=340)

        # Exit
        self.exit_button = tk.Button(self.master, text="Exit", width=15, command=self.exit)
        self.exit_button.pack()
        self.exit_button.place(x=250, y=340)

        self.status = tk.Label(self.master, text='', bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

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

    def mlp_clicked(self):
        if self.mlp.get() == 0:
            self.mlpm_entry.config(state="normal")
            self.mlp_cutoff_entry.config(state="normal")
        else:
            self.mlpm_entry.config(state="disable")
            self.mlp_cutoff_entry.config(state="disable")
            self.use_main_effect_nets = False
            self.use_cutoff = False

    def mlpm_clicked(self):
        if self.mlpm.get() == 0:
            self.mlp_entry.config(state="normal")
            self.mlp_cutoff_entry.config(state="normal")
        else:
            self.mlp_entry.config(state="disable")
            self.mlp_cutoff_entry.config(state="disable")
            self.use_main_effect_nets = True
            self.use_cutoff = False

    def mlp_cutoff_clicked(self):
        if self.mlp_cutoff.get() == 0:
            self.mlp_entry.config(state="normal")
            self.mlpm_entry.config(state="normal")
        else:
            self.mlp_entry.config(state="disable")
            self.mlpm_entry.config(state="disable")
            self.use_main_effect_nets = True
            self.use_cutoff = True


    def choosefile(self):
        self.file_path = tk.filedialog.askopenfilename()
        self.entryPath.delete(0, tk.END)
        self.entryPath.insert(0, self.file_path)

        if (not self.file_path):
            tk.messagebox .showinfo("fill params", "insert a file path")
            return

        if (not (self.file_path[-4:] == ".csv")):
            tk.messagebox .showerror("fill params", "insert a csv file")
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

def main():
    root = tk.Tk()
    my_gui = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()








