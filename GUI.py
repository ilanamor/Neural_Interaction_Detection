import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from NID import NID
from PIL import Image, ImageTk



class GUI:
    file_path = ''
    out_path = ''

    def __init__(self, master):
        self.master = master
        self.master.geometry("480x520")
        self.master.title("NID framework")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.labelTitle = tk.Label(self.master, text="Fill Params And Paths:")
        self.labelTitle.place(x=140, y=10)
        self.info_im = ImageTk.PhotoImage(Image.open('Images/info.png').resize((15,15),Image.ANTIALIAS))


        # path to file
        self.labelPath = tk.Label(self.master, text="File Path:")
        self.entryPath = tk.Entry(self.master, width=40)
        self.labelPath.place(x=10, y=40)
        self.entryPath.place(x=100, y=40)
        self.browse_button = tk.Button(self.master, text="Browse", width=10, command=self.choosefile)
        self.browse_button.pack()
        self.browse_button.place(x=370, y=40)
        self.info_button1 = tk.Button(self.master, image=self.info_im, width=20, height=15, command=lambda: self.info('Data-Set input file'))
        self.info_button1.pack()
        self.info_button1.place(x=340, y=40)

        # path to output
        self.labelOutPath = tk.Label(self.master, text="Output Path:")
        self.entryOutPath = tk.Entry(self.master, width=40)
        self.labelOutPath.place(x=10, y=80)
        self.entryOutPath.place(x=100, y=80)
        self.browse_out_button = tk.Button(self.master, text="Browse", width=10, command=self.chooseOutFolder)
        self.browse_out_button.pack()
        self.browse_out_button.place(x=370, y=80)
        self.info_button2 = tk.Button(self.master, image=self.info_im, width=20, height=15,command=lambda: self.info('Results output path'))
        self.info_button2.pack()
        self.info_button2.place(x=340, y=80)

        # Network Architecture
        self.architecture = tk.Label(self.master, text="Network Architecture:",font= "Calibri 10 underline")
        self.architecture.place(x=10, y=120)
        info_text = 'Please fill the following network architecture settings:\n\n\n' +\
                    '1. Number of units in each hidden layer, separating by comma\n'+\
                    '    e.g: 140,100,60,20\n    means 140 units in first layer, 100 in second layer etc..\n\n'+\
                    '2. Number of folds for the network running,\n    by default set to 5 folds\n\n' +\
                    '3. Number of epoches for the network running,\n    by default set to 200\n\n' +\
                    '4. Size of each batch in network running\n'
        self.info_button3 = tk.Button(self.master, image=self.info_im, width=20, height=15,command=lambda: self.info(info_text))
        self.info_button3.pack()
        self.info_button3.place(x=140, y=120)

        # units_arr
        self.units_arr_label = tk.Label(self.master, text="Units in hidden layers:")
        self.units_arr_label.place(x=30, y=160)
        self.units_arr_entry = tk.Entry(self.master, width=34)
        self.units_arr_entry.place(x=155, y=160)
        self.units_arr_entry.insert(0, '140,100,60,20')

        # k-fold
        self.k_fold_label = tk.Label(self.master, text="K-folds:")
        self.k_fold_entry = tk.Entry(self.master, width=5)
        self.k_fold_label.place(x=30, y=200)
        self.k_fold_entry.place(x=80, y=200)
        self.k_fold_entry.insert(0, '5')

        # num_epochs
        self.num_epochs_label = tk.Label(self.master, text="Num epochs:")
        self.num_epochs_entry = tk.Entry(self.master, width=5)
        self.num_epochs_label.place(x=160, y=200)
        self.num_epochs_entry.place(x=240, y=200)
        self.num_epochs_entry.insert(0, '200')

        # batch_size
        self.batch_size_label = tk.Label(self.master, text="Batch size:")
        self.batch_size_entry = tk.Entry(self.master, width=5)
        self.batch_size_label.place(x=320, y=200)
        self.batch_size_entry.place(x=390, y=200)
        self.batch_size_entry.insert(0, '100')

        # dataset details
        self.ds_details = tk.Label(self.master, text="Data-Set details:",font= "Calibri 10 underline")
        self.ds_details.place(x=10, y=240)
        info_text = 'Please fill the following dataset details:\n\n\n' + \
                    '1. Whether contains an header row or not \n\n' + \
                    '2. Whether contains index column or not\n    Note: index column should be the first one\n\n' + \
                    '3. Whether it is regression or classification\n\n' + \
                    'Note: target column should be the last one\n'
        self.info_button4 = tk.Button(self.master, image=self.info_im, width=20, height=15,
                                      command=lambda: self.info(info_text))
        self.info_button4.pack()
        self.info_button4.place(x=115, y=240)

        # has header
        self.is_header = tk.IntVar()
        self.is_header_entry = tk.Checkbutton(self.master, text="header", variable=self.is_header, width=10)
        self.is_header_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_header_entry.place(x=30, y=280)

        # index column
        self.is_index_col = tk.IntVar()
        self.is_index_col_entry = tk.Checkbutton(self.master, text="index", variable=self.is_index_col, width=10)
        self.is_index_col_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_index_col_entry.place(x=180, y=280)

        # regression
        self.is_regression_col = tk.IntVar()
        self.is_regression_entry = tk.Checkbutton(self.master, text="regression", variable=self.is_regression_col,
                                                  width=10,
                                                  command=self.regression_clicked)
        self.is_regression_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_regression_entry.place(x=40, y=320)

        # classification
        self.is_classification_col = tk.IntVar()
        self.is_classification_entry = tk.Checkbutton(self.master, text="classification",
                                                      variable=self.is_classification_col, width=10,
                                                      command=self.classification_clicked)
        self.is_classification_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.is_classification_entry.place(x=200, y=320)

        # running details
        self.ds_details = tk.Label(self.master, text="Running details:", font= "Calibri 10 underline")
        self.ds_details.place(x=10, y=360)
        info_text = 'Please choose the running type:\n\n\n' + \
                    '1. MLP : Multilayer perceptron \n\n' + \
                    '2. MLP-M : Multilayer perceptron with main effects\n\n' + \
                    '3. MLP-cutoff : MLP-M with cutoff (for validation needs)\n'
        self.info_button5 = tk.Button(self.master, image=self.info_im, width=20, height=15,
                                      command=lambda: self.info(info_text))
        self.info_button5.pack()
        self.info_button5.place(x=115, y=360)

        # MLP
        self.mlp = tk.IntVar()
        self.mlp_entry = tk.Checkbutton(self.master, text="MLP", variable=self.mlp, width=10, command=self.mlp_clicked)
        self.mlp_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlp_entry.place(x=30, y=400)

        # MLP-M
        self.mlpm = tk.IntVar()
        self.mlpm_entry = tk.Checkbutton(self.master, text="MLP-M",
                                         variable=self.mlpm, width=10, command=self.mlpm_clicked)
        self.mlpm_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlpm_entry.place(x=180, y=400)

        # MLP-Cutoff
        self.mlp_cutoff = tk.IntVar()
        self.mlp_cutoff_entry = tk.Checkbutton(self.master, text="MLP-Cutoff",
                                               variable=self.mlp_cutoff, width=10, command=self.mlp_cutoff_clicked)
        self.mlp_cutoff_entry.pack(side=tk.LEFT, anchor=tk.W)
        self.mlp_cutoff_entry.place(x=330, y=400)

        # start
        self.submit_button = tk.Button(self.master, text="Start", width=15, command=self.start, bg="#c7c9cc")
        self.submit_button.pack()
        self.submit_button.place(x=80, y=450)

        # Exit
        self.exit_button = tk.Button(self.master, text="Exit", width=15, command=self.on_closing, bg="#c7c9cc")
        self.exit_button.pack()
        self.exit_button.place(x=280, y=450)

        self.status = tk.Label(self.master, text='', bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def start(self):

        if (not self.file_path):
            tk.messagebox.showerror("Missing parameter", "Insert a file path")
            return
        if(not os.path.getsize(self.file_path) > 0):
            tk.messagebox.showerror("Incorrect parameter", "Selected file is empty, please choose again")
            return
        if (not self.out_path):
            tk.messagebox.showerror("Missing parameter", "Insert a directory path to output file")
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
            tk.messagebox.showerror("Missing parameter", "Choose regression or classification")
            return
        if (self.mlp.get() == 0 and self.mlpm.get() == 0 and self.mlp_cutoff.get() == 0):
            tk.messagebox.showerror("Missing parameter", "Choose MLP or MLP-M or MLP-Cutoff")
            return

        try:
            units_list = list(map(int, self.units_arr_entry.get().split(',')))
        except ValueError:
            tk.messagebox.showerror("Incorrect parameter", "Network structure must be numbers separated by comma")
            return

        try:
            nid = NID(self.use_main_effect_nets, self.use_cutoff, int(self.is_index_col.get()), int(self.is_header.get()),
                      self.file_path, self.out_path, units_list, self.is_classification_col.get(),
                      int(self.k_fold_entry.get()), int(self.num_epochs_entry.get()), int(self.batch_size_entry.get()))
        except ValueError as err:
            if err.args[0] == 'Empty file input':
                tk.messagebox.showerror("Incorrect parameter", "Selected file is empty, please choose again")
            elif err.args[0] == 'Mismatch between K-folds and dataset':
                tk.messagebox.showerror("Incorrect parameter", "Mismatch between K-folds and dataset")
            elif err.args[0] == 'Incompatible dataset type and target type':
                tk.messagebox.showerror("Incorrect parameter", "Incompatible dataset type and target type")
            else:
                tk.messagebox.showerror("Error", "Something went wrong, please try again")
            return

        self.status.config(text='Process Started...')
        self.status.update()
        self.freeze()
        start_time = time.time()
        err = nid.run()
        running_time = time.time() - start_time
        self.status.config(text='Process Completed')
        self.status.update()
        if self.is_classification_col.get() == 0 :
            messagebox.showinfo("Info", "NID Process Completed successfully!\nFinal RMSE is: " + str(err)+'\nRuning time: '+ str(running_time))
        else:
            messagebox.showinfo("Info", "NID Process Completed successfully!\nFinal (1-AUC) is: " + str(
                err) + '\nRunning time: ' + str(running_time))
        print('\nend')

    def on_closing(self):
        if tk.messagebox.askokcancel("Close", "Are you sure you want to exit?"):
            self.master.destroy()
            sys.exit()

    def info(self,text):
        tk.messagebox.showinfo("Info", text)

    def check_is_number(self,entry,entry_name):
        try:
            value = int(entry.get())
            if(value<2):
                tk.messagebox.showerror("Incorrect parameter", entry_name + " must be greater than 1")
                return 1;
        except ValueError:
            tk.messagebox.showerror("Incorrect parameter", entry_name +" must be a number")
            return 1;

    def check_empty_string(self, entry, entry_name):
        if (entry.get() == ""):
            tk.messagebox.showerror("Missing parameter", "Insert " + entry_name)
            return 1;

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
            tk.messagebox.showerror("Missing parameter", "Insert a file path")
            return

        if (not (self.file_path[-4:] == ".csv")):
            tk.messagebox.showerror("Missing parameter", "Insert a csv file")
            return

    def chooseOutFolder(self):
        self.out_path = tk.filedialog.askdirectory()
        self.entryOutPath.delete(0, tk.END)
        self.entryOutPath.insert(0, self.out_path)
        if (not self.out_path):
            tk.messagebox.showerror("Missing parameter", "Insert a directory path to output file")
            return

    def freeze(self):
        self.mlp_entry.config(state="disable")
        self.mlp_entry.config(state="disable")
        self.mlp_cutoff_entry.config(state="disable")
        self.entryPath.config(state="disable")
        self.entryOutPath.config(state="disable")
        self.units_arr_entry.config(state="disable")
        self.k_fold_entry.config(state="disable")
        self.num_epochs_entry.config(state="disable")
        self.batch_size_entry.config(state="disable")
        self.is_header_entry.config(state="disable")
        self.is_index_col_entry.config(state="disable")
        self.is_regression_entry.config(state="disable")
        self.is_classification_entry.config(state="disable")
        self.submit_button.config(state="disable")
        self.submit_button.update()

def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()








