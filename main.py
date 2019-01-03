import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import TrainNeuralNetwork as tn

class main:
    use_main_effect_nets = True # toggle this to use "main effect" nets #gui

    # Parameters
    file_path=""
    df = None
    learning_rate = 0.01 #gui
    num_epochs = 200 #gui
    batch_size = 100 #gui
    num_samples = 30000 #30k datapoints, split 1/3-1/3-1/3
    # Network Parameters
    number_of_hide_unit =0 #gui
    units_arr =[] #gui
    class_index ="" #gui
    num_input = 10 #num of features

    def start(self):
        tn.set_parameters(self.use_main_effect_nets, self.learning_rate,self.num_epochs, self.batch_size,
                    self.df.shape[0], self.number_of_hide_unit, self.hidden_layers, 10, self.df, self.class_index)

    def __init__(self,master):
        self.master = master
        master.geometry("550x650")
        master.title("fill params for neural network")

        #path to file
        self.labelPath = tk.Label ( master,  text="File Path:" )
        self.entryPath = tk.Entry(master, width=40)
        self.labelPath.place(x=10, y=10)
        self.entryPath.place(x=100, y=10)
        self.browse_button = tk.Button(master, text="Browse", width=10, command=self.choosefile)
        self.browse_button.pack()
        self.browse_button.place(x=350, y=10)

        #learning rate
        self.learning_rate_label = tk.Label ( master,  text="learning rate:" )
        self.learning_rate = tk.Entry (master, width=40)
        self.learning_rate_label.place(x=10, y=310)
        self.learning_rate.place(x=100, y=310)

       #num_epochs
        self.num_epochs_label = tk.Label(master, text="num epochs:")
        self.num_epochs = tk.Entry(master, width=40)
        self.num_epochs_label.place(x=10, y=70)
        self.num_epochs.place(x=100, y=70)

        #batch_size
        self.batch_size_label = tk.Label(master, text="batch size:")
        self.batch_size= tk.Entry(master, width=40)
        self.batch_size_label.place(x=10, y=370)
        self.batch_size.place(x=100, y=370)

        #number_of_hide_unit
        self.number_of_hide_unit_label = tk.Label(master, text="number of hide units:")
        self.number_of_hide_unit = tk.Entry(master, width=40)
        self.number_of_hide_unit_label.place(x=10, y=120)
        self.number_of_hide_unit.place(x=150, y=120)

        #units_arr
        self.units_arr_label = tk.Label(master, text="numbers in this format: (10,20,..)")
        self.units_arr_entry = tk.Entry(master, width=40)
        self.units_arr_label.place(x=10, y=180)
        self.units_arr_entry.place(x=190, y=180)
        #self.units_arr = self.units_arr_entry.split(',')

        #number_of_hide_unit
        self.number_of_hide_unit_label = tk.Label(master, text="number of hide unit")
        self.number_of_hide_unit = tk.Entry(master, width=40)
        self.number_of_hide_unit_label.place(x=10, y=240)
        self.number_of_hide_unit.place(x=150, y=240)

        #class_index
        self.class_index_label = tk.Label(master, text="class index:")
        self.class_index = tk.Entry(master, width=40)
        self.class_index_label.place(x=10, y=420)
        self.class_index.place(x=100, y=420)

        # use_main_effect_nets
        var1 = tk.IntVar()
        self.use_main_effect_nets_entry = tk.Checkbutton(master, text="use main effect", variable=var1, width=40)
        self.use_main_effect_nets_entry.pack()
        self.use_main_effect_nets_entry.place(x=10, y=470)
        self.use_main_effect_nets=var1

        self.submit_button = tk.Button(master, text="start", width=20, command=self.start, state=tk.DISABLED)
        self.submit_button.pack()
        self.submit_button.place(x=100, y=500)



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

        self.df = pd.read_excel(self.file_path)

        if self.df.empty:
            tk.messagebox .showerror("fill params", "invalid file!")
            return

root = tk.Tk()
my_gui = main(root)
root.mainloop()










