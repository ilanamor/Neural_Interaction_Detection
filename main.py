import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import TrainNeuralNetwork as tn
import InterpretWeights as iw

class main:
    use_main_effect_nets = True # toggle this to use "main effect" nets #gui

    # Parameters
    file_path = ''
    df = None
    learning_rate = 0.01 #gui
    num_epochs = 200 #gui
    batch_size = 100 #gui
    num_samples = 30000 #30k datapoints, split 1/3-1/3-1/3
    # Network Parameters
    # Network Parameters
    number_of_hide_unit =0 #gui
    units_arr =[] #gui
    class_index ="" #gui
    num_input = 10 #num of features

    def start(self):
        # tn.set_parameters(int(self.use_main_effect_nets), float(self.learning_rate.get()),int(self.num_epochs.get()), int(self.batch_size.get()),
        #            int(self.df.shape[0]), int(self.number_of_hide_unit.get()), self.units_arr, 10, self.df, int(self.class_index.get()))
        tn.set_parameters(True, 0.01,200, 100,30000, 4,[140,100,60,20], 10,  pd.read_csv('synthetic.csv'), 10)
        sess, weights = tn.prepare_network()
        w_dict = sess.run(weights)

        # Variable-Order Interaction Ranking
        print(iw.get_interaction_ranking(w_dict))

        # Pairwise Interaction Ranking
        print(iw.get_pairwise_ranking(w_dict))

    def __init__(self,master):
        self.master = master
        master.geometry("450x450")
        master.title("fill params for neural network")


        self.labelTitle = tk.Label(master, text="Fill Params And Path:")
        self.labelTitle.place(x=140, y=10)

        #path to file
        self.labelPath = tk.Label ( master,  text="File Path:" )
        self.entryPath = tk.Entry(master, width=40)
        self.labelPath.place(x=10, y=40)
        self.entryPath.place(x=100, y=40)
        self.browse_button = tk.Button(master, text="Browse", width=10, command=self.choosefile)
        self.browse_button.pack()
        self.browse_button.place(x=350, y=40)

        #learning rate
        self.learning_rate_label = tk.Label ( master,  text="learning rate:" )
        self.learning_rate = tk.Entry(master, width=40)
        self.learning_rate_label.place(x=10, y=80)
        self.learning_rate.place(x=100, y=80)

       #num_epochs
        self.num_epochs_label = tk.Label(master, text="num epochs:")
        self.num_epochs = tk.Entry(master, width=40)
        self.num_epochs_label.place(x=10, y=120)
        self.num_epochs.place(x=100, y=120)

        #batch_size
        self.batch_size_label = tk.Label(master, text="batch size:")
        self.batch_size= tk.Entry(master, width=40)
        self.batch_size_label.place(x=10, y=160)
        self.batch_size.place(x=100, y=160)

        #number_of_hide_unit
        self.number_of_hide_unit_label = tk.Label(master, text="number of hide units:")
        self.number_of_hide_unit = tk.Entry(master, width=40)
        self.number_of_hide_unit_label.place(x=10, y=200)
        self.number_of_hide_unit.place(x=150, y=200)

        #units_arr
        self.units_arr_label = tk.Label(master, text="numbers in this format: 100,60,..")
        self.units_arr_entry = tk.Entry(master, width=40)
        self.units_arr_label.place(x=10, y=240)
        self.units_arr_entry.place(x=190, y=240)
        self.units_arr1 = self.units_arr_entry
        self.units_arr= [140,100,60,20]

        #number_of_hide_unit
        # self.number_of_hide_unit_label = tk.Label(master, text="number of hide unit")
        # self.number_of_hide_unit = tk.Entry(master, width=40)
        # self.number_of_hide_unit_label.place(x=10, y=240)
        # self.number_of_hide_unit.place(x=150, y=240)

        #class_index
        self.class_index_label = tk.Label(master, text="class index:")
        self.class_index = tk.Entry(master, width=40)
        self.class_index_label.place(x=10, y=280)
        self.class_index.place(x=100, y=280)

        # use_main_effect_nets
        var1 = tk.IntVar()
        self.use_main_effect_nets_entry = tk.Checkbutton(master, text="use main effect", variable=var1, width=40)
        self.use_main_effect_nets_entry.pack()
        self.use_main_effect_nets_entry.place(x=50, y=320)
        self.use_main_effect_nets=var1.get()

        self.submit_button = tk.Button(master, text="start", width=20, command=self.start)
        self.submit_button.pack()
        self.submit_button.place(x=150, y=400)



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

        self.df = pd.read_csv(self.file_path)

        if self.df.empty:
            tk.messagebox .showerror("fill params", "invalid file!")
            return

root = tk.Tk()
my_gui = main(root)
root.mainloop()










