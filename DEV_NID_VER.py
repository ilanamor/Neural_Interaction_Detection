import time
from NID import NID


# The output (path taken from the UI): (1)Two csv files ,(2) Heat-Map for pairwise interactions, (3)Log file
    # 1st csv file for pairwise interaction
    # 2nd csv file for higher order interaction
    # every line contains interaction and strength
          
  '''input params, init format:
    use_main_effect_nets - whether use main effects or not (true / false)
    use_cutoff - whether use cutoff or not (true / false)
    is_index_col - is index column exists (1 true / 0 false)
    is_header - is header exists (1 true / 0 false)
    file_path - full path
    out_path - full path
    units_list - network architecture (list of numbers seperate by comma)
    is_classification_col - is classification dataset (1 true / 0 false, false means regression)
    k_fold_entry - number of folds (int, greater than 2)
    num_epochs_entry -  number of epochs (int, greater than 1)
    batch_size_entry -  number of batches (int, greater than 1)
    '''
file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\higgs\higgs.csv'
output_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\higgs'
is_classification = 1


start_time = time.time()
nid = NID(main_effects=1, cutoff=0, is_index=1, is_header=1, file_path=file_path, output_path=output_path,
          hidden_layers_structure=[140,100,60,20],is_classification_data=is_classification, k_fold_num = 5,
          num_of_epochs = 200, batch_size = 100)

assessment = nid.run()
running_time = time.time() - start_time
if is_classification == 0:
    print( "NID Process Completed successfully!\nFinal RMSE is: " + str(assessment)+'\nRuning time: '+ str(running_time))
else:
    print("Info", "NID Process Completed successfully!\nFinal (1-AUC) is: " + str(assessment) + '\nRunning time: ' + str(running_time))

print('\nend')
