import time
from NID import NID

file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\bike\hour_new.csv'
output_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\bike'
is_classification = 0


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