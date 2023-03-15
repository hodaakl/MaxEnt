import os 
import csv

def write_output(file_path, new_row ): 
    if os.path.exists(file_path): 
        with open(file_path, 'a') as add_file:
            csv_adder = csv.writer(add_file, delimiter = ',')
            csv_adder.writerow(new_row)
            add_file.flush()
    else:
        with open(file_path, 'w') as new_file:

            csv_maker = csv.writer(new_file, delimiter = ',')
            csv_maker.writerow(new_row)
            new_file.flush()
    return 
