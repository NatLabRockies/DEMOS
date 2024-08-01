from pandas import HDFStore, DataFrame
import time

def compare_datasets(dset1, dset2):
    df1 = DataFrame(dset1).sort_index(axis=1)
    df2 = DataFrame(dset2).sort_index(axis=1)

    if not df1.equals(df2):
        print(f"Datasets are different.")
        return False

    return True

def compare_hdf5_files(file1_path, file2_path):
    with HDFStore(file1_path, 'r') as store1, HDFStore(file2_path, 'r') as store2:
        keys1 = set(store1.keys())
        keys2 = set(store2.keys())

        common_keys = keys1 & keys2
        only_in_store1 = keys1 - keys2
        only_in_store2 = keys2 - keys1

        if only_in_store1:
            print(f"Keys only in {file1_path}: {only_in_store1}")
            return False
        if only_in_store2:
            print(f"Keys only in {file2_path}: {only_in_store2}")
            return False

        for key in common_keys:
            print(f"Comparing dataset {key}......", end=" ")
            dset1 = store1[key]
            dset2 = store2[key]
            if not compare_datasets(dset1, dset2):
                print("Not Equal.")
                return False
            else:
                print("Equal.")
        return True

# Example usage:
start = time.time()
file1_path = 'data/model_data_origin_win.h5' #you may change file path here, like 'data/model_data_origin_linux.h5' if you're in Linux
file2_path = 'data/model_data_2011.h5' #you may change file path here
if compare_hdf5_files(file1_path, file2_path):
    print("All output datasets are equal.")
else:
    print("Unequal happens. ")
end = time.time()
print(f"Time for comparing is {round(end-start)} sec.")