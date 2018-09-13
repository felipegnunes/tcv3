import os

def get_immediate_subdirectories(directory_path):
	return [sub_directory for sub_directory in os.listdir(directory_path) 
		if os.path.isdir(os.path.join(directory_path, sub_directory))]
		
print(get_immediate_subdirectories('/home/felipe'))
