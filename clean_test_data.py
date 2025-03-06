import numpy as np

# Load the clean test data
clean_data = np.load('project_clean_test.npz')

# Print the keys (file names) in the archive
print(clean_data.files)
