The package contains 3 python scripts and one jupyter notebook's ipynb file.
The clustering function was written and tested using jupyter notebook and was attached along with this folder. However, for the purpose of usability, three separate python files are created for convenience which are:

"cluster.py" - this is the main script used for clustering data, it contains a clustering function which takes csv file as an input, the input data can have any number of dimensions. However, only data with an exactly two dimensions will output a matplotlib visualization as a PNG file. Every output files will be written into a subfolder "clustering_output/"

Example Usage: "./cluster.py input.csv" or "python cluster.py input.csv" then enter the indexes of data points into the prompt that will be used as initial centroids seperating each one by space.

"generator-2d.py" - this file is used to generate two-dimensional test data which will output csv file that is readable by the cluster.py

"student_data-generator.py" - this file is used to generate data as required by the assignment instruction, it will produce all four attributes using normal distribution randomizer with a unique mean for each clusters. The script can be used by running it and enter preferred values into the prompt including the following:
- Points for each clusters: number of data points for each clusters
- No. of clusters : number of clusters to be generated
- Spread percentage: the percentage of the spread away from the mean in each clusters

List of Python packages used:
Numpy - for numerical computation
Matplotlib - for data visualization
Pandas - for data transformation and CSV File operations
scipy - for data randomization generation