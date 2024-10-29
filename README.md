# DynamicSpindleKymograph

This python script is designed to take in a series of MATLAB files containing spindle pole locations and raw image files and produce a kymograph of the cell flourescence intensity along the spindle (between the spindle pole locations).

In order to perform the analysis, just change the dictionaries `matlab_file_paths` and `image_file_paths` to contain the paths to the files you wish to analyze and then run the python script.

THe python script will create a folder named `dynamic_kymographs` containing the SVGs of the kymographs. 
