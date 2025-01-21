import os
print("Current working directory:", os.getcwd())
print("Does Dataset/lalonde.csv exist?", os.path.exists("Dataset/lalonde.csv"))
print("Full path:", os.path.abspath("Dataset/lalonde.csv"))