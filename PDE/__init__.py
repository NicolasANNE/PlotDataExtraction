import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory: {os.getcwd()}")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Changed working directory to: {os.getcwd()}")
from LangchainExtraction import PlotDataExtractor
from MAE_validation import *

