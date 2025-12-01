this project is based the paper:
S. Basak and B. Scheers, "Passive radio system for real-time drone detection and DoA estimation," 2018 International Conference on Military Communications and Information Systems (ICMCIS), Warsaw, Poland, 2018, pp. 1-6, doi: 10.1109/ICMCIS.2018.8398721.

# Notes
### folders / files
- Final Drone RF: contains the test dataset
- images: plots from test.py
- best_model.keras: the trained DRNN model
- training.py: model training code
- test.py: code for testing model on test dataset
- Report.pdf: milstone 2 report


#

# To run the code first install the libs

```
pip install -r requirements.txt
```
# please download the dataset from this link and place it on the same directory level of the codes
https://universe.roboflow.com/mingchenggg/final-drone-rf/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
# run the test.py to see the model result
```
python test.py
```
