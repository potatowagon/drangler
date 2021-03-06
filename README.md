# Validation and Testing of Models
Code for the results reflected in the report under Chapter 5.5.1 can be found under [validation_report](validation_report)

# Drangler
Dance Data Wrangler

## Setup python enviroment
If your havn't already, download <a href="https://conda.io/en/latest/miniconda.html">conda</a>.

cd to root of this folder, and 
```
conda env create -f environment.yml
conda activate cg3002
```

## Install drangler
cd to drangler module, and run the setup script
```
cd ./drangler
python setup.py install
```
You may now import drangler globally.

## Using drangler

```
from drangler import Sampler

num_of_signals = 6
frame_length = 5
sampling_interval = 2

# Extract frames
sampler = Sampler.Sampler(num_of_signals, frame_length, sampling_interval)
frame_collection = sampler.sample(bytestring_set) # bytestring_set is a 2D numpy array of bytestrings 
```
See drangler/pytest.py and examples for more sample scripts

### Sampler.sample()
![alt text](./img/1.PNG)

First Incoming bytestring set
![alt text](./img/2.PNG)

Subsequent bytestring set
![alt text](./img/3.PNG)
![alt text](./img/4.PNG)

### Feature extraction 
![alt text](./img/5.PNG)
