# SpectralAnalysisTutorial
A tutorial on spectral analysis for psychologists and social scientists.

In 'dummy_example/' there is  executable.py which makes running a spectral analysis easy.

To use simply open up the .py file and edit the settings (e.g. filename, individual/dyadic analysis etc.) and run.


## Notes about exectuable.py

No python understanding should be required beyond that needed to install and run it. To run the analysis, put executable.py in the same folder as the data and run:

(Linux/bash) cd to the working directory: ```cd /where/data/is/stored/```

(Linux/bash) run script by calling python3 ```python3 executable.py```

When you run it, it automatically checks if the required packages are installed, if not, they are installed and imported.

At the top of the script are listed some assumptions for the data.

Also at the top of the script there are some parameters for *you* to change, e.g.:

interpolation = True/False   <- set this to true to first run interpolation before the analysis. Note that it also resaves an interpolated version of the dataset with a suffix ‘_interp.csv’

missing_value = 999   <- you can change this if your missing values are different

T = number of timepoints

analysis type = dyadic/ind

sample rate = number of samples in a month

significance test = whether to do significance testing

num_bootstraps = number of bootstraps to undertake if doing significance testing

windowing = whether to do Hanning function windowing

filename = this is the directory/name of your data

Once you have set these, the rest should take care of itself. It will output a number of files depending on the type of analysis, including:

- P values for the average individual FFT
- P values for the average CPSD for dyads
- Average FFT across individuals
- FFTs for all individuals
- Average CPSD for dyads
- Phase discrepancies for all dyads
- Average phase discrapancies across dyads


