# SpectralAnalysisTutorial
A tutorial on spectral analysis for psychologists and social scientists.


# executable.py
There is  executable.py which makes running a spectral analysis easy. To use it, simply open up the .py file and edit the settings (e.g. filename, individual/dyadic analysis etc.) and run (more insructions/info below).

This script undertakes a number of analyses, following those used in:

**Vowels, Mark, Vowels, Wood (2018) Using spectral and cross-spectral analysis to identify patterns and synchrony in couples’ sexual desire. PloS One 13 (10). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0205330**

and those elaborated upon in:

**Vowels, Vowels, Wood (under review) Spectral and Cross-Spectral Analysis-a Tutorial for Psychologists and Social Scientists. PsyArXiv https://psyarxiv.com/mj75a/**

Please cite these works if you use this script : ]



## Notes about exectuable.py

No python understanding should be required beyond that needed to install and run it. To run the analysis, put executable.py in the same folder as the data and run:

(Linux/bash) cd to the working directory: ```cd /where/data/is/stored/```

(Linux/bash) run script by calling python3: ```python3 executable.py```

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


