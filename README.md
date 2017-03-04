# genrify
a tool to intelligently categorize music tracks by genres

## Dataset
The dataset used to train the algorithms can be found [here](http://marsyasweb.appspot.com/download/data_sets/)
Audio files are of 22050Hz Mono 16-bit .wav format.
Support for different audio formats will be added in the future.

## Train the algorithms
In order to train the Multilayer Perceptron and SVM add the following program arguments:
> -train genres/jazz genres/classical genres/metal genres/pop

**-train**      - sets the program to extract features from audio files found in the folders given as arguments
**genres/jazz** - folder given as argument, relative to the directory the tool is run in

The genres will be extracted from the last folder name. (e.g. for **genres/jazz** it will be **jazz**)

## Classify audio files
If no arguments are given, the program will attempt to classify all the audio files found in **genres/?** folder.
The output result of both the MLP and SVM will be displayed in the console.
