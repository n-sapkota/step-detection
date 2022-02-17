# Step Detection and Indoor Localization
This repo consist of python code for step detection using machine learning approaches. I have done this project as my master thesis. 
## Abstract
Localisation is a process of determination of the target location. Localisation helps in the
navigation of unfamiliar areas including location-based services such as weather information
and traffic. In indoors, localisation can be done by detecting consecutive steps of
users where other means of localisation is not possible. Here, we present the results for
the step detections using machine learning where the decision tree is used as a learning
algorithm. To detect whether the user’s step is identified or not, we used accelerometer
and gyroscope data that are obtained by attaching three sensors namely left, center,
and right on the user’s waist. The high-frequency noise is removed by employing a low
pass Butterworth filter, and the filtered signal is divided into numbers of segments. We
extracted features from the time domain and frequency domain utilising statistical analysis
for each segment. Furthermore, features are ranked based on Pearson’s correlation
coefficient. Investigating all the possible combinations of attached sensors, the step detector
that uses the datasets from the sensor attached at the center of the waist of the
user shows 7.40 per cent misdetection rate to detect step and that classifier used mostly
time domain features namely mean, crest factor, kurtosis, variance, and IQR.

## Moivation
Nowadays, human indoor activities are increasing steadily that demands reliable localisation.
Localisation is a process of determination of the target location. Localisation
helps in the navigation of unfamiliar areas as well as provides information and a variety
of services and applications. Information about weather, traffic, catering, advertising in a
more targeted way are some examples of information and services which can be achieved
by employing localisation [7]. To get accurate, reliable and permeating localisation in
diverse situations endeavours have been made. Satellite supported global navigation
system such as a global positioning system (GPS) provides a precise localisation at the
outdoors. These positioning systems can be used indoors as well as at outdoors. However,
performance differs significantly, since the environments have several significant dissimilarities
[8]. To determine the target location; a GPS device requires a series of satellites
signal. The signals sent from these satellites do not readily penetrate all sorts of physical
barriers such as roofs, walls and other objects and potential sources of interference, for
instance, reflection at a surface of ceilings which make difficult for the device to identify
the location accurately [9]. When the device has a reasonable observable line-of-sight
to the sky, a GPS works better. The more signal that can be accessed by the device,
the more precise it is. Whenever inside, there is no direct line-of-sight from the satellite
signals to the device and signal attenuated by construction materials. As a result, GPS
accuracy is degraded.

An indoor positioning system (IPS) has become one of the most promising technology
to deal with the limitations of GPS indoors. Signals such as lights, radio waves, inertial
measurements, acoustic signals, or other sensory information are used by IPS to localise
objects or people inside a building [10]. Localisation can be done by detecting consecutive
steps taken by the people in such areas where other means of localisation is not
possible. Detecting a user’s consecutive steps can be termed as a step detection. The
main goal of this thesis is to detect a user’s consecutive step. In general, to detect a step
we can find the abrupt changes in the signal. Many methods are in existence to detect a
step reliably; however, those methods are too general and somehow difficult. So in this
thesis, we seek for a machine learning approach which could detect a step reliably and
precisely utilising the good evaluation of features.

## Data Acquisition
We used the previously collected data by
the Institute of Scientific and Industrial Research (ISIR), Osaka University (OU) for the purpose of this thesis. The
original dataset contains the triaxial accelerometer and gyroscope sensor data. Also, class
label and step label are also present on the datasets. These datasets are collected via
three sensors namely left, center, and right that is attached on the subject’s waist. More
details regarding the dataset (The OU-ISIR Gait Database) are briefly explained in this original research paper [here](http://www.am.sanken.osaka-u.ac.jp/~makihara/pdf/pr_2015Apr.pdf). The datasets are in the form of text files (.txt) which are stored in pandas (python data analysis library)
data frame with double-precision floating point format for this work.
