# pre-post-maker

Python script to assign a set of two isometric assessment question sets from GenBio-MAPS. Can be adapted for other pre-post or multi-section tests question assignments.

GenBio-MAPS (short for _General Biology–Measuring Achievement and Progression in Science_) is an assessment tool that measure student understanding of the core concepts at key time points in a biology degree program consisting of 39 multi-part question items (see the [official paper](https://doi.org/10.1187/cbe.18-07-0117) published in _CBE—Life Sciences Education_ for more context).

In typically use, the assessment is conducted at least twice. To avoid [testing effect](https://en.wikipedia.org/wiki/Testing_effect), the questions on the two assessments should be distinct, yet the two question sets should cover similar range of concepts, which is where this script comes in.

The python script `PrePostMaker.py` implements an algorithm that aims to produce two question sets of similar concept coverage but with distinct questions. In addition, it also enforces constraints on the total number of subparts and the ratio of true (T) to false (F) answers. Additional information can be found in the module docstring of the source code, which is also accessible by running `PrePostMaker.py` at the command line with the `-H` option.

The `PrePostMaker.py` script can either be used directly from the command line or imported as a python module. In the former case, it takes an input file that describes the properties of the questions. By adapting this file (and changing the optional configuration `.json` file), the script can also be used for assigning other pre-post or multi-section tests.

Since the answers to the GenBio-MAPS questions should not be readily available to students, this repo includes an **altered** input file named `genbiomaps_attr_altered.csv`, for which the number of T and F answers for each question item is randomly generated. Except that, all other attributes do match the actual questions on GenBio-MAPS.
