# TopicModelingSS
This repository contains all of the work on the Proof of Concept for a topic modeling algorithm using prompt optimization and a llama model.

# To Begin Running locally
1. Download the Hein-Online Dataset (we chose only the Hein-daily) as specified in https://github.com/keyonvafa/tbip
2. Note the actual dataset can be found here: https://data.stanford.edu/congress_text
3. Run the PreProcessing Scripts(all they do is split the speeches by speakerId, and truncate speeches in such a way that no speech gets cut off halfway within some reasonable # of characters.
4. Get HuggingFace access for a llama-model; I used a local llama model, but getting one from a hugging face library should be fine (AutoTokenizer treats both the same (you'll need some refactorign if you do the latter))

# Deprecated Code
1. The VAE implementation with and without an LLM were done in earlier versions of the model, and are kept as a reference though not really used to generate any results.
# General
1. The most recent work was done in the tree_segmentation.ipynb where you can see the setup to manually try and run with different prompts repeatedly until we see a segmentation of the individual bodies of text. (an image can be seen along with a text file of what each tag/chunk looks like in the TreeSegmentationSample)
2. The other work focused on doing checks to see how the model would respond to good/bad prompts, and
3. All the config files for running on PACE can be found in the scripts section (be sure to follow these instructions for running said scripts https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
4. You'll see that these scripts all initially mapped to files in the sensitivity check section (note they're not in the same directory (that was done to run on the PACE server where I had them stored together)).
5. Even now a good practice to see that all code is running is to run the pre-processing scripts for just one congressional section( I chose the 97th for simplicity), and then run the PerplexityDifferenceMatrixGenerator.py to get csv's of how individual prompts are performing against one another.
6. These results will give you a good inference for how to build a prompt tree like the one seen in TreeSegmentationSample/Sample_Tree_Segment.png
#Important
1. Once the preprocessign and setup of the llama model is done. It's likely that the data_segmentation notebook will be the best workspace to continue building out the idea for the model.
2. A practice that I found useful is hosting the notebook on PACE (instructions on how are in the pace guide from earlier), and using that to do the repeated runs to get an idea for how to best build the prompts.
