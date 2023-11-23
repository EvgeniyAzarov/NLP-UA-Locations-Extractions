# NLP UA Locations Extractions

# * being updated *

## General thoughts and approaches 

## Combined multibert

1. First intention was to create combined dataset, which includes ukrainian and russian samples in the same proportion as in the test set, since I noticed that test set is strongly disbalanced in this sense. To save gpu time (I have 30hr per week from the kaggle) and be able to conduct different experiments, I used a part of the uk dataset with `100k` samples. Languages ratios I calculated on the test set with `langid` module. 

Training code for this case can be found here: [multibert_combined_train_1.ipynb](notebooks/multibert_combined_train_1.ipynb). Result of this model on LB is below


![image.png](attachment:image.png)

2. Lang ratio in the test dataset is approximetely `8`, i.e. there are `8` times more uk samples than ru ones. Taking `100k` uk samples for training means that in previous approach for ru will be taken only about `12k` samples. I wasn't sure if this can be enough for model fine-tuning and decided to try to train model on the dataset where there is the same number of ru samples as uk ones. For this I took `100k` uk and `100k` ru samples from the original dataset. Also, I noticed that in the uk dataset there are only `10k` samples proposed for validation, so I take them all instead of sampling too.

Script for creating these versions of datasets is placed here: [create_light_versions.ipynb](create_light_versions.ipynb).

Training notebook: [multibert-combined_train_2.ipynb](multibert-combined_train_2.ipynb).


In order to choose suitable threshold for entities score (0.93 in this case) I used `labeling_sample.csv` data, as it's the most close one to the real test set that we have. Since it's very small, it's not a perfect solution, yet calibrating threshold on this data boosted LB performance of the models a bit. This procedure can be observed for example in the inference notebook [multibert_combined_inference.ipynb](multibert_combined_inference.ipynb)



## Separated models 

For separated models I use `langid` module for language detection, because it allows restricting prediction to two languages from the box. Its performance estimation and small modification are placed here: [lang_detection.ipynb](lang_detection.ipynb). 

There were two possibilities (at least two) for base uk ner base model: multibert and youscan/roberta-uk. I suspected that pre-trained uk model should perform better, and set up experiment to check this hypothesis, with two models which differ only in `uk` component: 
- `langid`, `multibert` fine-tune for `uk`, `spacy` pre-trained ner for `ru`
- `langid`, `roberta-uk` fine-tune for `uk`, `spacy` pre-trained ner for `ru`

Here is a training notebook for the roberta-uk: [roberta_uk.ipynb](roberta_uk.ipynb)

Fine-tuning uk model on the `multibert` under the same training conditions performed much better. 




