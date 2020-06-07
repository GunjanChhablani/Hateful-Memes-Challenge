# Hateful-Memes-Challenge
Work on the Hateful Memes Challenge 2020 by Facebook AI

URLs : [Facebook](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set), [DrivenData](https://www.drivendata.org/competitions/64/hateful-memes/?fbclid=IwAR0tcPtO2MEYoCoPYMOFAf9LpkEVDvlJ2PDbXBFAkS1rjdOlECgIzNZOol4)


## Usage
```bash
make
conda activate hmc
python train.py
```
You can also use one of our notebooks for the same.

## Supported Models
- [x] ResNet 152, 2 Classification Layers

- [ ] BERT


## Data Description
This description of the data is based on the [paper](https://arxiv.org/pdf/2005.04790.pdf) on the Challenge:
- The dataset is created from scratch based on images from Getty Images
- Third-party annotation company for labelling the dataset.
- Contains contrastive or counterfactual examples
- Most memes that are uni-modal hateful are text-only.
- Total 10k memes - Types: multimodal hate (40%), unimodal hate(10%), benign image(20%), benign text(20%), random non-hateful(10%)
- Dev  = 5%, Test = 10%
- Evaluation Metrics : ROC AUC, Accuracy

More information will be added with the Exploratory Data Analysis on this dataset.

## Dataset Analysis (See Paper)
- Inter-Annotater Agreement
  - Use Cohen's kappa core for annotators : 67.2% ('moderate').
  - Final human accuracy is 84.7% on average (after human experts).
  - Hate Categories - Protected Categories - Race and Religion based hate Agreement most prevalent.
  - Type of Attack - Dehumanization is the most prevalent.
  - Lexical Studies - Text Analysis using word frequencies, **Visual Analysis using bounding boxes from Mask R-CNN**.
<NEEDS more EDA>


## Experiments

### Phase 1 - Expected Date : 7 June 2020.
For the first phase, we use the weighted Adam Optimization technique, with the suggest &eta;,&beta; values from the paper. We use the number of epochs for each case depending on the step value of 22000, and the batch size. A cosine decay scheduler with warm restart is used on the learning rate, and the values chosen are inspired from the paper.

|Model|Frozen Layers|Used Layer|Classification Layer|Train AUROC|Dev AUROC|Test AUROC|
|-|-|-|-|-|-|-|

## Baseline Performance (See Paper)
<NEED TO ADD BASELINE RESULTS FROM THE PAPER>

## Useful Links
- [Starter Kit Code](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes)
- [Facebook Blog on Hateful Memes Challenge](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set)
- [MMF - Multimodal Framework] (https://mmf.readthedocs.io/en/latest/)

## Useful Paper Descriptions
- **The Hateful Memes Challenge : Detecting Hate Speech in Multimodal Memes**, Authors : Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine - [Paper Link](https://arxiv.org/pdf/2005.04790.pdf)
  - The Introduction contains lots of related work on multimodal problems.
  - Cites papers where visual understanding mattered little compared to the language understanding.
  - Works with contrastive examples, has citations.
  - Benefits of custom memes:
    - Avoids noise from Optical Character Recognition.
    - Reduces bias existing in visual modality (using images from different sources/tools).
  - Annotation Process described in extreme detail.
  - Bibliography illed with resources for multimodal learning.


## Extra Papers
- Multimodal Deep Learning, Authors : - [Paper Link](https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf), Authors : Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, Andrew Y. Ng
  -
