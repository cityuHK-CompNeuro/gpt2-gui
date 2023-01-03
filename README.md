# Amazon Review Generator
GUI with Fine-Tuned GPT2 model to predict text and generate amazon reviews.
  


## **GPT2-Pytorch with Text-Generator**

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

**Better Language Models and Their Implications**

> Our model, called GPT-2 (a successor to [GPT](https://blog.openai.com/language-unsupervised/)), was trained simply to predict the next word in 40GB of Internet text. Due to our concerns about malicious applications of the technology, we are not releasing the trained model. As an experiment in responsible disclosure, we are instead releasing a much [smaller model](https://github.com/openai/gpt-2) for researchers to experiment with, as well as a [technical paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). from [openAI Blog](https://blog.openai.com/better-language-models/)

This repository is simple implementation GPT-2 about **text-generator** in **Pytorch** with **compress code**

- The original repertoire is [openai/gpt-2](https://github.com/openai/gpt-2). Also You can Read Paper about gpt-2, ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). To Understand more detail concept, I recommend papers about Transformer Model.
- Good implementation GPT-2 in Pytorch which I referred to, [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), You can see more detail implementation in huggingface repository.

- Transformer(Self-Attention) Paper : [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- First OpenAi-GPT Paper : [Improving Language Understanding by Generative Pre-Training(2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- See [OpenAI Blog](https://blog.openai.com/better-language-models/) about GPT-2 and Paper

## Dataset
[Amazon Review 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/)  

## Quick Start

1. Download GPT2 pre-trained model in Pytorch which huggingface/pytorch-pretrained-BERT already made! (Thanks for sharing! it's help my problem transferring tensorflow(ckpt) file to Pytorch Model!)
```Windows CMD
$ git clone https://github.com/graykode/gpt-2-Pytorch && chdir gpt-gui
# setup requirements
$ pip install -r requirements.txt
```

2. Run textbox_UI_trained.py
```
$ python textbox_UI_trained.py
```
3. Type into the text box as though writing an Amazon review for your favorite music album  
- Hint: start by typing the name of your favorite musician  
  

## Dependencies
pytorch
numpy
PySimpleGUI

Hugging Face transformers

```
$ pip install -r requirements.txt
```

# Example Functionality
![Individual Letter Prediction](http://github.com/cityuHK-CompNeuro/gpt2-gui/tree/main/README_support/text_predict_0.gif)

## Author

Jeremiah Palmerston

## License

- OpenAi/GPT2 follows MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- MIT license provided herein

## Acknowledgement
