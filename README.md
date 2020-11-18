# Applications of the Convexified Convolutional Neural Network

## Introduction
This repository contains files used in the application of the convexified convolutional neural network (CCNN). This class of models capture the parameter sharing of convolutional neural networks in a convex manner and have originally been described by Zhang, Liang & Wainwright [1]. They show that training a CCNN corresponds to a convex optimization problem which can be solved by a projected gradient descent algorithm. It was shown that the CCNN algorithm is able to obtain results comparable to state-of-the-art models from the nonconvex class of models when applied to image datasets. 

## The problem 
Clickbait is a term for web-based content which aims to convince users to click on said content and browse to the advertised website. The clickbait headlines typically aim to exploit the so-called "curiosity gap" by including certain terms that are psychologically stimulating the reader into desiring to know more. The type of language used in clickbait headlines is markedly different from serious headlines, but it can be diffcult to concisely describe these differences or to make a list of words which are typical for clickbait content. This provides us with an interesting proposition: whether it is possible for a neural network to distinguish between clickbait versus non-clickbait content by learning some underlying function of the words, or their combinations, used in the headlines. 

A dataset of 32,000 headlines was made available by [2], who have used it in their paper [3] which shows how several models (SVM, decision trees, random forest) can be used to reasonably good effect for the classification task. 

## Summary
In the work presented here, we investigate whether a text classification problem, when sentences are considered as an "image" or word vectors, can also be solved using the CCNN. To this end we conduct a simulation study and perform implementation on a classification problem of text data: clickbait. We find that the CCNN algorithm achieves a acceptable performance on the clickbait dataset while being faster to train than the convolutional neural networks that inspired them. The latter point is especially impressive considering the fact that the CNNs were implemented using the Tensorflow/Keras libraries in Python, which were ran efficiently on a GTX 1080 Ti GPU, while the CCNN was coded in NumPy and numexpr. Furthermore, it is to be expected that with further study, the CCNN could be further optimized and possibly gain in performance. However, a drawback of the CCNN is that it requires us to compute and approximate the kernel matrix which for meaningful problems will be very large. Also, it is not immediately clear how other deep neural network architectures could be convexified when there is a feedback loop of information, such as an RNN or an LSTM. Concluding, the results of the CCNN implementation show that it can be succesfully applied on text classification problems of this nature using vectorized word representations. For more details, see the thesis text. 

The following software was used:

- `Python 3`

The following `Python` libraries were used:
- `numpy`
- `linalg`
- `sys`
- `numexpr`
- `sklearn`
- `time`
- `pickle`
- `Tensorflow` + `keras`

Also note that I have used pre-trained vectors for word representation from GloVe. I used the Wikipedia 2014 + Gigaword 50-dimensional embeddings, which you can obtain here: https://nlp.stanford.edu/projects/glove/

[1] Zhang, Y., Liang, P., & Wainwright, M. J. (2016). Convexified Convolutional Neural Networks. 1–29. https://doi.org/10.1145/2951024

[2] B. Paranjape, \Stop clickbait: Github dataset," supplement to Chakraborty et al. (2016) - Stop Clickbait. [Online]. Available: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset

[3] A. Chakraborty, B. Paranjape, S. Kakarla, and N. Ganguly, Stop clickbait: Detecting and preventing clickbaits in online news media," in Advances in Social Networks Analysis and Mining (ASONAM), 2016 IEEE/ACM International Conference on. IEEE, 2016, pp. 9-16.
