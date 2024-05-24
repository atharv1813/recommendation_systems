
# Wide and Deep Autoencoders for Recommender Systems

## Introduction
In this project i have proposed a novel recommendation system that incorporates advanced techniques such as deep learning and autoencoders. Specifically, it combines wide and deep models(existing model) with autoencoders on its deep part to improve recommendation accuracy.

## Literature Review
The literature review covers various unsupervised and supervised collaborative filtering techniques for recommender systems:

### Unsupervised Collaborative Filtering
- User and Item-based Similarity
- Matrix Factorization
- Probabilistic Matrix Factorization 
- Blind Compressed Sensing
- Matrix Completion

### Supervised Collaborative Filtering
- Supervised Matrix Factorization

## Proposed Method
The proposed method is based on the Wide & Deep architecture from the paper "Wide & Deep Learning for Recommender Systems" by Cheng et al. (2016).

### Wide Component
The wide component is a generalized linear model that uses cross-product feature transformations.

### Deep Component 
The deep component is a feed-forward neural network that uses embeddings for categorical features.

### Wide and Deep Model
The wide and deep components are combined using a weighted sum, and the model is trained end-to-end.

### Improvements Using Autoencoders
To handle sparse input features in the deep component, autoencoders are used to learn dense representations of the sparse features. This encoding step is applied only to categorical features, while numerical features remain unchanged.

## Experiments and Results
The proposed method is evaluated on the MovieLens-100k dataset. Experiments are conducted with and without autoencoders, and the results show that incorporating autoencoders improves the accuracy and relevance of recommendations.

## Conclusion
The integration of autoencoders allows the deep component to benefit from dense representations while maintaining the simplicity and interpretability of the wide component. This hybrid approach leverages the strengths of both components, resulting in improved performance in predicting movie ratings.
