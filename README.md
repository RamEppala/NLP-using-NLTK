# NLP-using-NLTK

Implementing NLP using NLTK python library

1.Text Preprocessing
  
  Noise Removal
  
Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise.

For example – language stopwords (commonly used words of a language – is, am, the, of, in etc), URLs or links, social media entities (mentions, hashtags), punctuations and industry specific words. 

Another approach is to use the regular expressions while dealing with special patterns of noise

  Lexicon Normalization
  
  Lemmatization:
  
  Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word.
  
  Stemming:
  
  Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
  
  Object Standardization:
  
  Text data often contains words or phrases which are not present in any standard lexical dictionaries. These pieces are not recognized by search engines and models.

Some of the examples are – acronyms, hashtags with attached words, and colloquial slangs. With the help of regular expressions and manually prepared data dictionaries, this type of noise can be fixed.

 2.Text to Features (Feature Engineering on text data)
 
  Syntactical Parsing:
    
 Syntactical parsing involves the analysis of words in the sentence for grammar and their arrangement in a manner that shows the         relationships among the words.
 
  Dependency Grammar:
  
  Dependency tree:
  
  This type of tree, when parsed recursively in top-down manner gives grammar relation triplets as output which can be used as features for many nlp problems like entity wise sentiment analysis, actor & entity identification, and text classification. The python wrapper StanfordCoreNLP (by Stanford NLP Group, only commercial license) and NLTK dependency grammars can be used to generate dependency trees.
  
  The relationship among the words in a sentence is determined by the basic dependency grammar. 
  
  Part of Speech Tagging:
  
  Apart from the grammar relations, every word in a sentence is also associated with a part of speech (pos) tag (nouns, verbs, adjectives, adverbs etc). The pos tags defines the usage and function of a word in the sentence. 
  
 3. Entity Extraction:
  
  Entities are defined as the most important chunks of a sentence – noun phrases, verb phrases or both. Entity Detection algorithms are generally ensemble models of rule based parsing, dictionary lookups, pos tagging and dependency parsing. The applicability of entity detection can be seen in the automated chat bots, content analyzers and consumer insights.

 Topic Modelling & Named Entity Recognition are the two key entity detection methods in NLP.
 A. Named Entity Recognition (NER)
The process of detecting the named entities such as person names, location names, company names etc from the text is called as NER.

B Topic Modeling
Topic modeling is a process of automatically identifying the topics present in a text corpus, it derives the hidden patterns among the words in the corpus in an unsupervised manner. Topics are defined as “a repeating pattern of co-occurring terms in a corpus”. A good topic model results in – “health”, “doctor”, “patient”, “hospital” for a topic – Healthcare, and “farm”, “crops”, “wheat” for a topic – “Farming”.
Latent Dirichlet Allocation (LDA) is the most popular topic modelling technique.

  N-Grams:
  
 A combination of N words together are called N-Grams. N grams (N > 1) are generally more informative as compared to words (Unigrams) as  features.
  
   4. Statistical features
      TF – IDF
      
      TF-IDF is a weighted model commonly used for information retrieval problems. It aims to convert the text documents into vector models on the basis of occurrence of words in the documents without taking considering the exact ordering.
      
      Frequency / Density Features
      Readability Features
      Word Embeddings:
      
      Word embedding is the modern way of representing words as vectors. The aim of word embedding is to redefine the high dimensional word features into low dimensional feature vectors by preserving the contextual similarity in the corpus. They are widely used in deep learning models such as Convolutional Neural Networks and Recurrent Neural Networks.

Word2Vec and GloVe are the two popular models to create word embedding of a text. These models takes a text corpus as input and produces the word vectors as output.
      
5. Important tasks of NLP

  Text Classification
  
  Text classification, in common words is defined as a technique to systematically classify a text object (document or sentence) in one of the fixed category. It is really helpful when the amount of data is too large, especially for organizing, information filtering, and storage purposes.
  
  A typical natural language classifier consists of two parts: (a) Training (b) Prediction as shown in image below. Firstly the text input is processes and features are created. The machine learning models then learn these features and is used for predicting against the new text.
  
  Text Matching:
  
 One of the important areas of NLP is the matching of text objects to find similarities. Important applications of text matching includes automatic spelling correction, data de-duplication and genome analysis etc.
  
  text matching techniques are:
  
  Levenshtein Distance:
  The Levenshtein distance between two strings is defined as the minimum number of edits needed to transform one string into the other, with the allowable edit operations being insertion, deletion, or substitution of a single character.
  
  Phonetic Matching
  
  A Phonetic matching algorithm takes a keyword as input (person’s name, location name etc) and produces a character string that identifies a set of words that are (roughly) phonetically similar
  
  Flexible String Matching
  
  cosine similarity
  
  When the text is represented as vector notation, a general cosine similarity can also be applied in order to measure vectorized similarity
