# Language Detector using NLP
A Natural Language Processing (NLP) ML model to identify the Language of the given text. This model uses Naive Bayes for making predictions. It can detect upto 17 different languages. We are also using count vectorizer and label encoders.
Languages supported are:
- English
- Portuguese
- French
- Greek
- Dutch
- Spanish
- Japanese
- Russian
- Danish
- Italian
- Turkish
- Swedish
- Arabic
- Malayalam
- Hindi
- Tamil
- Telugu

**Count Vectorizer** : It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text. This is helpful when we have multiple such texts, and we wish to convert each word in each text into vectors (for using in further text analysis).

**Label Encoders** : Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.

**Naive Bayes** : Naive Bayes is a statistical classification technique based on Bayes Theorem. It is one of the simplest supervised learning algorithms. Naive Bayes classifier is the fast, accurate and reliable algorithm. Naive Bayes classifiers have high accuracy and speed on large datasets.
Naive Bayes classifier assumes that the effect of a particular feature in a class is independent of other features. For example, a loan applicant is desirable or not depending on his/her income, previous loan and transaction history, age, and location. Even if these features are interdependent, these features are still considered independently. This assumption simplifies computation, and that's why it is considered as naive. This assumption is called class conditional independence.


#### For complete details on how to create the Model, check out my [Blog](https://sharadmittal.hashnode.dev/language-detector-using-nlp)

#### For complete details on how to create the Web app using Streamlit library, check out my [Blog](https://sharadmittal.hashnode.dev/creating-a-web-app-of-a-ml-model-using-streamlitp)

#### For complete details on how to deploy the streamlit web app on Heroku, check out my [Blog](https://sharadmittal.hashnode.dev/language-detector-using-nlp)
