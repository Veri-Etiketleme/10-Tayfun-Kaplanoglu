# Microsoft Azure Captsone Project: Classify Malicious and Benign Websites

## Project Abstract: 

Over the past few years, Web Security has become a growing concern for internet users and organizations asthey rely on multiple web-sites for their daily tasks like shopping, banking, information retrival etc. With growing use of internet, number of malicious websites have also grown exponentially, developed by attackers with an intention to breach individual privacy and steal data to use it for fraudulant activities. This act of creating fake websites, which are in many cases imitating real organizations is a serious concern for many due to increasing number of scams using stolen identity and data theft. Internet users and many organizations have been a victim of phishing and other internat crimes, simply because no accurate classificaton can be obrained between malicious and benign websites simply by viewing content of the website.

A log term-goal of this project will be to develop a real-time system that uses Machine Learning techniques to detect malicious URLs (spam, phishing, exploits, etc.). Techniques explored involving classifying URLs based on their **Lexical** and **Host-based** features, as well as online learning to process large numbers of examples and adapt quickly to evolving URLs over time are captured from research published by PhD scholars from UC San Diago [1] & [2]. This project aims to extend research finding and construct a classifier which can predict malicious and benign website URLs, based on application layer and network characteristics utilizing captured data for this research work.

## Dataset:

### Dataset Overview:

For this capstone project, we have used [Malicious and Benign Websites](https://www.kaggle.com/xwolf12/malicious-and-benign-websites) dataset. Dataset is created with data obtained from verified sources of benign and malicious URL's in a low interactive client honeypot to isolate network traffic. To study malicious website, application and network layer features are identified, which are listed under file strucutre. There are 1781 unique URL records with 21 different featurs. Here, we will not include Web page content or the context of the URL as a feature to avoid downloading content as classifying URL with a trained odel is a lightweight operation compared to first downloading the page content and then analyzing them.  Appraoch with this dataset is to classify URLs without looking at the content of the website as malicious site may serve benign versions of a page to honeypot IP address run by a security practitioner, but serve alicious version to other user due to content "cloaking". Type of features are as follows:

1. **Lexical Features:** These features allow us to capture the property that malicious URLs tend to hold different from benign URLs, for example URL domain names, keywords and lengths of the hostname (phish.biz/www.indeed.com/index.php or www.craigslist.com.phishy.biz)

2. **Host-based features:** These features allow us to identify where malicious website is hosted from, who owns them, and how they are managed.

	- **WHOIS Information:** This includes domain name registration dates, registrars, and registrants. Using this feature, we can tag all the websites as malicious registered by the same individual and such ownership as a malicious feature.

	- **Location:** his refers to the host's geo-location, IP Address prefix, and autonomous system (AS) number. So websites hosted in a specific IP prefix of an Internet Service Provider can be tagged as a malicious website and account for such host can be classified disreputable ISP when classifying URLs.
	
	- **Connection Speed:** Connection speed of some malicious websites residing on compormised residential machines.
	
	- **Other DNS related properties:** These include time-to-live (TTL), spam-related domain name heuristics, and whether the DNS records share the same ISP. 

Below are the resources used to create this dataset.

- machinelearning.inginf.units.it/data-andtools/hidden-fraudulent-urls-dataset
- malwaredomainlist.com
- https://github.com/mitchellkrogza/The-Big-List-of-Hacked-Malware-Web-Sites

### File Structure:

Column Names | Details
------------ | -------------
`URL` | It is the anonimous identification of the URL analyzed in the study.
`URL_LENGTH` | It is the number of characters in the URL.
`NUMBER_SPECIAL_CHARACTERS` | It is number of special characters identified in the URL, such as, “/”, “%”, “#”, “&”, “. “, “=”.
`CHARSET` | It is a categorical value and its meaning is the character encoding standard (also called character set).
`SERVER` | It is a categorical value and its meaning is the operative system of the server got from the packet response.
`CONTENT_LENGTH` | It represents the content size of the HTTP header.
`WHOIS_COUNTRY` | It is a categorical variable, its values are the countries we got from the server response (specifically, our script used the API of Whois).
`WHOIS_STATEPRO` | It is a categorical variable, its values are the states we got from the server response (specifically, our script used the API of Whois).
`WHOIS_REGDATE` | Whois provides the server registration date, so, this variable has date values with format DD/MM/YYY HH:MM
`WHOIS_UPDATEDDATE` | Through the Whois we got the last update date from the server analyzed
`TCP_CONVERSATION_EXCHANGE` | This variable is the number of TCP packets exchanged between the server and our honeypot client
`DIST_REMOTETCP_PORT` | It is the number of the ports detected and different to TCP
`REMOTE_IPS` | This variable has the total number of IPs connected to the honeypot
`APP_BYTES` | This is the number of bytes transfered
`SOURCE_APP_PACKETS` | Packets sent from the honeypot to the server
`REMOTE_APP_PACKETS` | Packets received from the server
`APP_PACKETS` | This is the total number of IP packets generated during the communication between the honeypot and the server
`DNS_QUERY_TIMES` | This is the number of DNS packets generated during the communication between the honeypot and the server
`TYPE` | This is a categorical variable, its values represent the type of web page analyzed, specifically, 1 is for malicious websites and 0 is for benign websites

### Task:

In this capstone project, we aim to create a model to classify if a website URL is `Malicious` or `Benign` with the use of dataset features explaned above. To achieve this, we will be using two approaches and compare both using `Accuracy` as a Primary Metric.

1. **Using [AutomatedML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)** With this approach, we will be using Microsoft Azure's Automated ML feature to train and tune a model for given dataset to predict which category (Maclicious or Benign) new URL will fall into based on learnings from it's training data. In this approach, Azure Machine Learning taking user inputs such as `Dataset`, `Target Metric` and `Constraints` into account, train model into multiple iterations and will return best performing model with highest training score achieved.

2. **Using [HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py):** With this approach, we will train a Scikit-learn Logistic Regression model and automating hyperparaeter tuning by using Azure ML's Hyperdrive package. By defining hyperparameter space, we will tune model applying different combinations of hyperparameters and tuning it untill we find the best performing model. Here, models will be compared on primary metrics defined. Unlike AutoML, with this approach we will need to manually perform feature scaling, normalization and other data preprocessing on our dataset to reduce overfitting and effect of bad data on model performance.

Post model training using both the approches, we will be comparing performance of both the models using performance metrics **Accuracy** and best performing model will be deployed on **Azure Container Instance (ACI)** as a web service registered with Azure workspace and open to consume by external services with provided autorization. Finally, functionality of the deployed model will be demonstrated using response received for each successful HTTP POST Request to an end-point for real-time inferencing.

### Access:

Azure mainly supports two types of Datasets: **A. FileDataset B. TabularDataset**. Here, we have data captured in **csv file**, which can be handled using **TabularDataset** as it is used for tabular data. Dataset is uploaded to [github repository](https://github.com/Panth-Shah/nd00333-capstone/blob/master/Dataset/malicious_website_dataset.csv), which is later used to register datastore with Azure ML Workspace using `Dataset.Tabular.from_delimited_files()`. We can also creare a new TabularDataset by directly calling the corresponding factory methods of the class defined in `TabularDatasetFactory`.
	
```python
# Create AML Dataset and register it into Workspace
example_data = 'https://raw.githubusercontent.com/Panth-Shah/nd00333-capstone/master/Dataset/malicious_website_dataset.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)
# Create TabularDataset using TabularDatasetFactory
dataset = TabularDatasetFactory.from_delimited_files(path=example_data)
#Register Dataset in Workspace
dataset = dataset.register(workspace=ws, name=key, description=description_text)
```

## Automated ML

Azure Automated Machine Learning (`AutoML`) provides capabilities to automate iterative tasks of machine learning model development for given dataset to predict which category (Maclicious or Benign) new URL will fall into based on learnings from it's training data. In this approach, Azure Machine Learning taking user inputs such as `Dataset`, `Target Metric` and `Constraints` into account, train model into multiple iterations and will return best performing model with highest training score achieved. We will train and tune a model using the `Accuray` primary metric for this project.

### AutoMLConfig:

This class from Azure ML Python SDK represents configuration to submit an automated ML experiment in Azure ML. Configuration parameters used for this project includes:

Configration | Details | Value
------------ | ------------- | -------------
`compute_target` | Azure ML compute target to run the AutoML experiment on | autoML-compute
`task` | The type of task task to run, set as classification | classification
`training_data` | The training data to be used within the experiment contains training feature and a label column | Tabular Dataset
`label_column_name`	| The nae of the label column | 'Type'
`path` | The full path to the Azure ML project folder	| './capstone-project'
`enable_early_stopping` | Enable AutoML to stop jobs that are not performing well after a minimum number of iterations	| True
`featurization` | Config indicator for whether featurization step should be done autometically or not	| auto
`debug_log ` | The log file to write debug information to | 'automl_errors.log'
`enable_onnx_compatible_models` | Whether to enable or disable enforcing the Open Neural Network Exchange (ONNX)-compatible models |   True

Also AutoML settings were as follows:

Configration | Details | Value
------------ | ------------- | -------------
`experiment_timeout_minutes` | Maximum amount of time in hours that all iterations combined can take before the experiment terminates | 30
`max_concurrent_iterations` | Represents the maximum number of iterations that would be executed in parallel | 4
`primary_metric` | The metric that the AutoML will optimize for model selection | Accuracy
`n_cross_validations` | Number of cross validations to perform when user validation data is not specified |  5
`max_cores_per_iteration` | The maximum number of threads to use for a given training iteration. -1 indicate use of all possible cores per iteration per child-run	| -1

`Data Featurization` is an important process while training model, as features that best characterize the pattern in the data should be selected to create predictive models. Feature engineering is a process of creating additional features that provide information that better differentiates patterns in the data. In AzureML, data-scaling and normalization techniques are applied to make feature engineering easier indicated as `featurization` in automated ML experiments. 
Specifying `"featurization": 'auto'` enables autometic preprocessing of data which includes applying data guardrails and featurization steps like **Drop High Cardinality, Impute missing values, Generate more features, Transform and encode, Word embedding, Cluster Distance etc.**

### Results

- Among all the models trained by AutoML, `Voting Ensemble` outperformed all the other models with `97.249% accuracy`.

	- Ensemble models in Automated ML are combination of multiple iterations which may provide better predictions compared to a single iteration and appear as the final iterations of run.
	- Two types of ensemble methods for combining models: **Voting** and **Stacking**
	- Voting ensemble model predicts based on the weighted average of predicted class probabilities.
	- In our project, combined models by Voting ensemble with their selected hyperparameters are as follows.

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Widget.PNG)

Figure 1. Python SDK Notebook - AutoML Run Details widget

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Widget_Plot.PNG)

Figure 2. Python SDK Notebook - Accuracy plot using AutoML Run Details widget

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_RunMetric_Notebook.PNG)

Figure 3. Python SDK Notebook - Best performing run details

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Completed.PNG)

Figure 4. Azure ML Studio - AutoML experiment completed

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_BestPerformingModel.PNG)

Figure 5. Azure ML Studio - AutoML best perforing model summary

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Metric.PNG)

Figure 6. Azure ML Studio - Performance Metrics of best performing model trained by AutoML


![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Models.PNG)

Figure 7. Azure ML Studio - Models trained in multiple iterations using AutoML

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_RegModelDetail.PNG)

Figure 8. Azure ML Studio - AutoML best performing model register to workspace

### Hyperparameters generated for models ensembled in Voting Ensemble:
<details>
  <summary>Click to expand!</summary>

	datatransformer
	{'enable_dnn': None,
	 'enable_feature_sweeping': None,
	 'feature_sweeping_config': None,
	 'feature_sweeping_timeout': None,
	 'featurization_config': None,
	 'force_text_dnn': None,
	 'is_cross_validation': None,
	 'is_onnx_compatible': None,
	 'logger': None,
	 'observer': None,
	 'task': None,
	 'working_dir': None}

	prefittedsoftvotingclassifier
	{'estimators': ['1', '6', '32', '0', '8', '4', '33', '19', '12', '13'],
	 'weights': [0.07142857142857142,
				 0.14285714285714285,
				 0.07142857142857142,
				 0.21428571428571427,
				 0.07142857142857142,
				 0.07142857142857142,
				 0.14285714285714285,
				 0.07142857142857142,
				 0.07142857142857142,
				 0.07142857142857142]}

	1 - maxabsscaler
	{'copy': True}

	1 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 1,
	 'gamma': 0,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 3,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'binary:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 1,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	6 - maxabsscaler
	{'copy': True}

	6 - lightgbmclassifier
	{'boosting_type': 'gbdt',
	 'class_weight': None,
	 'colsample_bytree': 0.1,
	 'importance_type': 'split',
	 'learning_rate': 0.015797894736842105,
	 'max_bin': 110,
	 'max_depth': 8,
	 'min_child_samples': 5,
	 'min_child_weight': 1,
	 'min_split_gain': 0.3684210526315789,
	 'n_estimators': 600,
	 'n_jobs': 1,
	 'num_leaves': 137,
	 'objective': None,
	 'random_state': None,
	 'reg_alpha': 0.8421052631578947,
	 'reg_lambda': 1,
	 'silent': True,
	 'subsample': 0.5942105263157895,
	 'subsample_for_bin': 200000,
	 'subsample_freq': 0,
	 'verbose': -10}

	32 - maxabsscaler
	{'copy': True}

	32 - lightgbmclassifier
	{'boosting_type': 'gbdt',
	 'class_weight': None,
	 'colsample_bytree': 0.2977777777777778,
	 'importance_type': 'split',
	 'learning_rate': 0.0842121052631579,
	 'max_bin': 50,
	 'max_depth': -1,
	 'min_child_samples': 5,
	 'min_child_weight': 8,
	 'min_split_gain': 0.8421052631578947,
	 'n_estimators': 400,
	 'n_jobs': 1,
	 'num_leaves': 65,
	 'objective': None,
	 'random_state': None,
	 'reg_alpha': 0.7894736842105263,
	 'reg_lambda': 0.7368421052631579,
	 'silent': True,
	 'subsample': 0.7426315789473684,
	 'subsample_for_bin': 200000,
	 'subsample_freq': 0,
	 'verbose': -10}

	0 - maxabsscaler
	{'copy': True}

	0 - lightgbmclassifier
	{'boosting_type': 'gbdt',
	 'class_weight': None,
	 'colsample_bytree': 1.0,
	 'importance_type': 'split',
	 'learning_rate': 0.1,
	 'max_depth': -1,
	 'min_child_samples': 20,
	 'min_child_weight': 0.001,
	 'min_split_gain': 0.0,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'num_leaves': 31,
	 'objective': None,
	 'random_state': None,
	 'reg_alpha': 0.0,
	 'reg_lambda': 0.0,
	 'silent': True,
	 'subsample': 1.0,
	 'subsample_for_bin': 200000,
	 'subsample_freq': 0,
	 'verbose': -10}

	8 - standardscalerwrapper
	{'class_name': 'StandardScaler',
	 'copy': True,
	 'module_name': 'sklearn.preprocessing._data',
	 'with_mean': False,
	 'with_std': False}

	8 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 1,
	 'eta': 0.001,
	 'gamma': 0,
	 'grow_policy': 'lossguide',
	 'learning_rate': 0.1,
	 'max_bin': 1023,
	 'max_delta_step': 0,
	 'max_depth': 2,
	 'max_leaves': 0,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 1.5625,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 0.5,
	 'tree_method': 'hist',
	 'verbose': -10,
	 'verbosity': 0}

	4 - maxabsscaler
	{'copy': True}

	4 - randomforestclassifier
	{'bootstrap': True,
	 'ccp_alpha': 0.0,
	 'class_weight': 'balanced',
	 'criterion': 'gini',
	 'max_depth': None,
	 'max_features': 'log2',
	 'max_leaf_nodes': None,
	 'max_samples': None,
	 'min_impurity_decrease': 0.0,
	 'min_impurity_split': None,
	 'min_samples_leaf': 0.01,
	 'min_samples_split': 0.01,
	 'min_weight_fraction_leaf': 0.0,
	 'n_estimators': 25,
	 'n_jobs': 1,
	 'oob_score': True,
	 'random_state': None,
	 'verbose': 0,
	 'warm_start': False}

	33 - sparsenormalizer
	{'copy': True, 'norm': 'l1'}

	33 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 0.7,
	 'eta': 0.01,
	 'gamma': 0.1,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 5,
	 'max_leaves': 15,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 1.5625,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	19 - sparsenormalizer
	{'copy': True, 'norm': 'max'}

	19 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 0.5,
	 'colsample_bynode': 1,
	 'colsample_bytree': 1,
	 'eta': 0.3,
	 'gamma': 0,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 8,
	 'max_leaves': 0,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 2.1875,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	12 - maxabsscaler
	{'copy': True}

	12 - logisticregression
	{'C': 16.768329368110066,
	 'class_weight': 'balanced',
	 'dual': False,
	 'fit_intercept': True,
	 'intercept_scaling': 1,
	 'l1_ratio': None,
	 'max_iter': 100,
	 'multi_class': 'ovr',
	 'n_jobs': 1,
	 'penalty': 'l2',
	 'random_state': None,
	 'solver': 'saga',
	 'tol': 0.0001,
	 'verbose': 0,
	 'warm_start': False}

	13 - maxabsscaler
	{'copy': True}

	13 - logisticregression
	{'C': 51.79474679231202,
	 'class_weight': 'balanced',
	 'dual': False,
	 'fit_intercept': True,
	 'intercept_scaling': 1,
	 'l1_ratio': None,
	 'max_iter': 100,
	 'multi_class': 'multinomial',
	 'n_jobs': 1,
	 'penalty': 'l1',
	 'random_state': None,
	 'solver': 'saga',
	 'tol': 0.0001,
	 'verbose': 0,
	 'warm_start': False}
</details>

## Hyperparameter Tuning

In this section of the project, we will be training and tuning classifier using Azure ML's `HyperDrive package`. Here, we will train a model `Logisitc Regression` classification algorithm using AzureML python SDK with Scikit-learn to perform classification on Website URL dataset and classify URLs into malicious and benign categories.  This problem demands binary classification and as there are only two categories, this makes classification task perfect for logistic regression. It also is very fast at classifying unknown records and is easy to implement, intepret, and very efficient to train. Also for this project, as target variable falls into discrete categories, logistic regression is an ideal choice.

Steps required to tune hyperparameters using Azure ML's `HyperDrive package`;

1. Define the parameter search space using `Random Sampling` 
2. Specify a `Accuracy` as a primary metric to optimize
3. Specify early `Bendit Policy` as early termination policy for low-performing runs
4. Allocate `aml compute` resources
5. Launch an experiment with the defined configuration using `HyperDriveConfig`
6. Visualize the training runs with `RunDetails` Notebook widget
7. Select the best configuration for your model with `hyperdrive_run.get_best_run_by_primary_metric()`

### Type of parameters and sampling the Hyperparameter Space:

- `Regularization Strength (--C)`: This parameter is used to apply penalty on magnitude of parameters to reduce overfitting higher values of C correspond to less regularizationand and vice versa. Smaller values cause stronger regularization.

- `Max Iterations (--max_iter)`: Maximum number of iterations to converge to a minima.

To define random sampling over the search space of hyperparameter we are trying to optimize, we are using AzureML's `RandomParameterSampling` class. Levaraging this method of parameter sampling, users can randomly select hyperparameter from defined search space. With this sampling algorithm, AzureML lets users choose hyperparameter values from a set of discrete values or a distribution over a continuous range. This method also supports early termination of low performance runs, which is a cost effcient approach when training model on aml compute cluster.

Other two approaches supported by AzureML are Grid Sampling and Bayesian Sampling:

- As `Grid Sampling` only supports discrete hypeparameter, it searches over all the possibilities from defined search space. And so more compute resource is required, which is not very budget efficient fo this project. 

- `Bayesian Sampling` method is based on Bayesian optimization algorithm and picks samples based on how previous samples performed to improve the primary metric of new samples. Because of that, more number of runs benefit future selection of samples, which also is not a very cost efficient solution for this project. 

```python
# Specify parameter sampler
ps = RandomParameterSampling(
    {
        "--C": uniform(0.01, 100), 
        "max_iter": choice(16, 32, 64, 128, 256)
    }
)
```

### Advantages of Early Stopping Policy:

While working with Azure's managed aml compute cluster to train classification model for this project, it is important to maintain and imporve computational effciency.Specifying early termination policy autometically terminates poorly performing runs based on configuration parameters (`evaluation_interval`, `delay_evaluation`) provided upon defining these policies. These can be applied to HyperDrive runs and run is cancelled when the criteria of a specified policy are met.

`Bendit Policy`:

Among supported early termination policies by Azure ML, we are using Bendit Policy in this project. This policy is based on slack criteria, and a frequency and delay interval for evaluation. BenditPolicy determines best performing run based on selected primary metric (Accuracy for this project) and sets it as a benchmark to compare it against other runs. `slack_factor`/`slack_amount` configuration parameter is used to specify slack allowed with respect to best performing run. `evaluation_interval` specifies frequency of applying policy and `delay_evaluation` specifies number of intervals to delay policy evaluation for run termination. This parameter ensures protection against premature termination of training run.


### Results

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Hyperdrive_RunCopleted.PNG)

Figure 9. Azure ML Studio Experiment submitted with HyperDrive from notebook

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Hyperdrive_Running_2.PNG)

Figure 10. Python SDK Notebook: Monitor progress of run using Run Details widget

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Hyperdrive_BestModel_Notebook.PNG)

Figure 11. Python SDK Notebook: Best performing model from hyperparameter tuning using HyperDrive

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Hyperdrive_BestModel_Metric.PNG)

Figure 12. Python SDK Notebook: HyperDrive Run Primary Metric Plot - Accuracy

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Hyperdrive_BestModel_Metric2.PNG)

Figure 13. Python SDK Notebook: Plot displaying `C` and `max-itr` hyperparmeter values selected for all the child runs in an experiment 

## Model Improvment:

- According to Azure AutoML's Data Guardrails analysis, **class immbalance** is detected in the provided dataset for this project. Here, class distribution of sample space in the training dataset is severly disproportionated. Because input data has a bias towards one class, this can lead to a falsely perceived positive effect of a model's accuracy.
To improve accuracy of the prediction model, will use synthetic sampling techniques like `SMOTE`, `MSMOTE` and other ensemble techniques to increase the frequency of the minority class or decrease the frequncy of the majority class.
	- **How upsampling improves performance of the model:** Using oversampling techniques like SMOTE, minority class is over-sampled by taking each minority class sample and introducing synthetic examples to create large and less specific decision boundaries that increase the generalization capabilities of classifiers.

- Avoiding misleading data in order to imporve the performance of our prediction model is a critical step as irrelevant attributes in your dataset can result into overfitting. As a future enhancement of this project, leveraging **Automated ML's Model Interpretability** dashboard, will inspect which dataset feature is essential and used by model to make its predictions and determine best and worst performing features to include/exclude them from future runs. Based on this finding, will customize featurization settings used by Azure AutoML to train our model. `FeaturizationConfig` defines feature engineering configuration for our automated ML experiment, using which we will exclude irrelevent features identified from AutoML's model interpretability dashboard. While training SKLearn Logistic Regression classification model, **[Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)** can also be used to rank the feature and recursively reducing the sets of features.
	- **How feature selection improves performance of the model:** In classification models like logistic regression which can model binary variables, the information value is critical to understand how important given categorical variale is in explaning the binary target variable. An iterative process of dividing features into subsets based on weights of evidence will help us achieve higher AUC and we can expect evolution in the True Positive rate and the ROC metric for each feature selection iteration.

- With current project, we are using only two hyperparameters `C` and `max-itr` to train Logistic Regression model. Adding additional parameters like `penalty`, `class_weight`, `intercept_scaling`, `n_jobs`, `l1_ratio` etc. will allow us to control training model and performance of our classifier can be improvised.
	- **How choosing more hyperparameters improves performance of the model:** More number of parameters to tune classifier will increase the hyperparameter search space to explore and better accuracy can be achieved with different combinations of parameters applying sampling algorithms. 

- Due to class imbalance problem we have with the given data set, it is possible that model will always only predict class which has higher % instances in the dataset. This results into excellent classification accuracy as it only reflects the underlying class distribution. This situation is called **Accuracy Paradox**, where accuracy is not the best metric to use for performance evaluation of prediction model and can be misleading. As a future improvements of this model, will use additional measures such as **Precision, Recall, F1 Score** etc. to evaluate a trained classifier.
	- **How use of additional performance metrics help improve performance of the model:** With highly imbalance dataset, chances while performing k-fold cross validation procedure in the training set is that single fold may not contain a positive sample, which results into True Positive Rate (TPR) and False Negtive Rate (FNR) to 0. We will choose ROC AUC over Accuracy as plot from the ROC curve will help understand trade-off in performance for different threshold values when interpreting probabilistic predictions. Changing the threshold of classification will change the balance of predictions towards improving the TPR at the expense of FPR and vice versa and so ROC analysis doesn't have any bias towards models which performs well on either minority or majority class, which is a helpful metric to deal with imbalance dataset.

- With more compute resources in hand for future experiments, will perform parameter sampling over the search space of hyperparameter for **HyperDriveConfig** using **Bayesian Sampling** technique. To obtain better results, will go with Azure ML's recommended approach by maximizing number of runs greater than or equal to 20 times the number of hyperparameters being tuned using this sampling technique. Will also keep number of concurrent runs lower, which will lead us to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit parameter tuning process by taking reference from previously completed runs.
	- **How Bayesian Sampling will improve performance of the model:** Bayesian sampling over hyperparameter search space will be advantageous in improving our model performance iteratively as this technique intelligently picks the next sample of hyperparameter based on how previous sample performed with the aim to improve the primary metric determined. 

## Model Deployment

From both the approaches, `Voting Ensemble` is the best performing model obtained using `AutoML` experiment. Now, we will need to deploy this model as a HTTP web service in Azure Cloud. Below are the steps involved in deployment workflow for our model:

1. Register the model with Azure ML Workspace
2. Prepare inference configuration by providing entry script to receive data submitted to a deployed web service
3. Choose a compute target as `Azure Container Instance` to deploy model
4. Define deployment configuration using `AciWebservice.deploy_configuration()`
5. Deploy best perforing model using `Model.deploy()`

Configuration object created for deploying an AciWebservice used for this project is as follows:

	aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
			   memory_gb = 1, 
			   tags = {'area': "bmData", 'type': "capstone_autoML_Classifier"}, 
			   description = 'sample service for Capstone Project AutoML Classifier for Websites')

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_Deploy_Scorefile.PNG)

Figure 14. Entry script `scoring_file_v_1_0_0.py` located from the project folder 

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/Results/AutoML_DeployedModel.PNG)

Figure 15. Azure ML Studio: Deployed best performing model 

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/AutoML_DeployedModel_Notebook.PNG)

Figure 16. Python SDK Notebook: Deployment Completed 


![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/Results/AutoML_Deployed_ModelTest.PNG)

Figure 17. Python SDK Notebook: Best performing model test


## Screen Recording
Screen Recording with detailed explanation uploaded and can be found using [Link](https://youtu.be/oWLvQHeSA8Y). This screencast demonstrates:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions

### Enable Logging in your deployed web app

With this project, we have deployed best performing model to HTTP web service endpoints in `Azure Container Instance (ACI)`. To enable collecting additional data from an endpoint mentioned below, we will be **enabling** [Azure Application Insight](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview) feature, an extensible Application Performance Management (APM) service.

- Output Data
- Responses
- Request rates, response times, and failure rates
- Dependency rates, response times, and failure rates
- Exceptions for failed requests

To perform this step, we have used `logs.py` script uploaded with this repository. We will dynamically authenticate to Azure, enable Application Insight and Display logs for deployed model.

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/Deploy_ScriptEnableAppInsight.PNG)

Figure 17. logs.py script to enable Application Insight for deployed model

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/EnableLogging.PNG)

Figure 18. Result - Enable Application Insight using logs.py

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/EnableLogging_2.PNG)

Figure 19. Result - Enable Application Insight using logs.py

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/ApplicationInsight_Studio.PNG)

Figure 20. Azure ML Studio: Application Insight enabled for deployed service

![Alt Text](https://github.com/Panth-Shah/nd00333-capstone/blob/master/ExperimentResults/ApplicationInsight_Dashboard.PNG)

Figure 21. MS Azure: Application Insight dashboard for deployed web service
