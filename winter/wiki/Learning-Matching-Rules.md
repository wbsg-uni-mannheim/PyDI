Instead of defining the matching rule by ourselves, we use machine learning to train a classifier, which matches the entries.


1.	After loading our data like for Rule based identity resolution, we also load a training set for our classifier.

```java
// load the gold standard (training set)
MatchingGoldStandard gsTraining = new MatchingGoldStandard();
gsTraining.loadFromCSVFile(new File("usecase/movie/goldstandard/gs_academy_awards_2_actors.csv"));
```

2.	In a next step the classifier needs to be choosen. Winter uses the machine learning algorithms provided by [WEKA](http://www.cs.waikato.ac.nz/ml/weka/index.html). Therefore a couple of variants exist to initialize a classifier for the matching rule. Please check out the [Weka classifier´s documentation](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html) to understand the classifier and the options you can choose from.

```java
// create a matching rule & provide classifier, options
String tree = "J48"; // new instance of tree
String options[] = new String[1];
options[0] = "-U";

WekaMatchingRule<Movie, Attribute> matchingRule = new WekaMatchingRule<>(0.8, tree, options);
```

3.	Additionally a forward or a backward selection can be applied to improve the feature subset selection. Both selections are performed inside a 10-fold cross-validation to provide solid results.

Forward selection

```java
matchingRule.setForwardSelection(true);
```

Backward selection

```java
matchingRule.setBackwardSelection(true);
```


4.	Then the compartors are selected, which provide the similarity values for a feature comparison vector.

```java
// add comparators
matchingRule.addComparator(new MovieTitleComparatorEqual());
matchingRule.addComparator(new MovieDateComparator2Years());
matchingRule.addComparator(new MovieDateComparator10Years());
matchingRule.addComparator(new MovieDirectorComparatorJaccard());		
```

Besides these dedicated Movie comparators, more general Record Comparators are available to provide more flexibility.

5.	After creating a blocker for the Rule based identity resolution the matching rule needs to be trained.

```java
// learning Matching rule
RuleLearner<Movie, Attribute> learner = new RuleLearner<>();
learner.learnMatchingRule(dataAcademyAwards, dataActors, null, matchingRule, gsTraining);
```


6.	This newly trained matching rule can be stored to reuse it.

```java
// Store Matching Rule
matchingRule.storeModel(new File("usecase/movie/output/model"));
```


7.	To reuse the model read from the file system instead of initializing and training a new matching rule. Please note that it is possible to load PMML based models as well as WEKA models.

```java
// Store Matching Rule
matchingRule.readModel(new File("usecase/movie/output/model"));
```

The steps Blocking, MatchingEngine Initialization and Evaluation are equivalent to the ones performed for the rule based approach.