Identity resolution methods (also known as data matching or record linkage methods) identify records that describe the same real-world entity. 

## Rule-based Identity Resolution

### General Process

1.	First we load the two data sets

```java
// loading data
HashedDataSet<Movie, Attribute> dataAcademyAwards = new HashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("academy_awards.xml"), "/movies/movie", dataAcademyAwards);
HashedDataSet<Movie, Attribute> dataActors = new HashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("actors.xml"), "/movies/movie", dataActors);
```

2.	Then we define a matching rule that compares the records. We compare the movie title with Jaccard similarity and the release date with a custom date similarity function. Then we use a linear combination of the title similarity with a weight of 80% and the release date with a weight of 20% and a final similarity threshold of 70%.

```java
// create a matching rule
LinearCombinationMatchingRule<Movie, Attribute> matchingRule = new LinearCombinationMatchingRule<>(0.7);
// add comparators
matchingRule.addComparator(
(m1,  m2, c) -> new TokenizingJaccardSimilarity().calculate(m1.getTitle(), m2.getTitle()), 0.8);
matchingRule.addComparator(
(m1, m2, c) -> new YearSimilarity(10).calculate(m1.getDate(), m2.getDate()), 0.2);
```

3.	To speed up the whole process, we only want to compare records that seem similar instead of comparing all records. Hence, we add a blocking strategy that only compares movies from the same decade.

```java
// create a blocker (blocking strategy)
Blocker<Movie, Attribute> blocker = new StandardBlocker<Movie, Attribute>(
(m) -> Integer.toString(m.getDate().getYear() / 10));
```

4.	Finally, we initialise the matching engine, which does all the work for us, and run the identity resolution implementation with our matching rule.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

// Execute the matching
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runIdentityResolution(dataAcademyAwards, dataActors, null, matchingRule, blocker);
```

5.	To see how good our result is, we apply the built-in evaluation methods.

```java
// load the gold standard (test set)
MatchingGoldStandard gsTest = new MatchingGoldStandard();
gsTest.loadFromCSVFile(new File("gs_academy_awards_2_actors_v2.csv"));

// evaluate the result
MatchingEvaluator<Movie, Attribute> evaluator = new MatchingEvaluator<Movie, Attribute>();
Performance perfTest = evaluator.evaluateMatching(correspondences.get(),gsTest);

// print the evaluation result
System.out.println("Academy Awards <-> Actors");
System.out.println(String.format("Precision: %.4f\nRecall: %.4f\nF1: %.4f",
	perfTest.getPrecision(), perfTest.getRecall(),perfTest.getF1()));
```

### Matching Rules

Winter provides various pre-implemented Matching Rules.

#### Linear Combination Matching Rule

The [`LinearCombinationMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRule.java) uses a weighted linear combination of the attribute similarities for the matching.
If the aggregated similarity of the presented correspondence exceeds the defined threshold, the correspondence is seen as a match.
To calculate the attribute similarities, comparators are added to the matching rule along with an associated weight.

```java
// create a matching rule
LinearCombinationMatchingRule<Movie, Attribute> matchingRule = new LinearCombinationMatchingRule<>(0.7);
// add comparators
matchingRule.addComparator(new MovieDateComparator10Years(), 0.2);
matchingRule.addComparator(new MovieTitleComparatorLevenshtein(), 0.8);
```

#### Linear Combination Matching Rule with Penalty

The [`LinearCombinationMatchingRuleWithPenalty`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRuleWithPenalty.java) uses a weighted linear combination of the attribute similarities for the matching.
If the aggregated similarity of the presented correspondence exceeds the defined threshold, the correspondence is seen as a match.
To calculate the attribute similarities, comparators are added to the matching rule along with an associated weight.
The [`LinearCombinationMatchingRuleWithPenalty`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRuleWithPenalty.java) can deal with missing values.
If a comparator reports a missing value to the matching rule, the weights of this comparator are redistributed to all other comparators and the aggregated similarity is reduced by a defined penalty. The penalty accounts for the higher uncertainty of the matching decision, as it is based on a smaller number of attributes due to the missing value. This penalty is assigned to the comparator, when the comparator is added to the matching rule.

```java
// create a matching rule
LinearCombinationMatchingRuleWithPenalty<Movie, Attribute> matchingRule = new LinearCombinationMatchingRuleWithPenalty<>(0.7);
// add comparators
matchingRule.addComparator(new MovieDirectorComparatorMissingValueLevenshtein(), 0.2, 0.15);
matchingRule.addComparator(new MovieTitleComparatorLevenshtein(), 0.8, 0.0);
```

#### Learnable Weka Matching Rule

The [`WekaMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/WekaMatchingRule.java) learns a matching rule based on the attribute similarities.
For the attribute similarities various comparators are added to the matching rule.
Among these attribute similarities a classifier learns to choose the relevant ones for matching.
The learning algorithms used inside the matching rule are provided by [WEKA](http://www.cs.waikato.ac.nz/ml/weka/index.html).
Please refer to [Learning Matching Rules](https://github.com/olehmberg/winter/wiki/Learning-Matching-Rules) for a detailed description of the learning process.

```java
String options[] = new String[1];
options[0] = "";
String logisticRegression = "SimpleLogistic";
WekaMatchingRule<Movie, Attribute> matchingRule = new WekaMatchingRule<>(0.7, logisticRegression, options);

// add comparators
matchingRule.addComparator(new MovieTitleComparatorEqual());
matchingRule.addComparator(new MovieDateComparator2Years());
matchingRule.addComparator(new MovieDateComparator10Years());
matchingRule.addComparator(new MovieDirectorComparatorJaccard());
matchingRule.addComparator(new MovieTitleComparatorLevenshtein());
matchingRule.addComparator(new MovieTitleComparatorJaccard());
```