This tutorial gives a step-by-step introduction to using the WInte.r framework for identity resolution and data fusion. 
The goal of identity resolution (also known as data matching or record linkage) is to identify records in different datasets that describe the same real-world entity. 
Data fusion methods merge all records which describe the same real-world entity into a single, consolidated record while resolving data conflicts.

The tutorial explains the usage of WInte.r along the use case of integrating data about movies. 
The goal is to integrate the two datasets [Actors](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/movie/input/actors.xml) and [Academy awards](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/movie/input/academy_awards.xml) into a single, duplicate-free dataset containing comprehensive descriptions of all movies. 
The complete source code of the tutorial is found in the folder [use case / movies](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies).

The tutorial is structured as follows:

1. [Running the Tutorial Code](#running-the-tutorial-code)
2. [Overview of the Datasets](#overview-of-the-datasets)
3. [Define Data Model and Load Data](#define-data-model-and-load-data)
4. [Identity Resolution](#identity-resolution)
    1. [Creating a Matching Rule](#creating-a-matching-rule)
    2. [Running the Identity Resolution](#running-the-identity-resolution)
	3. [Creating a Gold Standard for Identity Resolution](#creating-a-gold-standard-for-identity-resolution)
    4. [Evaluating the Matching Result](#evaluating-the-matching-result)
	5. [Adjusting the Matching Rule](#adjusting-the-matching-rule)
    6. [Learning a Matching Rule](#learning-a-matching-rule)
    7. [Using Blocking to Reduce the Runtime](#using-blocking-to-reduce-the-runtime)
5. [Data Fusion](#data-fusion)
    1. [Loading and Inspecting the Data and Correspondences](#loading-and-inspecting-the-data-and-correspondences)
    2. [Creating a Data Fusion Strategy](#creating-a-data-fusion-strategy)
    3. [Creating a Gold Standard for Data Fusion](#creating-a-gold-standard-for-data-fusion)
	4. [Evaluating the Data Fusion Strategy](#evaluating-the-data-fusion-strategy)
	5. [Adjusting the Data Fusion Strategy](#adjusting-the-data-fusion-strategy)


# Running the Tutorial Code

To run the tutorial code, first clone the repository:

```
git clone https://github.com/olehmberg/winter.git
```

Then use [Maven](http://maven.apache.org/) to build the framework and usecases projects:

```
winter/> cd winter-framework
winter/winter-framework/> mvn install
winter/winter-framework/> cd ../winter-usecases
winter/winter-usecases/> mvn install
```

You can execute each part of the tutorial individually by running the respective java class:

```
winter/winter-usecases/> java -cp target/winter-usecases-1.0-jar-with-dependencies.jar de.uni_mannheim.informatik.dws.winter.usecase.movies.Movies_Tutorial_IdentityResolution_Step01
```

# Overview of the Datasets

This tutorial assumes that all datasets have already been mapped into a single target schema. 
If this is not the case for your data, please refer to the wiki page on [schema matching](https://github.com/olehmberg/winter/wiki/SchemaMatching) which explains how WInte.r is used for finding mappings between schemata.

The example below shows a movie record from the [Actors.xml](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/movie/input/actors.xml) file, which contains movie data together with detailed information about actors:

```xml
<movies>
	<movie>
	  <id>actors_104</id>
	  <title>Stalag 17</title>
	  <date>1954-01-01</date>
	  <actors>
	    <actor>
		  <name>William Holden</name>
		  <birthday>1918-01-01</birthday>
		  <birthplace>Illinois</birthplace>
		</actor>
	  </actors>
	</movie>
    ...
</movies>
```

The following movie record from the [academy_awards.xml](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/movie/input/academy_awards.xml) file describes the same movie. 
The academy awards dataset contains movie data together with information about the awards a movie has won as well as about movie directors.
```xml
<movies>
	<movie>
	  <id>academy_awards_3059</id>
	  <title>Stalag 17</title>
	  <date>1953-01-01</date>
	  <oscar>yes</oscar>
	  <director>
		<name>Billy Wilder</name>
	  </director>
	  <actors>
	    <actor>
		  <name>Robert Strauss</name>
		</actor>
	  </actors>
	</movie>
    ...
</movies>
```
During identity resolution, WInte.r discovers that the two record describe the same movie by comparing property values such as the title, date, and actors.
Using the discovered correspondences, WInte.r can fuse the two records into a single consolidated record, such as the one below:  

```xml
<movies>
  <movie>
    <id>academy_awards_3059+actors_104</id>
    <title provenance="actors_104">Stalag 17</title>
    <director provenance="academy_awards_3059">Billy Wilder</director>
    <date provenance="academy_awards_3059">1953-01-01T00:00</date>
    <actors provenance="actors_104+academy_awards_3059">
      <actor>
        <name>Robert Strauss</name>
      </actor>
      <actor>
        <name>William Holden</name>
        <birthplace>Illinois</birthplace>
        <birthday>1918-01-01T00:00</birthday>
      </actor>
    </actors>
  </movie>
    ...
</movies>
```
 

# Define Data Model and Load Data

WInte.r uses domain-specific data models which are implemented as Java classes. 
Before we can load the movie data into WInte.r, we thus need to implement a [movie data model](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/Movie.java). 
We define a class `Movie`, which contains all attributes of the movie schema of our input files. 
We also implement XML readers which create instances of the `Movie` class for each movie in the input files.
Please visit the [wiki page on data models](https://github.com/olehmberg/winter/wiki/DataModel) for a detailed description on how WInte.r data models and the corresponding readers are implemented.

Once the movie class is defined and a corresponding [MovieXMLReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLReader.java) is implemented, we can read the movies from the two input files into WInte.r datasets: 

```java
// load data
HashedDataSet<Movie, Attribute> dataAcademyAwards = new HashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/academy_awards.xml"), "/movies/movie", dataAcademyAwards);
HashedDataSet<Movie, Attribute> dataActors = new HashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/actors.xml"), "/movies/movie", dataActors);
```

# Identity Resolution

During identity resolution, the records of two datasets are compared using a matching rule (finding duplicates in a single dataset is very similar and called [Duplicate Detection](https://github.com/olehmberg/winter/wiki/DuplicateDetection)). 
The rule decides whether two records likely describe the same real-world entity or not. 
The result of the identity resolution step is a set of correspondences containing the IDs of the matching record pairs. 

In order to determine the correspondences between the records of two datasets, we need to implement:

1. Comparators that specify which similarity metric is used to compare the values of a specific attribute as well as how these values are transformed/normalized before being compared.
2. A matching rule, which defines how the similarity scores that are produced by the comparators are combined into a matching decision. 

## Creating a Matching Rule

We use a [`LinearCombinationMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRule.java) for comparing movies. 
The rule calculates a weighted sum of the similarity scores and compares the result to a threshold in order to determine if two records match or not. 
An example of a linear combination matching rule is given below:

```
sim(x,y) = 
	0.5 * MovieTitleComparatorJaccard(x,y)+	
	0.5 * MovieDateComparator2Years(x,y)  

If sim(x,y) >= 0.7 Then match
Else non-match
```
The example matching rule averages the similarity values of the [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) and the [`MovieDateComparator2Years`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieDateComparator2Years.java) in order to calculate the similarity of two movie records. 
If the similarity score is above the threshold of 0.7, the record pair is considered a match and a correspondence between the records is created. 

The [`MovieDateComparator2Years`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieDateComparator2Years.java) calculates the relative difference between the 'date' values of the compared movie records using the formula `Max( 1 - (abs(x - y) / 2), 0)`. 
Likewise, the [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) calculates the Jaccard similarity based on word tokens between the 'title' values of the two movie records.

To initialise this example matching rule, use the following code:

```java
// create a matching rule
LinearCombinationMatchingRule<Movie, Attribute> matchingRule 
				= new LinearCombinationMatchingRule<>(0.7);

// add comparators
matchingRule.addComparator(new MovieDateComparator2Years(), 0.5);
matchingRule.addComparator(new MovieTitleComparatorJaccard(), 0.5);
```

As mentioned above, we want to compare movie titles using Jaccard Similarity. 
Thus, we implement a dedicated [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) class, which gets the two movie titles from the movie records and hands them over to the [`TokenizingJaccardSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/TokenizingJaccardSimilarity.java) class.

```java
public class MovieTitleComparatorJaccard implements Comparator<Movie, Attribute> {
	private TokenizingJaccardSimilarity sim = new TokenizingJaccardSimilarity();
	...

	@Override
	public double compare(Movie record1, Movie record2, 
				Correspondence<Attribute, Matchable> schemaCorrespondences) {
		String s1 = record1.getTitle();
		String s2 = record2.getTitle();
		double similarity = sim.calculate(s1, s2);
		...
		return similarity;
	}
.
}
```
The [`TokenizingJaccardSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/TokenizingJaccardSimilarity.java) class defines a similarity measure, which takes as input two strings and returns the Jaccard similarity of these two strings based on word tokens. 

WInte.r provides a wide range of different similarity metrics for strings, dates, numbers, as well as lists. 
Detailed information about the similarity metrics is found on the [wiki page about similarity measures](SimilarityMeasures).


## Running the Identity Resolution

After defining the comparators and adding them to the matching rule, we can instantiate a [`MatchingEngine`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/MatchingEngine.java) and pass the two data sets as well as the matching rule to the method `runIdentityResolution(...)` of the matching engine. 
We also create a blocker that makes sure only movies from the same decade are compared by the matching rule. 
This prevents the calculation of all possible pairs of records and hence drastically reduces the runtime.
Details about blocking are explained in the [Using Blocking to Reduce the Runtime](#using-blocking-to-reduce-the-runtime) Section later in this tutorial.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

// create a blocker (blocking strategy)
StandardRecordBlocker<Movie, Attribute> blocker = new StandardRecordBlocker<Movie, Attribute>(new MovieBlockingKeyByDecadeGenerator());

// Execute the matching
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runIdentityResolution(
	dataAcademyAwards, dataActors, null, matchingRule, blocker);
```
The result of running the identity resolution is a list of correspondences. 
In order to inspect the created correspondences, we write them into a csv file by using the [`CSVCorrespondenceFormatter`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/model/io/CSVCorrespondenceFormatter.java):

```java
// write the correspondences to the output file
new CSVCorrespondenceFormatter().writeCSV(new File("usecase/movie/output/academy_awards_2_actors_correspondences.csv"), correspondences);
```

Within the file, you find the ids of the matching records as well as their similarity:

| Academy_awards Record ID | Actors Record ID | Total Similarity |
|---------------------|-----------|-------|
| academy_awards_723 | actors_144 | 0.75 |
| academy_awards_2337 | actors_117 | 0.75 |
| academy_awards_503 | actors_148 | 1.0 |
| ... | ... | ...  |

For executing the identity resolution yourself and inspecting the results, please run the [`Movies_Tutorial_IdentityResolution_Step01`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_IdentityResolution_Step01.java) class. 
After running the identity resolution you find the correspondences in the file `usecase/movie/output/academy_awards_2_actors_correspondences.csv`. 

## Creating a Gold Standard for Identity Resolution

In order to estimate the quality of a matching result, we compare the discovered correspondence to known correspondences in a gold standard and calculate Precision, Recall, and F1-measure from the comparison result.  

A gold standard for identity resolution must contain matching as well as non-matching record pairs. 
The gold standard should contain enough record pairs to cover the general profile of the data. 
In addition, the gold standard should contain interesting corner cases, such as very similar records that describe two different entities and very different records that describe the same real-world entity. 
As there are usually much more non-matching pairs than matching pairs, the gold standard should also contain more non-matches. 
In general, multiple, skilled annotators should manually verify and agree on the record pairs in the gold standard (high inter-annotator agreement).

WInte.r can read gold standard files that contain the ids of a pair of records as well as the information whether the pair is a `true` match or a `false` one. 
If we know that the movie records `academy_awards_1430` and `actors_132` represent the same real world entity, we store this information in the following format:

```csv
academy_awards_1430,actors_132,true
academy_awards_608,actors_146,true
academy_awards_1132,actors_146,false
academy_awards_1999,actors_146,false
academy_awards_668,actors_146,false
```

## Evaluating the Matching Result

We will now evaluate the matching result using a predefined [gold standard](https://github.com/olehmberg/winter/tree/master/winter-usecases/usecase/movie/goldstandard) for the movies use case.

In order to load a gold standard into WInte.r, create a new [`MatchingGoldStandard`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/model/MatchingGoldStandard.java) instance and load the matches and non-matches from the csv file.

```java
// load the gold standard (test set)
MatchingGoldStandard gsTest = new MatchingGoldStandard();
		gsTest.loadFromCSVFile(new File("usecase/movie/goldstandard/gs_academy_awards_2_actors_test.csv"));
```

The gold standard can now be used to evaluate the matching result:

```java
// evaluate your result
MatchingEvaluator<Movie, Attribute> evaluator = new MatchingEvaluator<Movie, Attribute>();
Performance perfTest = evaluator.evaluateMatching(correspondences, gsTest);

// print the evaluation result
logger.info("Academy Awards <-> Actors");
logger.info(String.format("Precision: %.4f", perfTest.getPrecision()));
logger.info(String.format("Recall: %.4f", perfTest.getRecall()));
logger.info(String.format("F1: %.4f", perfTest.getF1()));
```

The evaluation result looks as follows:

```
Academy Awards <-> Actors
Precision: 	1.0000
Recall: 	0.6596
F1: 		0.7949
```

These results do not satisfy our expectation. Fortunately, WInte.r supports us in improving them.

In order to find out what went wrong, we can enable the [extended event logging](Event-and-Result-Logging) by changing the logging level from 'default' to 'trace'.

```java
private static final Logger logger = WinterLogManager.activateLogger("trace");
```

When running the application with the logging level 'trace', the logs show us which record pairs from the gold standard were correctly identified as matches/non-matches, which pairs were wrong, and which pairs were missed. 
This information gives us a starting point for the analysis of the errors and for afterwards improving the matching rule.

```
[correct] academy_awards_2334,actors_39,0.75
[correct] academy_awards_3300,actors_100,0.75
...
[missing] academy_awards_4529,actors_2
[missing] academy_awards_4500,actors_3
[missing] academy_awards_4475,actors_4
...
```

For more details on the event logging, please visit the [wiki page on event logging](Event-and-Result-Logging).

Now we know which gold standard pairs were missed and which wrong correspondences were created by the matching rule. 
To find the reason for the wrong matching, we want to analyse the application of the matching rule in detail. 
WInte.r supports the user in this analysis by writing detailed logs about the execution of the rule containing all property values that were compared as well as the similarity scores produced by the comparators.

**Before adding the comparators to the matching rule** and running the identity resolution, the user has to activate the result logging via the method `activateDebugReport()`. 
As inputs the method accepts a path to the designated log file and a maximum size of the log.

```java
// create a matching rule
LinearCombinationMatchingRule<Movie, Attribute> matchingRule = new LinearCombinationMatchingRule<>(0.7);

// collect debug results
matchingRule.activateDebugReport("usecase/movie/output/debugResultsMatchingRule.csv", -1);
		
// add comparators
matchingRule.addComparator(new MovieDateComparator2Years(), 0.5);
matchingRule.addComparator(new MovieTitleComparatorJaccard(), 0.5);
```

From the extended result logging (logging level 'trace'), we know that the records academy_awards_4529 and actors_2 have not been matched. Looking up this record pair in the generated debug file, gives us the follow information:

| MatchingRule | Record1Identifier | Record2Identifier | TotalSimilarity |
| ------------- | ------------- |------------- | ------------- | 
|LinearCombinationMatchingRule|academy_awards_4529|actors_2|0.5|

|comparatorName | record1Value | record2Value | record1PreprocessedValue | record2PreprocessedValue | similarity | postproccesedSimilarity |
| ------------- | ------------- |------------- | ------------- | ------------- |------------- | ------------- | 
MovieDateComparator2Years | 1928-01-01T00:00|1930-01-01T00:00|1928-01-01T00:00|1930-01-01T00:00|0.0|0.0|

| comparatorName | record1Value | record2Value | record1PreprocessedValue | record2PreprocessedValue | similarity | postproccesedSimilarity |
| ------------- | ------------- |------------- | ------------- | ------------- |------------- | ------------- | 
MovieTitleComparatorLevenshtein | Coquette | Coquette | Coquette | Coquette |1.0|1.0|


The debug result log reveals that the [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) has correctly found that the titles are equal. 
The [`MovieDateComparator2Years`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieDateComparator10Years.java) has calculated a similarity value of 0.0 for the release dates. 
Given 0.5/0.5 weighting of the two comparators, the low date similarity leads to the wrong decision that the two records match. 

For executing the identity resolution including extended logging and debug reporting yourself, please run the [`Movies_Tutorial_IdentityResolution_Step02`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_IdentityResolution_Step02.java) class.
After running the code, you find the debug file in the folder `usecase/movie/output/`.

## Adjusting Weights of the Matching Rule

Our error analysis has shown us that we have a problem with the weights that are used to aggregate the similarity scores.
Therefore, we decide to reduce the weight of the [`MovieDateComparator2Years`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieDateComparator10Years.java) to 0.3 and to increase the weight of the [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) to 0.7. 

```java

// create a matching rule
LinearCombinationMatchingRule<Movie, Attribute> matchingRule = new LinearCombinationMatchingRule<>(0.7);

// add comparators
matchingRule.addComparator(new MovieDateComparator2Years(), 0.3);
matchingRule.addComparator(new MovieTitleComparatorJaccard(), 0.7);

```

Doing this improves the overall results to:

```
Academy Awards <-> Actors
Precision: 	0.9318
Recall: 	0.8723
F1: 		0.9011
```

For more details on the result logging for matching rules, please visit the [wiki page on result logging](Event-and-Result-Logging).
To execute the identity resolution with the changed comparators, please run the [`Movies_Tutorial_IdentityResolution_Step03`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_IdentityResolution_Step03.java) class.

## Dealing with Missing Values

During data profiling we notice that the attribute actors has a lot of missing values in both data sets.
Despite the missing values the attribute actors can be beneficial if the attribute is filled.
We decide to add the comparator [`MovieActorMissingValueComparator`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieActorMissingValueComparator.java) with a weight of 0.1 and decrease the weight of the [`MovieTitleComparatorJaccard`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieTitleComparatorJaccard.java) to 0.6.
To deal with missing values we use the [`LinearCombinationMatchingRuleWithPenalty`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/LinearCombinationMatchingRuleWithPenalty.java) instead of the [`LinearCombinationMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRule.java).
If the comparator [`MovieActorMissingValueComparator`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieActorMissingValueComparator.java) reports a missing value for the attribute actor, the [`LinearCombinationMatchingRuleWithPenalty`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/LinearCombinationMatchingRuleWithPenalty.java) distributes the weight of the comparator to the other comparators and penalises the aggregated similarity score by the defined penalty of 0.05.

```java

// create a matching rule
LinearCombinationMatchingRuleWithPenalty<Movie, Attribute> matchingRule = new LinearCombinationMatchingRuleWithPenalty<>(0.7);

// add comparators
matchingRule.addComparator(new MovieDateComparator2Years(), 0.3, 0.0);
matchingRule.addComparator(new MovieTitleComparatorJaccard(), 0.6, 0.0);
matchingRule.addComparator(new MovieActorMissingValueComparator(), 0.1, 0.05);

```

Doing this improves the overall results to:

```
Academy Awards <-> Actors
Precision: 	1.0000
Recall: 	0.8723
F1: 		0.9318
```

To execute the identity resolution with the changed comparators, please run the [`Movies_Tutorial_IdentityResolution_Step04`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_IdentityResolution_Step04.java) class. More information about [handling missing values](IdentityResolution#linear-combination-matching-rule-with-penalty).


## Learning a Matching Rule

Instead of manually defining and tuning a matching rule, WInte.r also allows us to learn matching rules from training data (matching and non-matching record pairs). 
WInte.r uses the [WEKA library](https://www.cs.waikato.ac.nz/~ml/weka/) to learn such rules. 
As identity resolution can be cast as a two-class classification problem, we can use all [classification algorithms provided by Weka library](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html) to learn matching rules. 
We just pass the algorithm's name as well as additional options as parameters to a new [`WekaMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/WekaMatchingRule.java) instance. 

As we want the learning algorithm to do the attribute selection and the similarity metric selection for us, we add many different comparators to the matching rule. 
The comparators should implement various attribute/similarity metric combinations. 
This allows the learning algorithm to assign high weights to useful combinations and discard/assign low weights to useless attribute/similarity metric combinations.

```java
// create a matching rule
String options[] = new String[] { "-S" };
String modelType = "SimpleLogistic"; // use a logistic regression
WekaMatchingRule<Movie, Attribute> matchingRule = new WekaMatchingRule<>(0.7, modelType, options);

// add comparators
matchingRule.addComparator(new MovieTitleComparatorEqual());
matchingRule.addComparator(new MovieDateComparator2Years());
matchingRule.addComparator(new MovieDateComparator10Years());
matchingRule.addComparator(new MovieDirectorComparatorJaccard());
matchingRule.addComparator(new MovieDirectorComparatorLevenshtein());
matchingRule.addComparator(new MovieDirectorComparatorLowerCaseJaccard());
matchingRule.addComparator(new MovieTitleComparatorLevenshtein());
matchingRule.addComparator(new MovieTitleComparatorJaccard());
```

For training the matching rule, we need positive and negative examples. 
Ideally, the training set should contain many corner cases, such as very similar records that describe different entities, as these are the most useful examples for learning rules. 
The training set is expected to be stored in the same CSV format that is also used for the evaluation gold standards. 
We use the following code to load the training set and learn the rule afterwards:


```java
// load the training set
MatchingGoldStandard gsTraining = new MatchingGoldStandard();
gsTraining.loadFromCSVFile(new File("usecase/movie/goldstandard/gs_academy_awards_2_actors_train.csv"));

// learn the matching rule
RuleLearner<Movie, Attribute> learner = new RuleLearner<>();
learner.learnMatchingRule(dataAcademyAwards, dataActors, null, matchingRule, gsTraining);
```

The learned matching rule can be applied and evaluated like any other matching rule.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

// Execute the matching
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runIdentityResolution(
	dataAcademyAwards, dataActors, null, matchingRule, new NoBlocker<Movie,Attribute>();


// load the gold standard (test set)
MatchingGoldStandard gsTest = new MatchingGoldStandard();
gsTest.loadFromCSVFile(new File("usecase/movie/goldstandard/gs_academy_awards_2_actors_test.csv"));
		
// evaluate your result
MatchingEvaluator<Movie, Attribute> evaluator = new MatchingEvaluator<Movie, Attribute>();
Performance perfTest = evaluator.evaluateMatching(correspondences, gsTest);
```

In our case, we benefit from learning a matching rule.

```
Academy Awards <-> Actors
Precision: 	1.0000
Recall: 	0.9574
F1: 		0.9783
```

We might be interested in inspecting the learned matching rule in order to understand these results. 
For this purpose, the [`WekaMatchingRule`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/WekaMatchingRule.java) provides the method `getModelDescription()` to output a description of the trained matching rule.

```java
System.out.println(matchingRule.getModelDescription());
```

In the presented case the output looks as follows:

```
SimpleLogistic:
Class 0 :
-5.07 + 
[[6] MovieTitleComparatorLevenshtein] * 2.53 +
[[2] MovieDateComparator10Years] * 2.25 +
[[7] MovieTitleComparatorJaccard] * 1.89 +
[[8] MovieActorComparator] * 1.49
Class 1 :
5.07 + 
[[6] MovieTitleComparatorLevenshtein] * -2.53 +
[[2] MovieDateComparator10Years] * -2.25 +
[[7] MovieTitleComparatorJaccard] * -1.89 +
[[8] MovieActorComparator] * -1.49
```

This output shows that the trained logistic regression relies on the date, title and actors of a movie to determine whether two records are a match. 
Now we can check in detail which matches we could not cover with this approach and by performing this error analysis maybe further improve our matching result (e.g. be including additional comparators into the learning process which do value pre-processing that might be needed to allow the similarity metrics to work correctly).

Please visit the [wiki page on Learning Matching Rules](https://github.com/olehmberg/winter/wiki/Learning-Matching-Rules) for further information on this topic.

Please visit the [wiki page on RapidMiner Integration](RapidMiner-Integration) for further information on how to use external tools such as RapidMiner for learning matching rules.

For executing the rule learning yourself and evaluating the learned rule, please run the [`Movies_Tutorial_IdentityResolution_Step05`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_IdentityResolution_Step05.java) file.

## Using Blocking to Reduce the Runtime

Comparing each record from one dataset with each record from another dataset requires (n * m) / 2 record comparisons, given that n and m are the number of records in each dataset. 
Given larger real world datasets, the number of comparisons quickly gets prohibitive large even given a decent amount of hardware. 
In these situations we can apply blocking in order to heuristically reduce the number of record pairs that are compared using the matching rule to pairs that have a fair chance to match and thus avoid costly comparisons of pair that obviously do not match. 
By applying such blocking heuristics, we can reduce the runtime significantly at the cost of maybe reducing the recall a bit. 

WInte.r implements different blocking strategies. 
In this tutorial, we cover the [`NoBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/NoBlocker.java), the [`StandardRecordBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardRecordBlocker.java), and the [`SortedNeighbourhoodBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/SortedNeighbourhoodBlocker.java). 
Please visit the [wiki page on blocking](https://github.com/olehmberg/winter/wiki/Blocking) for further information.

In cases where no blocker should be used, the [`NoBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/NoBlocker.java) class can be used for identity resolution. 
The no blocker simply generates all pairs of records ((n * m) / 2 ).

```java
..
NoBlocker<Movie, Attribute> blocker = new NoBlocker<>();
..
```

The standard blocker uses blocking keys which are calculated from the records. 
Each record is assigned to one block based on its blocking key value. 
Afterwards, only records in the same block are compared during matching. 
For instance, customers could be blocked by the first two characters of their zip code, which would approximately reduce the runtime by the factor 100.

The implementation of the standard blocker is provided by the [`StandardRecordBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardRecordBlocker.java) class. 
The [`MovieBlockingKeyByDecadeGenerator`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieBlockingKeyByDecadeGenerator.java) calculates the decade from the release date of a movie and uses the resulting value to group movies.

```java
..
StandardRecordBlocker<Movie, Attribute> blocker = new StandardRecordBlocker<Movie, Attribute>(new MovieBlockingKeyByDecadeGenerator());
..
```

The sorted neighbourhood method (SNM) sorts all records by their blocking key and then compares all records within a specified window size.
An implementation of the sorted neighbourhood method is provided by the [SortedNeighbourhoodBlocker](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/SortedNeighbourhoodBlocker.java) class.

```java
..
SortedNeighbourhoodBlocker<Movie, Attribute, Attribute> blocker = new SortedNeighbourhoodBlocker<>(new MovieBlockingKeyByDecadeGenerator(), 30);
..
```

Different blocking approaches as well as different blocking keys result in different record pairs being selected for the detailed comparison using the matching rule. 
In the following table, the result of using different blocking algorithms for matching our movie data are shown. 
The same matching rule is used by all three algorithms, such that only the blocking changes across the three setups.

| Blocking Algorithm | Blocked Pairs | Reduction Ratio | Run time | Identity Resolution Correspondences |
|---------------------|-----------|-----------|-------|-------|
| No Blocker | 691,580 | 0 % | 28.704 sec | 168 | 
| Standardblocker | 82,907 | 88 % | 1.991 sec | 129 | 
| SortedNeighborhoodblocker | 121,856 | 83 % | 2.002 sec | 141 |

We can see from the result table that without blocking a significant number of record pairs is compared. 
Consequently, the runtime of the identity resolution is rather long. 
Nevertheless, the number of correspondences found in the identity resolution only slightly increases for the [`NoBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/NoBlocker.java) compared to the [`StandardRecordBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardRecordBlocker.java) and the [`SortedNeighbourhoodBlocker`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/blockers/SortedNeighbourhoodBlocker.java). 
Hence, it is advised to use some blocking algorithm instead of no blocking.

In a next step we are interested in improving the blocking algorithms by using different blocking keys. 
For supporting us with this task, WInte.r can write a report containing all blocking values together with the number of records in the respective blocks. 
Using this report, we can see if the records are evenly distributed over the blocks. 


```java
..
StandardRecordBlocker<Movie, Attribute> blocker = new StandardRecordBlocker<Movie, Attribute>(new MovieBlockingKeyByDecadeGenerator());
blocker.collectBlockSizeData("usecase/movie/output/debugResultsBlocking.csv", 1000);
..
```

The parameter 1000 restricts the number of blocks that are considered in the report. 
The report looks as follows:

| Blocking Key Value | Frequency |
| ------------- | ------------- |
| 194  | 15600  |
| 196  | 11487  |
| ...  | ...  |

From this result we conclude that the number of records per blocking key value is very high. Following this conclusion, we can e.g. implement a [MovieBlockingKeyByYearGenerator](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/identityresolution/MovieBlockingKeyByYearGenerator.java), which calculates more fine-grained blocks in order to further reduce the number of pairs. 
Using this blocking key generator, we can further speed up the identity resolution process. 

Nevertheless, a more fine-grained blocking key may have a negative impact on the overall matching result, because some actually matching pairs may not end up in the same block and thus not be compared. 
Hence, we recommend to measure the recall (or a least the number of correspondences) of different blocking algorithms with multiple settings to reach satisfying results.

Please visit the [wiki page on result logging](Event-and-Result-Logging) for more information on how to use the reporting about blocking.

Please visit the [wiki page on blocking](https://github.com/olehmberg/winter/wiki/Blocking) for further information.

# Data Fusion

Multiple records that describe the same real-world entity often provide different values for the same attribute. 
The goal of data fusion is to create a single, consolidated record describing the entity while resolving data conflicts.

In the following Sections of this tutorial, we will fuse data about movies originating from the two datasets that we have matched previously in the identity resolution phase and a third dataset, [golden_globes.xml](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/movie/input/golden_globes.xml).

## Loading and Inspecting the Data and Correspondences

The data fusion expects all datasets to be represented using a single schema. In addition, it is expected that correspondences between records have already been discovered during the identity resolution phase. 

We start with loading the datasets and the correspondences and inspecting size of the clusters that result from the correspondences.

```java
// Load the Data into FusibleDataSet
FusibleDataSet<Movie, Attribute> ds1 = new FusibleHashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/academy_awards.xml"), "/movies/movie", ds1);

FusibleDataSet<Movie, Attribute> ds2 = new FusibleHashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/actors.xml"), "/movies/movie", ds2);

FusibleDataSet<Movie, Attribute> ds3 = new FusibleHashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/golden_globes.xml"), "/movies/movie", ds3);
```

After loading a file into a [`FusibleDataSet`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/model/FusibleDataSet.java), we can print the dataset density using the `printDataSetDensityReport()` method:

```java
ds1.printDataSetDensityReport();
```

```
DataSet density: 0,58
Attributes densities:
	Title: 1,00
	Director: 0,09
	Date: 1,00
	Actors: 0,23
```

The correspondences are loaded into a [`CorrespondenceSet`](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/CorrespondenceSet.html).

```java
// load correspondences
CorrespondenceSet<Movie, Attribute> correspondences = new CorrespondenceSet<>();
correspondences.loadCorrespondences(new File("usecase/movie/correspondences/academy_awards_2_actors_correspondences.csv"),ds1, ds2);
correspondences.loadCorrespondences(new File("usecase/movie/correspondences/actors_2_golden_globes_correspondences.csv"),ds2, ds3);
```

Based on the correspondences, WInte.r groups records into clusters, which represent the same entity. 
WInte.r gives us the possibility to inspect these clusters. 

We expect the input datasets to be duplicate free, meaning that each real-world entity is only described by a single record per dataset. 
Thus if we fuse two datasets, we expect a single entity to be described by at most two records. 
If we fuse more than two datasets, we expect entities to be described by no more records than we have datasets.

```java
// write group size distribution
correspondences.printGroupSizeDistribution();
```

```
Group Size 	| Frequency 
————————————————————————————
	2	| 	43
	3	| 	103
	4	| 	2
```

If the group size statistics that we have just outputted derivate significantly from this expectation and we are sure that our input datasets are duplicate free, we can do two things in order to improve the quality of the correspondence sets:

1. Increase the threshold that is used by our matching rule in order to reduce the amount of correspondences that is created (hoping that this results only in a single correspondence per entity).
2. Instead of applying the local strategy to create a correspondence for each record pair having a similarity value above the threshold, apply a global strategy such as a maximum matching, which ensure that only one correspondence per record is created.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

// Execute the matching
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runIdentityResolution(dataAcademyAwards, dataActors, null, matchingRule, blocker);

// Create a maximum weight mapping
MaximumBipartiteMatchingAlgorithm<Movie,Attribute> maxWeight = new MaximumBipartiteMatchingAlgorithm<>(correspondences);
maxWeight.run();
correspondences = maxWeight.getResult();
```

## Creating a Data Fusion Strategy

Different data sources might provide conflicting values for the attributes describing an entity. 
A WInte.r [`DataFusionStrategy`](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionStrategy.html) defines for each attribute how data conflicts are resolved using attribute-specific conflict resolution functions, which are implemented as [`AttributeFuser`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/AttributeFuser.java). 

The code below shows how WInte.r's [`DataFusionEngine`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionEngine.java) is initialised with a [`DataFusionStrategy`](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionStrategy.html) for fusing the movie data and how different [`AttributeFuser`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/AttributeFuser.java)s are added to this strategy. 
In addition to the attribute fusers, we also need to provide an [`EvaluationRule`](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/EvaluationRule.html) for each attribute. 
For the later evaluation of the fusion strategy, these rules define which tolerance should be applied when comparing values to the gold standard. 

```java
// define the fusion strategy
DataFusionStrategy<Movie, Attribute> strategy = new DataFusionStrategy<>(new FusibleMovieFactory());
..		
// add attribute fusers
strategy.addAttributeFuser(Movie.TITLE, new TitleFuserShortestString(),new TitleEvaluationRule());
strategy.addAttributeFuser(Movie.DIRECTOR,new DirectorFuserLongestString(), new DirectorEvaluationRule());
strategy.addAttributeFuser(Movie.DATE, new DateFuserVoting(),new DateEvaluationRule());
strategy.addAttributeFuser(Movie.ACTORS,new ActorsFuserUnion(),new ActorsEvaluationRule());
		
// create the fusion engine
DataFusionEngine<Movie, Attribute> engine = new DataFusionEngine<>(strategy);
```

As an example of a [`AttributeFuser`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/AttributeFuser.java), we show the implementation of the [`TitleFuserShortestString`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/datafusion/fusers/TitleFuserShortestString.java). 
The fuser applies the conflict resolution heuristic to prefer short stings over long strings ([`ShortestString`](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/string/ShortestString.html)), which for instance could make sense to select titles that do not contain additions such as "(DVD version)". 

```java
public class TitleFuserShortestString extends
		AttributeValueFuser<String, Movie, Attribute> {

	public TitleFuserShortestString() {
		super(new ShortestString<Movie, Attribute>());
	}

	@Override
	public void fuse(RecordGroup<Movie, Attribute> group, Movie fusedRecord, Processable<Correspondence<Attribute, Matchable>> schemaCorrespondences, Attribute schemaElement) {

		// get the fused value
		FusedValue<String, Movie, Attribute> fused = getFusedValue(group, schemaCorrespondences, schemaElement);

		// set the value for the fused record
		fusedRecord.setTitle(fused.getValue());

		// add provenance info
		fusedRecord.setAttributeProvenance(Movie.TITLE, fused.getOriginalIds());
	}

..

}
```

WInte.r implements a wide range of different conflict resolution functions. 
Please visit the [wiki page on data fusion](https://github.com/olehmberg/winter/wiki/DataFusion) for a complete list of the functions.

Further, the data fusion strategy allows us to create a consistency report for the records clusters using the `printClusterConsistencyReport()` method:

```java
engine.printClusterConsistencyReport(correspondences, null);
```

```
Attribute Consistencies:
	Director: 0,98
	Title: 0,97
	Actors: 0,58
	Date: 0,47
```

With this setup, we can run the data fusion, which returns a fused data set.

```java
// run the fusion
FusibleDataSet<Movie, Attribute> fusedDataSet = engine.run(correspondences, null);
```
We can write the dataset into an XML file using the [`MovieXMLFormatter`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLFormatter.java). 
Please visit the [wiki page on the data model](https://github.com/olehmberg/winter/wiki/DataModel) for a detailed description of the [`MovieXMLFormatter`](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLFormatter.java):

```java
// write the result
new MovieXMLFormatter().writeXML(new File("usecase/movie/output/fused.xml"), fusedDataSet);
```

For executing the data fusion yourself and inspecting the resulted fused dataset, please run the [`Movies_Tutorial_DataFusion_Step01`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_DataFusion_Step01.java) class.


## Creating a Gold Standard for Data Fusion

In order to calculate the accuracy of the fused values, we need to compare the fused values to the true values. 
Such ground truth can be loaded into WInte.r in the form of a data fusion gold standard. 
This gold standard contains correctly fused movie records in the target format. 

Ground truth values should be collected manually from trustable sources (e.g. official statistical bodies, the homepage of a band itself). 
In order to help you to select interesting records to include into your gold standard, WInte.r can rank records according to the amount of conflicts that they contain. 
Please run the [`Movies_Tutorial_DataFusion_Step02`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_DataFusion_Step02.java) class in order to generate such a ranking. 

```java
// print record groups sorted by consistency
engine.writeRecordGroupsByConsistency(new File("usecase/movie/output/recordGroupConsistencies.csv"), correspondences, null);
```

Running the ranking for our movies data tells us that the record cluster containing the movies `academy_awards_3624` and `actors_17` has a very low average consistency of only 0.33. 

```xml
<movies>
..
	<movie>
		<id>academy_awards_3624</id>
		<title>Gaslight</title>
		<director/>
		<actors>
			<actor>
				<name>Charles Boyer</name>
			</actor>
			<actor>
				<name>Ingrid Bergman</name>
			</actor>
			<actor>
				<name>Angela Lansbury</name>
			</actor>
		</actors>
		<date>1944-01-01</date>
		<oscar>yes</oscar>
	</movie>
	<movie>
		<id>actors_17</id>
		<title>Gaslight</title>
		<actors>
			<actor>
				<name>Ingrid Bergman</name>
			</actor>
		</actors>
		<date>1945-01-01</date>
	</movie>
    ...
</movies>
```

Thus, we search for the true values and include the record containing the true values into our gold standard:

```xml
<movie>
	<id>academy_awards_3624</id>
	<title>Gaslight</title>
	<director/>
	<actors>
		<actor>
			<name>Charles Boyer</name>
		</actor>
		<actor>
			<name>Ingrid Bergman</name>
		</actor>
		<actor>
			<name>Angela Lansbury</name>
		</actor>
	</actors>
	<date>1944-01-01</date>
	<oscar>yes</oscar>
</movie>
```

Another option to analyse record groups based on their average consistency in detail is to consult result logging for data fusion.
Please refer to the section on [Result Logging for Data Fusion](https://github.com/olehmberg/winter/wiki/Event-and-Result-Logging#data-fusion) for further details.

## Evaluating the Data Fusion Strategy

Before we can calculate the accuracy of the fused data utilizing our newly created gold standard, we still need to define which values should be considered the same in the evaluation. 
For instance, population numbers of large cities often differ by a small number of people. 
Nevertheless, we might want to consider a fused population number of 1,700,023 close enough to the gold standard value of 1,700,000 people in order to count it as correct. 
WInte.r allows us to define attribute-specific tolerance ranges in the form of [`EvaluationRules`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/EvaluationRule.java).

An example of such a rule is the [TitleEvaluationRule](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/datafusion/evaluation/TitleEvaluationRule.java), which compares the titles of two movies and returns true, in case their
similarity based on the Jaccard similarity is 1.0 meaning that it ignores the order of words, but is strict about typos in the words.

```java
public class TitleEvaluationRule extends EvaluationRule<Movie, Attribute> {

	SimilarityMeasure<String> sim = new TokenizingJaccardSimilarity();

	@Override
	public boolean isEqual(Movie record1, Movie record2, Attribute schemaElement) {
		// the title is correct if all tokens are there, but the order does not
		// matter
		return sim.calculate(record1.getTitle(), record2.getTitle()) == 1.0;
	}
..	
}
```

WInte.r contains an [example gold standard](https://github.com/olehmberg/winter/tree/master/winter-usecases/usecase/movie/goldstandard) for the movies use case.

To load the gold standard into WInte.r, use the already known [MovieXMLReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLReader.java) class and load the corresponding gold standard from the XML file. 
Afterwards we use the [DataFusionEvaluator](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionEvaluator.java) to measure the overall accuracy of the fused data set compared to the gold standard.

```java
// load the gold standard
DataSet<Movie, Attribute> gs = new FusibleHashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/goldstandard/fused.xml"), "/movies/movie", gs);

// evaluate
DataFusionEvaluator<Movie, Attribute> evaluator = new DataFusionEvaluator<>(
				strategy, new RecordGroupFactory<Movie, Attribute>());
		
double accuracy = evaluator.evaluate(fusedDataSet, gs, null);

logger.info(String.format("Accuracy: %.2f", accuracy));
```

This setup reveals an overall accuracy of 0.80 on the fused data set compared with gold standard. Again, WInte.r can guide us to improve this result.
Please execute the [`Movies_Tutorial_DataFusion_Step03`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_DataFusion_Step03.java) class to run the evaluation.

## Adjusting the Data Fusion Strategy

Before adjusting our data fusion strategy, we have to analyse how we can improve our data fusion strategy.
Hence, in a first step we can change the logging level from 'default' to 'trace' to retrieve a more detailed log. Please visit the [wiki page on event logging](Event-and-Result-Logging) for further details on this topic.

```java
private static final Logger logger = WinterLogManager.activateLogger("trace");
```

Running the data fusion with the log level 'trace' adds two types of debug information to the output: First, an overview of the attribute-specific accuracies is presented to the user.

```
Attribute-specific Accuracy:
	Title: 0.90
 	Director: 0.75
	Date: 0.40
 	Actors: 0.90
```

This reveals that we should try to optimize the `Date` fuser in order to improve the overall accuracy.

The second debug output helps us to get ideas on how we might be able to improve the fusers. 
The output shows us the individual differences between the fused values and the gold standard values. 
Thereby, we get a first impression why the conflict resolution function did not provide the expected result.

```
Error in 'Date': 
	[Movie academy_awards_3480+actors_97: The Best Years of Our Lives / William Wyler / 1947-01-01T00:00] 
	<> [Movie academy_awards_3480: The Best Years of Our Lives / William Wyler / 1946-01-01T00:00]
```
In the above shown example the `Date` of the fused data set does not match the `Date` value in the gold standard. 
To further analyse the reason for this mismatch, we can log all intermediate debug results of the data fusion strategy:

```java
// define the fusion strategy
DataFusionStrategy<Movie, Attribute> strategy = new DataFusionStrategy<>(new FusibleMovieFactory());
// write debug results to file
strategy.collectDebugData("usecase/movie/output/debugResultsDatafusion.csv", 1000);
```

The output from the debug data looks as follows:

| Attribute Name | Consistency | ValueIDS | Values | FusedValue | Is Correct? | Correct Value |
| --- | --- | ------------- | ------------- |------------- | --- | --- |
| Date | 0.67 | Date-{actors_132 \| golden_globes_1330 \| academy_awards_1430} | {1985-01-01T00:00 \| 1985-01-01T00:00 \| 1984-01-01T00:00} | 1985-01-01T00:00 | false | 1984-01-01T00:00 |
| ... | ... | ...  | ...  | ...|

Please execute the [`Movies_Tutorial_DataFusion_Step04`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_DataFusion_Step04.java) class to see the detailed logs.

Analysing the debug data reveals that the `Date` data is correctly provided by the 'academy_awards' data set. 
Thus, we change our conflict resolution strategy for the attribute `Date` and decide to use the [DateFuserFavourSource](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/datafusion/fusers/DateFuserFavourSource.java) as AttributeFuser for the attribute `Date`. 
We rate our data sets, such that we assign the 'academy_awards' data set a higher trustworthiness score than the 'actors' dataset. 
This way the date value of the 'academy_awards' dataset is always preferred over the value provided by the 'actors' dataset. 
If the 'academy_awards' dataset does not contain a release date value, the fusion heuristic falls back to using the value from the 'actors' dataset.
We also replace the TitleFuserShortestString with a TitleFuserLongestString:

```java
// set scores
ds1.setScore(3.0);
ds2.setScore(1.0);
ds3.setScore(2.0);

// define the fusion strategy
DataFusionStrategy<Movie, Attribute> strategy = new DataFusionStrategy<>(new FusibleMovieFactory());
		
// add attribute fusers
strategy.addAttributeFuser(Movie.TITLE, new TitleFuserLongestString(),new TitleEvaluationRule());
strategy.addAttributeFuser(Movie.DIRECTOR,new DirectorFuserLongestString(), new DirectorEvaluationRule());
strategy.addAttributeFuser(Movie.DATE, new DateFuserFavourSource(),new DateEvaluationRule());
strategy.addAttributeFuser(Movie.ACTORS,new ActorsFuserUnion(),new ActorsEvaluationRule());
```

Evaluating the updated data fusion strategy reveals that we were able to increases the overall accuracy from 0.80 to 0.95. 
Additionally, the attribute-specific accuracy for the attribute `Date` increased from 0.40 to 0.95, and for the attribute `Title` from 0.90 to 0.95. 

Now we could return to the debug log in order to find further insights on how we can improve our data fusion strategy for the remaining attributes.

For executing the data fusion yourself and evaluating the accuracy of the fused dataset, please run the [`Movies_Tutorial_DataFusion_Step05`](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_Tutorial_DataFusion_Step05.java).

Please visit the [wiki page on data fusion](https://github.com/olehmberg/winter/wiki/DataFusion) for further details on how WInte.r implements data fusion.


## Feedback about the Tutorial

We hope that you consider this tutorial helpful for getting started with using WInte.r. 
If you have ideas on how this tutorial could be improved, please send them via email to Alexander Brinkmann (alex.brinkmann@informatik.uni-mannheim.de) and Christian Bizer (chris@informatik.uni-mannheim.de).
