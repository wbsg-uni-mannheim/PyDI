# Data Fusion
The data fusion process can be executed with the [DataFusionEngine](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionEngine.html). The following Figure illustrates the process:

![Data Fusion Overview](img/fusion_overview.png)

The data fusion method expects datasets in a consolidated schema and correspondences between their records.
The correspondences can be loaded into a [CorrespondenceSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/CorrespondenceSet.html).
From these correspondences, groups of records which represent the same entity are collected for each attribute.
Such an entity/attribute group then contains all values for this combination from the input datasets.
To decide for a final value, a ConflictResolutionFunction is applied to the values.
The DataFusionEngine is initialised with a [DataFusionStrategy](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/DataFusionStrategy.html). This strategy defines a [ConflictResolutionFunction](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/ConflictResolutionFunction.html) and an [EvaluationRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/EvaluationRule.html) for each attribute. These functions determine how a final value is chosen from multiple possible values and how it is evaluated.

## Implemented Conflict Resolution Functions

- [Voting](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/Voting.html): Applies majority voting to the values
- [ClusteredVote](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/ClusteredVote.html): Clusters all values using the provided similarity measure and returns the centroid of the largest resulting cluster
- [Intersection](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/list/Intersection.html): Creates the intersection of all values (applicable if values are sets)
- [IntersectionKSources](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/list/IntersectionKSources.html): Creates a set of all values that are included in at least k input values (applicable if values are sets)
- [Union](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/list/Union.html): Creates the union of all values (applicable if values are sets)
- [FavourSources](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/meta/FavourSources.html): Returns the value from the source with the highest score (as defined by the user)
- [MostRecent](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/meta/MostRecent.html): Returns the value from the source that is most recent (as defined by the user)
- [Average](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/numeric/Average.html): Returns the average of all values (numeric)
- [Median](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/numeric/Median.html): Returns the median of all values (numeric)
- [LongestString](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/string/LongestString.html): Returns the longest value by character count (strings)
- [ShortestString](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/datafusion/conflictresolution/string/ShortestString.html): Returns the shortest value by character count (strings)


# Example

Load the datasets:

```java
// Load the Data into FusableDataSet
FusableDataSet<Movie, Attribute> ds1 = new FusableDataSet<>();
new MovieXMLReader().loadFromXML(
  new File("usecase/movie/input/academy_awards.xml"),
  "/movies/movie",
  ds1);
ds1.printDataSetDensityReport();

FusableDataSet<Movie, Attribute> ds2 = new FusableDataSet<>();
new MovieXMLReader().loadFromXML(
  new File("usecase/movie/input/actors.xml"),
  "/movies/movie",
  ds2);
ds2.printDataSetDensityReport();

FusableDataSet<Movie, Attribute> ds3 = new FusableDataSet<>();
new MovieXMLReader().loadFromXML(
  new File("usecase/movie/input/golden_globes.xml"),
  "/movies/movie",
  ds3);
ds3.printDataSetDensityReport();
```

Load the record correspondences between the datasets (obtained from identity resolution):

```java
// load correspondences
CorrespondenceSet<Movie, Attribute> correspondences = new CorrespondenceSet<>();
correspondences.loadCorrespondences(
  new File("usecase/movie/correspondences/academy_awards_2_actors_correspondences.csv"),
  ds1,
  ds2);
correspondences.loadCorrespondences(
  new File("usecase/movie/correspondences/actors_2_golden_globes_correspondences.csv"),
  ds2,
  ds3);
```

Define the data fusion strategy and add fusers for multiple attributes:

```java
// define the fusion strategy
DataFusionStrategy<Movie, Attribute> strategy = new DataFusionStrategy<>(new MovieXMLReader());
// add attribute fusers
strategy.addAttributeFuser(
  Movie.TITLE,
  new TitleFuserShortestString(),
  new TitleEvaluationRule());
strategy.addAttributeFuser(
  Movie.DIRECTOR,
  new DirectorFuserLongestString(),
  new DirectorEvaluationRule());
strategy.addAttributeFuser(
  Movie.DATE,
  new DateFuserVoting(),
  new DateEvaluationRule());
strategy.addAttributeFuser(
  Movie.ACTORS,
  new ActorsFuserUnion(),
  new ActorsEvaluationRule());
```

Initialise the data fusion engine with the strategy and run it:

```java
// create the fusion engine
DataFusionEngine<Movie, Attribute> engine = new DataFusionEngine<>(strategy);
// run the fusion
FusableDataSet<Movie, Attribute> fusedDataSet = engine.run(correspondences, null);
```

Load the gold standard and evaluate the fusion result:

```java
// load the gold standard
DataSet<Movie, Attribute> gs = new FusableDataSet<>();
new MovieXMLReader().loadFromXML(
  new File("usecase/movie/goldstandard/fused.xml"),
  "/movies/movie",
  gs);
// evaluate
DataFusionEvaluator<Movie, Attribute> evaluator = new DataFusionEvaluator<>(
  strategy, new RecordGroupFactory<Movie, Attribute>());
evaluator.setVerbose(true);

double accuracy = evaluator.evaluate(fusedDataSet, gs, null);

System.out.println(String.format("Accuracy: %.2f", accuracy));
```
