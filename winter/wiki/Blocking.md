This page gives an overview over the Blockers that are provided by WInte.r. Blocking (also known as Indexing) is applied to reduce the number of records or attributes that have to be compared during a matching operation.

All blockers are defined in the `de.uni_mannheim.informatik.dws.winter.matching.blockers` package and implement either the `SingleDataSetBlocker` or the `CrossDataSetBlocker` interface.
A `SingleDataSetBlocker` uses one input dataset and is, for example, used in duplicate detection.
A `CrossDataSetBlocker` uses two input datasets and is used for identity resolution and schema matching.

## No Blocker
In cases where no blocker should be used, the `NoBlocker` class can be used for identity resolution and the `NoSchemaBlocker` for schema matching. These blockers simply generate all possible pairs of records.

## Standard Blocker
A standard blocker uses blocking keys, which are generated from the records. Each record is assigned to one or more blocks based on the blocking key. Then, only records in the same block are compared during matching.
The implementation of the standard blocker is provided by the `StandardBlocker` class. It implements both the the `SingleDataSetBlocker` and the `CrossDataSetBlocker` interface.
For convenience, the `StandardRecordBlocker` and `StandardSchemaBlocker` can be used for identity resolution and schema matching respectively.
Both extend the `StandardBlocker`, but have less type parameters which makes them easier to use.

An example can be seen in the use-case example `de.uni_mannheim.informatik.dws.winter.usecase.movies.Movies_IdentityResolution_Main`.

The standard blocker can receive additional correspondences to generate the blocking key or to be forwarded to the matching rule. A typical scenario is that the result of schema matching, i.e., schema correspondences, are used for identity resolution. In this case, the schema correspondences are added to every generated pair as causal correspondences. These correspondences are then available in the matching rule that is executed after the blocker.

## Value-based Blocker
The value-based blocker is a variaton of the standard blocker. It assumes that the blocking keys are the values from the records, using the `MatchableValue` class.
The implementation of the standard blocker is provided by the `ValueBasedBlocker` class. It implements both the the `SingleDataSetBlocker` and the `CrossDataSetBlocker` interface.
When generating pairs, causal correspondences are added for all matching values with the number of matches for each value as similarity score.
For convenience, the `InstanceBasedRecordBlocker` and `InstanceBasedSchemaBlocker` can be used for identity resolution and schema matching respectively.
Both extend the `ValueBasedBlocker`, but have less type parameters which makes them easier to use.

An example can be seen in the use-case example `de.uni_mannheim.informatik.dws.winter.usecase.movies.Movies_SimpleIdentityResolution`.
In this example, the following record exists in both datasets:

```csv
"A","Spirited Away",,,"0.0","0.0","Hayao Miyazaki","2001-01-01T00:00:00.000+01:00"
"B","Spirited Away",,,"0.0","0.0","Hayao Miyazaki","2001-01-01T00:00:00.000+01:00"
```

The value-based blocker creates a pair for the two versions of the record with the following causal correspondences:

```
Hayao Miyazaki <-> Hayao Miyazaki (1,0)
Spirited Away <-> Spirited Away (1,0)
2001-01-01T00:00:00.000+01:00 <-> 2001-01-01T00:00:00.000+01:00 (1,0)
0.0 <-> 0.0 (2,0)
```

## Sorted Neighbourhood method
The sorted Neighbourhood method sorts all records by their blocking key and then compares all records within a specified window size.
The implementation of the sorted neighbourhood method is provided by the `SortedNeighbourhoodBlocker` class. It implements both the the `SingleDataSetBlocker` and the `CrossDataSetBlocker` interface.

# Blocker Evaluation
To choose the most suitable blocker it can be evaluate how well/poor the selected blocker filters the candidate pairs.
For this evaluation blocking needs to be executed on the two data sets first.
The `MatchingEngine` provides the respective `runBlocking` method.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();
// Execute blocking
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runBlocking(
        dataAcademyAwards, dataActors, null, blocker);
```

The blocked pairs are now evaluated on a goldstandard using the `MatchingEvaluator`.
To learn more about how to construct such a goldstandard for matching please refer to the article on general [evaluation](https://github.com/olehmberg/winter/wiki/Evaluation).

```java
// Evaluate result
MatchingEvaluator<Movie, Attribute> evaluator = new MatchingEvaluator<Movie, Attribute>();
Performance perfTest = evaluator.evaluateMatching(correspondences.get(), gsTest);
```

The results of the evaluation are printed to the console.

```
Academy Awards <-> Actors
Precision:  0.0114
Recall:     0.8085
F1:         0.0225
```

In the context of blocking precision is referred to as pairs quality and recall is referred to as pairs completeness.

