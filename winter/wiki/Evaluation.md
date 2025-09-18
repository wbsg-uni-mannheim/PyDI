Evaluation is the process of objectively measuring the performance of the data integration process. It consists of defining a Gold Standard, i.e., the perfect result for a subset of the data, running the methods and then comparing the result of the methods to the Gold Standard. WInte.r provides you with implementations to perform such an evaluation, which are described on this page.

## Defining a Gold Standard for Matching
Gold standards for matching are simple CSV files that list two record IDs and a boolean flag specifying whether the match is correct or not:
```csv
Dataset1_record1,Dataset2_record2,true
Dataset1_record2,Dataset5_record5,true
Dataset1_record1,Dataset2_record1,false
Dataset1_record2,Dataset2_record7,false
```

The gold standards can either be complete or partial:

**Complete Gold Standard**: Contains all possible matches in the data sets. All correspondences must be marked as correct (the third value is "true").

```java
MatchingGoldStandard gs = new MatchingGoldStandard();
gs.loadFromCSVFile(new File("complete.csv"));
gs.setComplete(true);
```

**Partial Gold Standard**: Contains positive and negative examples for matches, indicated by the flag “true” or “false” as third value. Only correspondences that are included in the partial gold standard are evaluated.

```java
MatchingGoldStandard gs = new MatchingGoldStandard();
gs.loadFromCSVFile(new File("partial.csv"));
```

The evaluation of a matching result is performed by the matching evaluator:
```java
MatchingEvaluator<Record, Attribute> evaluator =
  new MatchingEvaluator<Record, Attribute>(true);
Performance perf = evaluator.evaluateMatching(correspondences.get(),gs);
```

## Defining a Gold Standard for Data Fusion
For data fusion, a gold standard is just another dataset. If the fused values are the same as the values in this dataset (identified by the record ids), they are evaluated as correct. How much variation is allowed is specified by the EvaluationRules in the DataFusionStrategy.
```java
// load the gold standard
DataSet<Movie, Attribute> gs = new FusableDataSet<>();
new MovieXMLReader().loadFromXML(new File("fused.xml"), "/movies/movie", gs);

// evaluate
DataFusionEvaluator<Movie, Attribute> evaluator = new DataFusionEvaluator<>(
  strategy,
  new RecordGroupFactory<Movie, Attribute>());

double accuracy = evaluator.evaluate(fusedDataSet, gs, null);
```
