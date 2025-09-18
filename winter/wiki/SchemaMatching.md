Schema Matching is the process of finding correspondences between the attributes of two data sources. The result of schema matching is a mapping that allows the transformation of one schema into the other. This page shows examples for the different pre-implemented methods introduced in the [Matching](../Matching) Section.

## Label-based schema matching
Label-based schema matching aligns the attributes of two datasets based on their names (labels).

```csv
scifi1.csv:
"id","title","studio","movie genre","budget","gross","director","date"
...

scifi2.csv:
"id","film title","production studio","genre","cost","earned","dir.","released"
...
```

Load the two datasets from CSV files using the default model:

```java
// load data
DataSet<Record, Attribute> data1 = new HashedDataSet<>();
new CSVRecordReader(0).loadFromCSV(new File("scifi1.csv"), data1);
DataSet<Record, Attribute> data2 = new HashedDataSet<>();
new CSVRecordReader(0).loadFromCSV(new File("scifi2.csv"), data2);
```

Initialise the matching engine and run the label-based schema matching:

```java
// Initialize Matching Engine
MatchingEngine<Record, Attribute> engine = new MatchingEngine<>();
// run the matching
Processable<Correspondence<Attribute, Record>> correspondences
  = engine.runLabelBasedSchemaMatching(data1.getSchema(), data2.getSchema(), new LabelComparatorJaccard(), 0.5);
```

The result is the alignment/mapping represented by correspondences between the attributes:

```java
// print results
for(Correspondence<Attribute, Record> cor : correspondences.get()) {
  System.out.println(String.format("'%s' <-> '%s' (%.4f)",
    cor.getFirstRecord().getName(),
    cor.getSecondRecord().getName(),
    cor.getSimilarityScore()));
}
```

Output:

```
'title' <-> 'film title' (0,5000)
'studio' <-> 'production studio' (0,5000)
'id' <-> 'id' (1,0000)
'movie genre' <-> 'genre' (0,5000)
```

## Instance-based schema matching
Instance-based schema matching aligns the attributes of two datasets based on their values.

Load two datasets from CSV files using the default model:

```java
// load data
DataSet<Record, Attribute> data1 = new HashedDataSet<>();
new CSVRecordReader(-1).loadFromCSV(new File("usecase/movie/input/scifi1.csv"), data1);
DataSet<Record, Attribute> data2 = new HashedDataSet<>();
new CSVRecordReader(-1).loadFromCSV(new File("usecase/movie/input/scifi2.csv"), data2);
```

Define a blocker that uses the attribute values to create pairs potentially matching attributes.

```java
// define a blocker that uses the attribute values to generate pairs
InstanceBasedSchemaBlocker<Record, Attribute, MatchableValue> blocker
  = new InstanceBasedSchemaBlocker<>(
    new AttributeValueGenerator(data1.getSchema()),
    new AttributeValueGenerator(data2.getSchema()));
```

Define an aggregator that calculates a similarity score based on all matching values between an attribute combination

```java
// to calculate the similarity score, aggregate the pairs by counting
// and normalise with the number of record in the smaller dataset
// (= the maximum number of records that can match)
VotingAggregator<Attribute, MatchableValue> aggregator
  = new VotingAggregator<>(
    false,
    Math.min(data1.size(), data2.size()),
    0.0);
```

Run the instance-based schema matching via the matching engine

```java
// Initialize Matching Engine
MatchingEngine<Record, Attribute> engine = new MatchingEngine<>();
// run the matching
Processable<Correspondence<Attribute, MatchableValue>> correspondences
= engine.runInstanceBasedSchemaMatching(data1, data2, blocker, aggregator);
```

Finally, print the results to the console		

```java
// print results
for(Correspondence<Attribute, MatchableValue> cor : correspondences.get()) {
  System.out.println(String.format("'%s' <-> '%s' (%.4f)",
    cor.getFirstRecord().getName(),
    cor.getSecondRecord().getName(),
    cor.getSimilarityScore()));
}
```

Output:
```
'budget' <-> 'cost' (0,0500)
'id' <-> 'id' (0,7500)
'director' <-> 'dir.' (0,7500)
'gross' <-> 'earned' (0,0500)
'title' <-> 'film title' (0,7500)
'gross' <-> 'cost' (0,0500)
'date' <-> 'released' (0,7000)
'budget' <-> 'earned' (0,0500)
```

## Duplicate-based schema matching

Duplicate-based schema matching compares the attribute values of known duplicates in the two data sources to infer the attribute mapping. This requires a set of instance correspondences, which can be created by [Identity Resolution](IdentityResolution.md)].

Load two datasets with different schemas and overlapping records:

```java
// load data
DataSet<Record, Attribute> data1 = new HashedDataSet<>();
new CSVRecordReader().loadFromCSV(new File("usecase/movie/input/scifi1.csv"), data1);
DataSet<Record, Attribute> data2 = new HashedDataSet<>();
new CSVRecordReader().loadFromCSV(new File("usecase/movie/input/scifi2.csv"), data2);
```

Load a set of duplicates:

```java
// load duplicates
Processable<Correspondence<Record, Attribute>> duplicates
  = Correspondence.loadFromCsv(
    new File("scifi1_2_scifi2_instance_correspondences.csv"),
    data1,
    data2);
```

Define the rule for duplicate-based schema matching. Here, the similarity function only accepts exact matches.		

```java
// define the schema matching rule
SchemaMatchingRuleWithVoting<Record, Attribute, Attribute> schemaRule
  = new DuplicateBasedSchemaMatchingRule<>(

	(a1,a2,c) -> {
    // get both attribute values
		String value1 = c.getFirstRecord().getValue(a1);
		String value2 = c.getSecondRecord().getValue(a2);

    // check if they are equal
		if(value1!=null && value2!=null && value1.equals(value2)) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

, 1.0);
```

Run the matching with the just defined rule:

```java
// Initialize Matching Engine
MatchingEngine<Record, Attribute> engine = new MatchingEngine<>();
// Execute the matching
Result<Correspondence<Attribute, Record>> correspondences = engine.runDuplicateBasedSchemaMatching(data1.getSchema(), data2.getSchema(), duplicates, schemaRule);
```

And print the results to the console:

```java
// print results
for(Correspondence<Attribute, Record> cor : correspondences.get()) {
	System.out.println(
String.format("'%s' <-> '%s' (%.4f)",
cor.getFirstRecord().getName(),
cor.getSecondRecord().getName(),
cor.getSimilarityScore()));
}
```

Output:

```
'director' <-> 'dir.' (1,0000)
'title' <-> 'film title' (1,0000)
'id' <-> 'id' (1,0000)
'date' <-> 'released' (1,0000)
```
