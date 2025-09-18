Duplicate detection methods (also known as data matching or record linkage methods) identify records in the same dataset that describe the same real-world entity. 

## Rule-based duplicate detection

1.	First we load the data set

```java
// loading data
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

3.	To speed up the whole process, we only want to compare records that seem similar instead of comparing all pairs of records. Hence, we add a blocking strategy that only compares movies from the same decade.

```java
// create a blocker (blocking strategy)
Blocker<Movie, Attribute> blocker = new StandardBlocker<Movie, Attribute>(
(m) -> Integer.toString(m.getDate().getYear() / 10));
```

4.	Finally, we initialise the matching engine, which does all the work for us, and run the duplicate detection with our matching rule.

```java
// Initialize Matching Engine
MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

// Execute the matching
Processable<Correspondence<Movie, Attribute>> correspondences = engine.runDuplicateDetection(dataActors, matchingRule, blocker);
```

The ```correspondences``` variable then contains all duplicates that were detected by our matching rule.