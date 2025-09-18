For [Schema Matching](../SchemaMatching) and [Identity Resolution](../IdentityResolution), the [MatchingEngine](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/MatchingEngine.html) facade provides a starting point for many common operations. The following Figure gives an overview of the rule-based matching process:

![Matching Overview](img/matching_overview.png)

Data is stored in DataSets, which can contain a schema and records.
These datasets are passed to the MatchingEngine, which first runs a Blocker such as the [StandardRecordBlocker](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardRecordBlocker.html) or [StandardSchemaBlocker](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardSchemaBlocker.html).
These blockers use a [BlockingKeyGenerator](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/generators/BlockingKeyGenerator.html) to generate blocking key values from attributes or records.
The result of the blocking is a set of attribute or record pairs, which are then evaluated by a [MatchingRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/MatchingRule.html), which internally uses [Comparator](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/Comparator.html)s that calculate similarity values.
For example, the [LinearCombinationMatchingRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRule.html) allows you to specify different weights for the comparators and a final threshold for the combined similarity value.
The result of the matching operation is a set of [Correspondence](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/Correspondence.html)s.
Given these correspondences and a [MatchingGoldStandard](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/MatchingGoldStandard.html), the [MatchingEvaluator](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/MatchingEvaluator.html) calculates a [Performance](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/Performance.html) that contains Precision, Recall, and F1-Measure.


For schema matching and identity resolution, the MatchingEngine facade provides implementations for the following matching operations:

## Token-based Duplicate Detection / Identity Resolution
Finds duplicate records in a single data set (duplicate detection) or in two different data sets (identity resolution). Uses a Blocker to generate tokens from the records and a CorrespondenceAggregator to calculate a similarity value from the matching tokens.

## Rule-Based Duplicate Detection / Identity Resolution
Finds duplicate records in a single data set (duplicate detection) or in two different data sets (identity resolution). Accepts a MatchingRule and a Blocker as parameters. First, the Blocker is used to generate candidate pairs of records, which are then evaluated using the MatchingRule.

## Label-based Schema Matching
Finds similar attributes in two different data sets by applying a comparator to attribute names. A comparator specifies value pre-processing and a similarity measure to compare the values.

## Instance-based Schema Matching
Finds similar attributes in two different data sets by comparing the values of attributes. Uses a Blocker to generate tokens from the attribute values and a CorrespondenceAggregator to calculate a similarity value from the matching tokens.

## Duplicate-based Schema Matching
Finds similar attributes in two different data sets by comparing the values of duplicate records in both datasets. Accepts the correspondences between duplicate records, a SchemaMatchingRule and a SchemaBlocker as parameters. First, the SchemaBlocker is used to generate possible pairs of attributes. Then, the SchemaMatchingRule is applied to the values of all generated pairs for all duplicate records. Finally, the voting definition of the SchemaMatchingRule is used to aggregate the results to corespondences between attributes.

# Building Blocks for Matching
Internally, the MatchingEngine uses the algorithms from the de.uni_mannheim.informatik.wdi.matching.algorithms package. These algorithms are composed of different building blocks, which can be re-used for the implementation of further algorithms.
There are three types of building blocks: blockers, matching rules and aggregators:

**[Blockers](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/package-summary.html)**: A blocker transforms one or multiple datasets into pairs of records. The main objective is to reduce the number of comparisons that have to be performed in the subsequent steps while still keeping the pairs that are actual matches. The blockers can receive correspondences as additional input. These can be used to perform the blocking and/or can be added to the generated pairs to be used by the following matcher.

**[MatchingRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/MatchingRule.html)**: A matching rule is applied to a pair of records and determines if they are a match or not. A pair can contain additional correspondences that can be used by the matching rule to make its decision. There are two different types of matching rules:

**[Filtering Matching Rule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/FilteringMatchingRule.html)**: receives a pair of records and possibly many correspondences from the blocker and decides if this pair is a match, in which case it is produced as correspondence. Non-matching pairs are filtered out.

**[Aggregable Matching Rule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/AggregableMatchingRule.html)**: receives a pair of records and one of the correspondences from the blocker and decides if this pair is a match, in which case it is produced as correspondence. If multiple correspondences exist for a pair, it can create multiple output correspondences (which act as votes). The rule also specifies how such correspondences should be grouped for aggregation by an Aggregator.

**[Aggregators](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/aggregators/package-summary.html)**: An aggregator combines multiple correspondences by aggregating their similarity scores. Such aggregations can be sum, average, voting or top-k.

These building blocks can be combined into two basic matcher architectures:

(1)	Rule-based Matching
![Rule-based Matching](img/rule_based_matching.png)

(2)	Voting-based Matching
![Voting-based Matching](img/voting_based_matching.png)

## Implemented Blockers:
**[Standard Blocker](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/StandardBlocker.html)**: Uses a blocking key generator to define record pairs. One or multiple blocking keys are generated for each record. All records with the same blocking key form a “block”. All possible pairs of records from the same block are created as result.

**[Sorted Neighbourhood Method](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/blockers/SortedNeighbourhoodBlocker.html)**: The records are sorted by their blocking key. Then, a sliding window is applied to the data and each record is paired with its neighbours.

## Implemented Matching Rules:
**[LinearCombinationMatchingRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/LinearCombinationMatchingRule.html) (FilteringMatchingRule)**: Applies a pre-defined set of comparators to the pair of records and calculates a linear combination of their scores as similarity value.

**[VotingMatchingRule](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/rules/VotingMatchingRule.html) (AggregatableMatchingRule)**: Creates a correspondence for each causal correspondence provided by the blocker. In duplicate-based schema matching, the blocker creates the duplicates as causes for the attribute pairs and the voting rule then casts a vote for each duplicate.

## Implemented Aggregators:
**[CorrespondenceAggregator](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/aggregators/CorrespondenceAggregator.html)**: Aggregates the scores of correspondences in the same group. Possible Configurations: Sum, Count, Average, Voting

**[TopKAggregator](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/matching/aggregators/TopKAggregator.html)**: Filters out all but the k correspondences with the highest similarity per group.
