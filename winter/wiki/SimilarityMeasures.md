Similarity measures define a function which compares two values and returns a number that indicates the similarity of the values.

All similarity measures in WInte.r extend the [`SimilarityMeasure`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/SimilarityMeasure.java) class:

```java
public abstract class SimilarityMeasure<DataType> implements Serializable {

	private static final long serialVersionUID = 1L;

	/**
	 * Calculates the similarity of first and second
	 * 
	 * @param first
	 *            the first record (must not be null)
	 * @param second
	 *            the second record (must not be null)
	 * @return the similarity of first and second
	 */
	public abstract double calculate(DataType first, DataType second);

}
```

Users can implement any similarity measure by extending this class, but WInte.r already provides several similarity measures, which can be found in the [`similarity`](https://github.com/olehmberg/winter/tree/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity) package and are described in the following.

## String Similarity

- **Jaccard Similarity on Word Tokens**: The class [`TokenizingJaccardSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/TokenizingJaccardSimilarity.java) tokenises strings and then calculates the Jaccard similarity on the resulting token sets.

- **Jaccard Similarity on N-Grams**: The class [`JaccardOnNGramsSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/JaccardOnNGramsSimilarity.java) splits strings into n-grams and then calculates the Jaccard similarity on the resulting n-gram sets.

- **Generalised Jaccard Similarity**: The class [`GeneralisedStringJaccard`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/GeneralisedStringJaccard.java) tokenises strings and then calculates the generalised Jaccard similarity on the resulting token sets, which compares tokens with an inner similarity measure (instead of equality as in standard Jaccard similarity).

- **Levenshtein Edit Distance**: The class [`LevenshteinEditDistance`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/LevenshteinEditDistance.java) calculates the absolute edit distance (= number of insert, delete or replace operations required to transform one string into the other)

- **Levenshtein Similarity**: The class [`LevenshteinSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/LevenshteinSimilarity.java) calculates a relative similarity value based on the Levenshtein edit distance by dividing the edit distance by the number of characters in the longest of both strings.

- **Maximum Of Token Containment**: The class [`MaximumOfTokenContainment`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/MaximumOfTokenContainment.java) tokenises strings, calculates the size of intersection of the token sets of both strings divided by the size of the token set of either string, and returns the maximum of both values.

- **TF-IDF Cosine Similarity**: The class [`TFIDFCosineSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/TFIDFCosineSimilarity.java) tokenises strings based on a user-defined [`TokenGenerator`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/string/generator/TokenGenerator.java), computes the term frequency-inverse document frequencies of the two token sets, and calculates the cosine similarity of the resulting TF-IDF vectors. 

## Numeric Similarity

- **Absolute Differences Similarity**: The class [`AbsoluteDifferenceSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/numeric/AbsoluteDifferenceSimilarity.java) calculates the absolute difference between two numbers and divides it by a user-specified maximum difference. If the difference exceeds the maximum difference, the similarity is 0.

- **Deviation Similarity**: The class [`DeviationSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/numeric/DeviationSimilarity.java) divides the smaller number by the larger number and rescales all similarity values below 1 to the range [0, 0.5].

- **Unadjusted Deviation Similarity**: The class [`UnadjustedDeviationSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/numeric/UnadjustedDeviationSimilarity.java) divides the smaller number by the larger number.

- **Percentage Similarity**: The class [`PercentageSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/numeric/PercentageSimilarity.java) divides the absolute difference between two number by the maximum of both numbers.

## Date Similarity

- **Day Similarity**: The class [`DaySimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/date/DaySimilarity.java) divides the absolute difference in days between two dates and divides it by a user-specified maximum difference.

- **Year Similarity**: The class [`YearSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/date/YearSimilarity.java) divides the absolute difference in years between two dates and divides it by a user-specified maximum difference.

- **WeightedDateSimilarity**: The class [`WeightedDateSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/date/WeightedDateSimilarity.java) calculates a weighted average of the similarity between two dates' day, month, and year part. The weights are specified by the user.

## Geo Coordinate Similarity

- **Geo Coordinate Similarity**: The class [`GeoCoordinateSimilarity`](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/similarity/geo/GeoCoordinateSimilarity.java) calculates the similarity of two geo coordinates based on the distance in kilometers between them. If the distance exceeds the maximum distance, the similarity is 0.
