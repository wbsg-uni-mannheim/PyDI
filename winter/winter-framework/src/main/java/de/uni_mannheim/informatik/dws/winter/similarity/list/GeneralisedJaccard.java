/*
 * Copyright (c) 2017 Data and Web Science Group, University of Mannheim, Germany (http://dws.informatik.uni-mannheim.de/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

package de.uni_mannheim.informatik.dws.winter.similarity.list;

import de.uni_mannheim.informatik.dws.winter.matrices.SimilarityMatrix;
import de.uni_mannheim.informatik.dws.winter.matrices.matcher.BestChoiceMatching;
import de.uni_mannheim.informatik.dws.winter.similarity.SimilarityMeasure;

/**
 * Jaccard similarity with an inner similarity function for token comparisons.
 * @author Oliver Lehmberg (oli@dwslab.de)
 *
 * @param <T>
 */
public class GeneralisedJaccard<T extends Comparable<? super T>> extends ComplexSetSimilarity<T> {

	private static final long serialVersionUID = 1L;

    public GeneralisedJaccard(SimilarityMeasure<T> innerSimilarity, double innerSimilarityThreshold) {
		setInnerSimilarity(innerSimilarity);
		setInnerSimilarityThreshold(innerSimilarityThreshold);
	}

	@Override
	protected Double aggregateSimilarity(SimilarityMatrix<T> matrix) {
        double firstLength = matrix.getFirstDimension().size();
        double secondLength = matrix.getSecondDimension().size();
        
        BestChoiceMatching best = new BestChoiceMatching();
        best.setForceOneToOneMapping(true);
        matrix = best.match(matrix);
        
        double fuzzyMatching = matrix.getSum();
        
        return fuzzyMatching / (firstLength + secondLength - fuzzyMatching);
	}
}
