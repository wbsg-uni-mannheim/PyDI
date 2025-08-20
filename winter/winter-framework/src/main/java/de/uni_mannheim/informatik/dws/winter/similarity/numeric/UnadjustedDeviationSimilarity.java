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
package de.uni_mannheim.informatik.dws.winter.similarity.numeric;

import de.uni_mannheim.informatik.dws.winter.similarity.SimilarityMeasure;

/**
 * returns (smaller value / larger value)
 * 
 * @author Oliver Lehmberg (oli@dwslab.de)
 *
 */
public class UnadjustedDeviationSimilarity extends SimilarityMeasure<Double> {

	private static final long serialVersionUID = 1L;

	@Override
    public double calculate(Double first, Double second) {                
        if(first==null || second == null) {
            return 0.0;
        }
        if(first.equals(second)) {
            return 1.0;
        }
        else {
            return Math.min(Math.abs(first),Math.abs(second))/Math.max(Math.abs(first),Math.abs(second));
        }
    }   
}
