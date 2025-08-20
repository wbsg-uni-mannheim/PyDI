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
package de.uni_mannheim.informatik.dws.winter.processing.aggregators;

import de.uni_mannheim.informatik.dws.winter.model.Pair;
import de.uni_mannheim.informatik.dws.winter.processing.DataAggregator;

/**
 * {@link DataAggregator} that sums up the elements.
 * 
 * @author Oliver Lehmberg (oli@dwslab.de)
 *
 */
public abstract class SumAggregator<KeyType, RecordType> implements DataAggregator<KeyType, RecordType, Double> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public abstract Double getValue(RecordType record);
	
	@Override
	public Pair<Double,Object> aggregate(Double previousResult,
			RecordType record, Object state) {
		if(previousResult==null) {
			return stateless(getValue(record));
		} else {
			return stateless(previousResult+getValue(record));
		}
	}
	
	/* (non-Javadoc)
	 * @see de.uni_mannheim.informatik.wdi.processing.DataAggregator#initialise(java.lang.Object)
	 */
	@Override
	public Pair<Double,Object> initialise(KeyType keyValue) {
		return stateless(null);
	}
	
	/* (non-Javadoc)
	 * @see de.uni_mannheim.informatik.dws.winter.processing.DataAggregator#merge(de.uni_mannheim.informatik.dws.winter.model.Pair, de.uni_mannheim.informatik.dws.winter.model.Pair)
	 */
	@Override
	public Pair<Double, Object> merge(Pair<Double, Object> intermediateResult1,
			Pair<Double, Object> intermediateResult2) {
		return stateless(intermediateResult1.getFirst()+intermediateResult2.getFirst());
	}
	
}
