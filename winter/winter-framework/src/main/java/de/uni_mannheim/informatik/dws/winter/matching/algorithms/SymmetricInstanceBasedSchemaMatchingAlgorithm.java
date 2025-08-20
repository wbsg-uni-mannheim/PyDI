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


package de.uni_mannheim.informatik.dws.winter.matching.algorithms;

import de.uni_mannheim.informatik.dws.winter.matching.aggregators.CorrespondenceAggregator;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.SymmetricBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.rules.IdentityMatchingRule;
import de.uni_mannheim.informatik.dws.winter.model.Correspondence;
import de.uni_mannheim.informatik.dws.winter.model.DataSet;
import de.uni_mannheim.informatik.dws.winter.model.Matchable;
import de.uni_mannheim.informatik.dws.winter.model.MatchableValue;
import de.uni_mannheim.informatik.dws.winter.model.Pair;
import de.uni_mannheim.informatik.dws.winter.processing.Processable;

/**
 * 
 * A variant of the {@link InstanceBasedSchemaMatchingAlgorithm} that uses only a single input dataset.
 * 
 * @author Oliver Lehmberg (oli@dwslab.de)
 *
 * @param <RecordType>
 * @param <SchemaElementType>
 */
public class SymmetricInstanceBasedSchemaMatchingAlgorithm<RecordType extends Matchable, SchemaElementType extends Matchable> implements MatchingAlgorithm<SchemaElementType, MatchableValue> {

	public SymmetricInstanceBasedSchemaMatchingAlgorithm(DataSet<RecordType, SchemaElementType> dataset,
			SymmetricBlocker<RecordType, SchemaElementType, SchemaElementType, MatchableValue> blocker,
			CorrespondenceAggregator<SchemaElementType, MatchableValue> aggregator) {
		this.dataset = dataset;
		this.blocker = blocker;
		this.aggregator = aggregator;
	}
	
	DataSet<RecordType, SchemaElementType> dataset;
	SymmetricBlocker<RecordType, SchemaElementType, SchemaElementType, MatchableValue> blocker;
	CorrespondenceAggregator<SchemaElementType, MatchableValue> aggregator;
	Processable<Correspondence<SchemaElementType, MatchableValue>> result;
	
	/**
	 * @return the dataset1
	 */
	public DataSet<RecordType, SchemaElementType> getDataset() {
		return dataset;
	}
	
	/* (non-Javadoc)
	 * @see de.uni_mannheim.informatik.wdi.matching.algorithms.MatchingAlgorithm#run()
	 */
	@Override
	public void run() {
		// run the blocker to generate initial correspondences between the schema elements 
		Processable<Correspondence<SchemaElementType, MatchableValue>> blocked = blocker.runBlocking(getDataset(), null);
		
		// aggregate the correspondences to calculate a similarity score
		Processable<Pair<Pair<SchemaElementType, SchemaElementType>, Correspondence<SchemaElementType, MatchableValue>>> aggregated = blocked.aggregate(new IdentityMatchingRule<SchemaElementType, MatchableValue>(0.0), aggregator);
		
		// transform the result to the expected correspondence format
		Processable<Correspondence<SchemaElementType, MatchableValue>> result = aggregated.map((p, collector) -> {
			if(p.getSecond()!=null)
			{
				collector.next(p.getSecond());
			}
		});
		
		Correspondence.setDirectionByDataSourceIdentifier(result);
	
		setResult(result);
	}

	/**
	 * @param result the result to set
	 */
	public void setResult(Processable<Correspondence<SchemaElementType, MatchableValue>> result) {
		this.result = result;
	}
	
	/* (non-Javadoc)
	 * @see de.uni_mannheim.informatik.wdi.matching.algorithms.MatchingAlgorithm#getResult()
	 */
	@Override
	public Processable<Correspondence<SchemaElementType, MatchableValue>> getResult() {
		return result;
	}

}
