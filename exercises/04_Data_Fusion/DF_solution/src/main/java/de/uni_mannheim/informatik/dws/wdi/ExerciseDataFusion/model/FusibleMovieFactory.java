package de.uni_mannheim.informatik.dws.wdi.ExerciseDataFusion.model;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;

import de.uni_mannheim.informatik.dws.winter.model.FusibleFactory;
import de.uni_mannheim.informatik.dws.winter.model.RecordGroup;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.Attribute;

public class FusibleMovieFactory implements FusibleFactory<Movie, Attribute> {

	@Override
	public Movie createInstanceForFusion(RecordGroup<Movie, Attribute> cluster) {

		List<String> ids = new LinkedList<>();

		for (Movie m : cluster.getRecords()) {
			ids.add(m.getIdentifier());
		}

		Collections.sort(ids);

		String mergedId = StringUtils.join(ids, '+');

		return new Movie(mergedId, "fused");
	}
	
}
