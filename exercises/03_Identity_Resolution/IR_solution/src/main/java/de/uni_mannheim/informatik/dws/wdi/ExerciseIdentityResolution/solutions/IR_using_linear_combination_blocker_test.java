package de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.solutions;

import java.io.File;
import java.time.Duration;
import java.time.LocalDateTime;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.slf4j.Logger;

import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.Blocking.MovieBlockingKeyByDecadeGenerator;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.Blocking.MovieBlockingKeyByTitleGenerator;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.Blocking.MovieBlockingKeyByYearGenerator;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.Comparators.MovieDateComparator2Years;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.Comparators.MovieTitleComparatorJaccard;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.model.Movie;
import de.uni_mannheim.informatik.dws.wdi.ExerciseIdentityResolution.model.MovieXMLReader;
import de.uni_mannheim.informatik.dws.winter.matching.MatchingEngine;
import de.uni_mannheim.informatik.dws.winter.matching.MatchingEvaluator;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.Blocker;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.NoBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.SortedNeighbourhoodBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.StandardRecordBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.rules.LinearCombinationMatchingRule;
import de.uni_mannheim.informatik.dws.winter.matching.rules.MatchingRule;
import de.uni_mannheim.informatik.dws.winter.model.Correspondence;
import de.uni_mannheim.informatik.dws.winter.model.DataSet;
import de.uni_mannheim.informatik.dws.winter.model.HashedDataSet;
import de.uni_mannheim.informatik.dws.winter.model.MatchingGoldStandard;
import de.uni_mannheim.informatik.dws.winter.model.Performance;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.Attribute;
import de.uni_mannheim.informatik.dws.winter.processing.Processable;
import de.uni_mannheim.informatik.dws.winter.utils.WinterLogManager;

public class IR_using_linear_combination_blocker_test 
{
	
	/*
	 * Logging Options:
	 * 		default: 	level INFO	- console
	 * 		trace:		level TRACE     - console
	 * 		infoFile:	level INFO	- console/file
	 * 		traceFile:	level TRACE	- console/file
	 *  
	 * To set the log level to trace and write the log to winter.log and console, 
	 * activate the "traceFile" logger as follows:
	 *     private static final Logger logger = WinterLogManager.activateLogger("traceFile");
	 *
	 */

	private static final Logger logger = WinterLogManager.activateLogger("default");
	
    public static void main( String[] args ) throws Exception
    {
		// loading data
		logger.info("*\tLoading datasets");
		HashedDataSet<Movie, Attribute> dataAcademyAwards = new HashedDataSet<>();
		new MovieXMLReader().loadFromXML(new File("data/input/academy_awards.xml"), "/movies/movie", dataAcademyAwards);
		HashedDataSet<Movie, Attribute> dataActors = new HashedDataSet<>();
		new MovieXMLReader().loadFromXML(new File("data/input/actors.xml"), "/movies/movie", dataActors);

		// load the gold standard (test set)
		logger.info("*\tLoading gold standard\t*");
		MatchingGoldStandard gsTest = new MatchingGoldStandard();
		gsTest.loadFromCSVFile(new File(
				"data/goldstandard/gs_academy_awards_2_actors_test.csv"));
		
		// create a blocker (blocking strategy)
		Blocker<Movie,Attribute,Movie,Attribute> blocker = new StandardRecordBlocker<>(new MovieBlockingKeyByTitleGenerator());

		logger.info("*\tNo Blocker\t*");
		blocker = new NoBlocker<>();
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		logger.info("*\tStandard Blocker: by title\t*");
		blocker = new StandardRecordBlocker<>(new MovieBlockingKeyByTitleGenerator());
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		logger.info("*\tStandard Blocker: by decade\t*");
		blocker = new StandardRecordBlocker<>(new MovieBlockingKeyByDecadeGenerator());
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		logger.info("*\tStandard Blocker: by year\t*");
		blocker = new StandardRecordBlocker<>(new MovieBlockingKeyByYearGenerator());
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		logger.info("*\tSorted-Neighbourhood Blocker: by year\t*");
		blocker = new SortedNeighbourhoodBlocker<>(new MovieBlockingKeyByYearGenerator(), 30);
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		logger.info("*\tSorted-Neighbourhood Blocker: by title\t*");
		blocker = new SortedNeighbourhoodBlocker<>(new MovieBlockingKeyByTitleGenerator(), 30);
		testBlocker(blocker, dataAcademyAwards, dataActors, gsTest);

		
	}
	
	protected static void testBlocker(Blocker<Movie,Attribute,Movie,Attribute> blocker, DataSet<Movie,Attribute> ds1, DataSet<Movie,Attribute> ds2, MatchingGoldStandard gsTest) {

    	// Initialize Matching Engine
		MatchingEngine<Movie, Attribute> engine = new MatchingEngine<>();

		// Execute the blocking
		LocalDateTime start = LocalDateTime.now();
		Processable<Correspondence<Movie, Attribute>> correspondences = engine.runBlocking(ds1, ds2, null, blocker);
		LocalDateTime afterBlocking = LocalDateTime.now();

		logger.info(String
				.format("Blocking %,d x %,d elements after %s; %,d blocked pairs (reduction ratio: %s)",
						ds1.size(), ds2.size(),
						DurationFormatUtils.formatDurationHMS(Duration.between(start, afterBlocking).toMillis()),
						correspondences.size(), Double.toString(blocker.getReductionRatio())));

		// evaluate your result
		logger.info("*\tEvaluating result\t*");
		MatchingEvaluator<Movie, Attribute> evaluator = new MatchingEvaluator<Movie, Attribute>();
		Performance perfTest = evaluator.evaluateMatching(correspondences,
			gsTest);



		// print the evaluation result
		logger.info("Academy Awards <-> Actors");
		logger.info(String.format(
			"Precision: %.4f",perfTest.getPrecision()));
		logger.info(String.format(
			"Recall: %.4f",	perfTest.getRecall()));
		logger.info(String.format(
			"F1: %.4f \n",perfTest.getF1()));
	}
}
