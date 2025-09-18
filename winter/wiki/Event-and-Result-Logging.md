WInte.r is equipped with loggers to track events and results during the execution of the data integration process. The loggers inform the user about the current status of the data integration at all times, so that potential issues can be easily spotted. WInte.r distinguishes two types of loggers:
- **Event logging** informs the user about the application's status
- **Result logging** helps the user to better understand the impact of different configurations during the execution of the data integration process

# Event Logging
The event logging informs the user about the application's current status. The logging system is based on the [log4j2](https://logging.apache.org/log4j/2.x/manual/index.html) framework and comes with four predefined logging options:
* **default:** Sets the log level to INFO and writes the events to the console.
* **trace:** Sets the log level to TRACE and writes the events to the console.
* **infoFile:** Sets the log level to INFO and writes the events to the console as well as to the log file winter.log in the project's folder /logs/ .
* **traceFile:** Sets the log level to TRACE and writes the events to the console as well as to the log file winter.log in the project's folder /logs/ .

The used log levels are defined as follows:
* **TRACE:**  	Provides fine-grained informational events that describe data outputs from the data integration process.
* **INFO:** Provides informational events and summary statistics that highlight the progress of the integration process.
* **ERROR:** Provides error events that still allow the application to continue.

 To select one of these logging options, use the [WinterLogManager](https://github.com/olehmberg/winter/blob/master/docs/javadoc/de/uni_mannheim/informatik/dws/winter/utils/WinterLogManager.html) as shown below:

```java
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
```
The activated logger is used by all components except for the [ProgressReporter](https://github.com/olehmberg/winter/blob/master/docs/javadoc/de/uni_mannheim/informatik/dws/winter/utils/ProgressReporter.html) of the WInte.r framework to track events. The events created by the [ProgressReporter](https://github.com/olehmberg/winter/blob/master/docs/javadoc/de/uni_mannheim/informatik/dws/winter/utils/ProgressReporter.html) class only inform about the progress of the current task (e.g. 10%). Hence, a user may not be interested in this information after the execution of the data integration process, thus the [ProgressReporter](https://github.com/olehmberg/winter/blob/master/docs/javadoc/de/uni_mannheim/informatik/dws/winter/utils/ProgressReporter.html) uses the dedicated console logger "progress" to allow a separation of these log messages.

The configuration of the predefined logging options can be found in the [log4j2.xml](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/resources/log4j2.xml) file, which is customizable as described by the [log4j2 reference](https://logging.apache.org/log4j/2.x/manual/configuration.html). If you define your own [log4j2.xml](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/resources/log4j2.xml) file, please pass the file path of the [log4j2.xml](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/resources/log4j2.xml) file to the Java Virtual Machine via the following system property:

 *-Dlog4j.configurationFile=<path/to/>log4j2.xml*

## Example configurations
Log messages contain the date/time and the corresponding class name. To add the 
 [calling method's name](https://logging.apache.org/log4j/2.x/manual/layouts.html#PatternLayout), add %M to the *PatternLayout* of the desired appender (console in the example).
 
 *Use this pattern with care, as generating the method's name is an expensive operation, which may decrease the runtime performance.*

 ```xml
 <?xml version="1.0" encoding="UTF-8"?>
<Configuration status="WARN">
    <Appenders>
        <Console name="console" target="SYSTEM_OUT">
            <PatternLayout pattern="[%-5level] %d{yyyy-MM-dd HH:mm:ss.SSS} [%C-%M] - %msg%n"/>
        </Console>
        ...
    </Appenders>
    ...
</Configuration>
 ```

[To change the log file destination](https://logging.apache.org/log4j/2.x/manual/appenders.html#FileAppender), change the corresponding *fileName* of the file appender.

 ```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="WARN">
    <Appenders>
        <File name="file" fileName="<YOUR FILEPATH>/<YOUR FILENAME>.log">
            ...
        </File>
        ...
    </Appenders>
    ...
</Configuration>
 ```


To write the events from the [ProgressReporter](https://github.com/olehmberg/winter/blob/master/docs/javadoc/de/uni_mannheim/informatik/dws/winter/utils/ProgressReporter.html) to a file, [add the file appender to the progress logger](https://logging.apache.org/log4j/2.x/manual/configuration.html):

 ```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="WARN">
    ...
    <Loggers>
        ...
        <Logger name="progress" level="trace" additivity="false">
            ...
            <AppenderRef ref="file"/> 
        </Logger>
        ...
    </Loggers>
</Configuration>
 ```
# Result Logging

Result logging helps the user to better understand the impact of different configurations in order to improve the data integration process. A user can, for example, inspect in the data matching step how the similarity of two string values changes, when lowercasing these string values before applying the similarity metric. WInte.r provides specific result logs for the steps blocking, matching and data fusion in the data integration process.

*All result logging options have an impact on the runtime performance and are hence deactivated by default.*

## Blocking

Defining a blocking strategy is the first step of the identity resolution and can have drastic impacts on runtime and result quality.
To allow the designer of the identity resolution process to understand the impact of different design choices, such as blocker type or blocking key generation, all blockers can be configured to log additional information.

This log gives an overview of the blocking key values generated by the blocker and their frequencies (number of records for which this blocking key was generated = block size):

| Blocking Key Value | Frequency |
| ------------- | ------------- |
| 194  | 15600  |
| 196  | 11487  |
| ...  | ...  |

**Before running the blocking**, the user has to activate this feature by trigger the method `collectBlockSizeData()`. As inputs the method needs a path to the designated log file and a maximum size of the log to prevent a buffer overflow.

```java
blocker.collectBlockSizeData("data/output/debugResultsBlocking.csv", 1000);
```

After the execution of the blocking step, the results are written into the log file.

## Matching
Matching rules make use of comparators to calculate similarities between the records that should be matched.
These comparators may apply specific pre- or postprocessing steps as well as different similarity metrics. To better understand the individual processing steps and retrace the matching rule's final decision, WInte.r can log detailed information about each comparator's execution. There are two possible logging formats:

* **Comparator:** Logs details such as input values, pre-processed values, similarity score and post-processed similarity score for each comparator in a separate line. Can be used to debug individual comparators. The following table shows an example:

| MatchingRule | Record1Identifier | Record2Identifier | comparatorName | record1Value | record2Value | record1PreprocessedValue | record2PreprocessedValue | similarity | postproccesedSimilarity |
| ------------- | ------------- |------------- | ------------- | ------------- |------------- | ------------- | ------------- |------------- |------------- |
| LinearCombinationMatchingRule  | academy_awards_4098  | actors_3 | MovieTitleComparatorLowercaseLevenshtein  | The Wizard of Oz | The Divorcee | the wizard of oz | the divorcee | 0.375 | 0.375 |
| LinearCombinationMatchingRule  | academy_awards_4098  | actors_8 | MovieTitleComparatorLowercaseLevenshtein  | THE WIZARD OF OZ | Dangerous | the wizard of oz | dangerous  | 0.11 | 0.0 |
| ...  | ...  | ...| ...  | ...| ...  | ...  | ...| ...  | ...|


* **Matching Rule:** Logs all values which are created during a matching rule's execution in a single line. This comprises all values logged in the Comparator-format as well as the final similarity value as calculated by the matching rule. Additionally matching details from the goldstandard can be added if the goldstandard passed to the debug results log.
As a matching rule may contain multiple comparators, the header of the comparator's values is pre-fixed with the comparator's position, the comparator's name. For example, the value of record1 (record1Value in the Comparator-format) for the MovieDirectoryComparatorLevenshtein, which is the first comparator of the matching rule, has the header  '0-MovieDirectorComparatorLevenshtein-record1Value'.

| MatchingRule | Record1Identifier | Record2Identifier | TotalSimilarity | IsMatch| result of comparator1 | result of comparator2 | ... |
| ------------- | ------------- |------------- | ------------- | ------------- |------------- |------------- |------------- |
| LinearCombinationMatchingRule  | academy_awards_2631  | actors_40 | 0.8  | 1 |... | ... | ... | 
| LinearCombinationMatchingRule  | academy_awards_2370  | actors_110 | 0.0  | 0 | ... | ... |... | 
| ...  | ...  | ...| ...  | ...| ...  |... | 

**Before adding the comparators to the matching rule** and running the identity resolution, the user has to activate the result logging via the method `activateDebugReport()`. As inputs the method needs a path to the designated log file and a maximum size of the log to prevent a buffer overflow.

```java
// collect debug data
matchingRule.activateDebugReport
("usecase/movie/output/debugResultsMatchingRule.csv", 1000);
```

To additionally add ground truth information from the goldstandard to the results log, load and pass the goldstandard when activating the debug data collection:

```java
// load the gold standard (test set)
MatchingGoldStandard gsTest = new MatchingGoldStandard();
gsTest.loadFromCSVFile(new File(
		"usecase/movie/goldstandard/gs_academy_awards_2_actors_v2.csv"));

// collect debug data  
matchingRule.activateDebugReport
("usecase/movie/output/debugResultsMatchingRule.csv", 1000, gsTest);
```

After the execution of the identity resolution, the result are written into two separate log files for the comparator and the matching rule. Therefore, the path to the matching rule log has to be specified as stated below. WInte.r internally enhances this file path by '_short' to write the comparator log into a different file.

## Data Fusion

Different fusion strategies may be promising for resolving data conflicts. 
In order to allow the user to detect the best strategy in his or her case, WInte.r has the capability to provide for each group of fused values the corresponding values and the finally chosen value. 
Thereby, the impact of each fusion strategy becomes visible to the designer of the integration process, which allows him or her to fine-tune the fusion strategy.

For data fusion the results are logged on attribute and on record level.
The result log on attribute level contains for each group of fusible values a unique identifier, the corresponding values and the fused value. 
The unique identifier *ValueIDS* is defined by the *AttributeName* and a list of identifiers of the fused records (*RecordIDS*). 
The *Values* are a list of all attribute values of the group. 
Based on this list of values the *FusedValue* is selected by the fusion strategy. 
To evaluate the fusable values the *Consistency* is provided.
If a goldstandard is supplied, the columns *IsCorrect* and *CorrectValue* indicate if the *FusedValue* is correct and provide the correct value.


| AttributeName | Consistency | ValueIDS                                          | RecordIDS                              | Values                                   | FusedValue         | IsCorrect | CorrectValue       |
|---------------|-------------|---------------------------------------------------|----------------------------------------|------------------------------------------|--------------------|-----------|--------------------|
| Title         | 1           | Title-{academy_awards_4081+actors_12}             | academy_awards_4081+actors_12          | {Gone with the Wind, Gone With the Wind} | Gone with the Wind | TRUE      | Gone with the Wind |
| Director      | 1           | Director-{academy_awards_3305+golden_globes_2263} | academy_awards_3305+golden_globes_2263 | {William Wyler, William Wyler}           | William Wyler      |           |                    |
| ..            | ..          | ..                                                | ..                                     | ...                                      | ...                | ...       | ...                |

For the result log on record level the attribute level log is grouped by record.
The fused record is identified by the *RecordIDS* of the individual records.
For each attribute the *Consistency* and *Values* are provided.
To evaluate a complete record, the *AverageConsistency* across all attributes is calculated.
Fused records with a low *AverageConsistency* indicate that multiple conflicts are unsolved.
The truth record of these fused records is a good candidate to improve the gold standard for data fusion.
After checking the correct values via an external source, this record can be added to the gold standard for data fusion.

| RecordIDS                     | AverageConsistency | Title-Consistency | Title-Values                     | Actors-Consistency | Actors-Values                                                                                                | Director-Consistency | Director-Values   | Date-Consistency | Date-Values                          |
|-------------------------------|--------------------|-------------------|----------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------|----------------------|-------------------|------------------|--------------------------------------|
| academy_awards_4207+actors_10 | 0.75               | 1                 | {The Good Earth, The Good Earth} | 1                  | {\[\[Actor: Luise Rainer / null / null\]\],\[\[Actor: Luise Rainer /   1910-01-01T00:00 / Germany\]\]}       | 1                    | {Sidney Franklin} | 0                | {1937-01-01T00:00,1938-01-01T00:00} |
| academy_awards_3624+actors_17 | 0.25               | 1                 | {Gaslight, Gaslight}             | 0                  | {\[\[Actor: Charles Boyer / null / null\]\],\[\[Actor: Ingrid Bergman   / 1915-01-01T00:00 / Sweden\]\]}     | 0                    | {}                | 0                | {1944-01-01T00:00,1945-01-01T00:00} |

**Before adding different attribute fusers to the fusion strategy** and running the data fusion, the user has to activate this feature by trigger the method `activateDebugReport()`. As inputs the method needs a path to the designated log file and a maximum size of the log to prevent a buffer overflow.

```java
// collect debug results
strategy.activateDebugReport
("usecase/movie/output/debugResultsDatafusion.csv", 1000);
```

To additionally add ground truth information from the goldstandard to the results log, load and pass the goldstandard when activating the debug data collection:

```java
// load the gold standard (test set)
DataSet<Movie, Attribute> gsTest = new FusibleHashedDataSet<>();
    new MovieXMLReader().loadFromXML(new File("usecase/movie/goldstandard/fused.xml"), "/movies/movie", gs);

// collect debug data  
strategy.activateDebugReport
("usecase/movie/output/debugResultsDatafusion.csv", 1000, gsTest);
```


After the execution of the data fusion step, the results are written into two result log files - one on attribute level and one on record level.
