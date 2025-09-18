WInte.r does not require any specific data model for its operations. All methods implemented in the MatchingEngine expect objects implementing the Matchable interface, and the DataFusionEngine methods expect objects implementing the Fusable interface.
For a quick start, you can use the default model, which has pre-implemented readers of CSV and XML files.
The default model provides simple classes to represent [Attribute](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/Attribute.html)s and [Record](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/Record.html)s.

However, if you decide to extensively work with data from a specific domain, it is recommended to implement a dedicated data model for this domain. This specific data model equips you with an additional layer of abstraction during the data integration process. For demoing purposes, WInte.r contains example use cases like a [movie use case](https://github.com/olehmberg/winter/tree/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies) and a [city use case](https://github.com/olehmberg/winter/tree/preprocessingdefaultmodel/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/cities) including specific data models with readers for well-known CSV and XML files.
A record in this user-specific model is represented by a specifc record like a [movie record](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/Movie.java) or a [city record](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/cities/model/City.java).

In general, default and user-specific records are loaded into a [DataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/DataSet.html), which contains records and their schema, represented by attributes.
Two implementations of a dataset are already provided: the [HashedDataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/HashedDataSet.html) and the [ParallelHashDataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/ParallelHashedDataSet.html).

# Datasets

The default implementation of a [DataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/DataSet.html) is the [HashedDataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/HashedDataSet.html), which is a `ProcessableCollection` and hence uses single-threaded processing. Analogously, the [ParallelHashedDataSet](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/ParallelHashedDataSet.html) is a `ParallelProcessableCollection` and uses multi-threaded processing.

Datasets can store records and metadata, i.e., attributes.
To access the metadata, use the `getSchema()` function, which returns another dataset with the metadata (you could actually manage another level of metadata by calling `getSchema()` on the metadataset).

# Collections

All collections in WInte.r implement the [Processable](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/processing/Processable.html) interface, which enables a replaceable processing archtecture behind the matching and fusion methods. The default implementation of a `Processable` is [ProcessableCollection](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/processing/ProcessableCollection.html), which uses a single-threaded processing model. For multi-threaded processing, load your data into a [ParallelProcessableCollection](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/processing/parallel/ParallelProcessableCollection.html).

To access the elements in a `Processable` use the `get()` function, which returns a java `Collection`.

```java
Processable<String> myProcessable = new ProcessableCollection<>();
myProcessable.add("a");
myProcessable.add("bb");
myProcessable.add("ccc");
for(String element : myProcessable.get()) {
	System.out.println(element);
}
```
```
Output:
	a
	bb
	ccc
```

**Note**: All implementations in WInte.r use the methods defined in `Processable` to manipulate data. If you need to optimise for performance, you should consider defining you own `Processable` with custom implementations of the processing operations.

```java
Processable<String> myProcessable = new ProcessableCollection<>();
myProcessable.add("a");
myProcessable.add("bb");
myProcessable.add("ccc");
myProcessable.add("a");
myProcessable.add("bb");
myProcessable.add("ccc");
String result = myProcessable
				.distinct()
				.filter((s)->s.length()>1)
				.take(1)
				.firstOrNull();
System.out.println(result);
```
```
Output:
	bb
```


# Default Model 

## Records and Attributes

The [Record](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/Record.html) class represents records in a dataset. It implements both the `Matchable` and `Fusible` interfaces.
The record class can manage attributes with atomic values and attributes with lists of values.
To access values, you need to define attributes and pass them to the `setValue`/`getValue` or `setList`/`getList` methods.

The [Attribute](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/Attribute.html) class represents attributes in a dataset. It implements the `Matchable` interface.
The only data that an attribute has is its name.

```java
	Attribute name = new Attribute("name");
	Attribute phoneNumbers = new Attribute("phoneNumbers");

	Record record = new Record("record1");
	record.setValue(name, "the name of the entity");
	record.setList(phoneNumbers, Arrays.toList(
		new String[] {
			"001",
			"002",
			"003"
		}));

	logger.info(String.format("Name: %s", record.getValue(name)));
	for(String pn : record.getList(phoneNumbers)) {
		logger.info(pn);
	}
```

```
Output:
	Name: the name of the entity
	001
	002
	003
```

## Loading a Dataset
WInte.r provides implementations to load data from CSV and XML files using the default data model:

### CSV
Reading a CSV file: The [CSVRecordReader](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/CSVRecordReader.html) handles the CSV reading for the default data model and assumes that the first row contains the attribute names. You must specify the index of the column that contains the unique identifier in the constructor (if none exists, use -1).

```java
// create a dataset
DataSet<Record, Attribute> data1 = new HashedDataSet<>();
// load data into the dataset
new CSVRecordReader(0).loadFromCSV(new File("usecase/movie/input/scifi1.csv"), data1);
```
If no header row is available, a mapping can be provided to connect an attribute to a column.

```java
// generate mapping
Map<String, Attribute> columnMappingITunes = new HashMap<>();

columnMappingITunes.put("1", iTunesSong.PAGE);
columnMappingITunes.put("2", iTunesSong.URI0);
columnMappingITunes.put("3", iTunesSong.URI1);
columnMappingITunes.put("4", iTunesSong.URI2);
columnMappingITunes.put("5", iTunesSong.URI3);
columnMappingITunes.put("6", iTunesSong.POSITION);
columnMappingITunes.put("7", iTunesSong.NAME);
columnMappingITunes.put("8", iTunesSong.ARTIST);
columnMappingITunes.put("9", iTunesSong.TIME);

// load data
DataSet<Record, Attribute> dataITunes = new HashedDataSet<>();
new CSVRecordReader(0, columnMappingITunes).loadFromCSV(new File("usecase/itunes/input/itunes.csv"),
				dataITunes);
```

### XML
Reading an XML file: The [XMLRecordReader](https://olehmberg.github.io/winter/javadoc/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/XMLRecordReader.html) handles the XML reading for the default data model and requires a map that assigns an `Attribute` to each XML Tag. Additionally, you must specify the XML tag that contains the record ids and an XPath expression the returns all records from the XML file.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<movies>
...
	<movie>
		<id>movie_list_3</id>
		<title>Alice in Wonderland</title>
		<studio>Disney</studio>
	</movie>
...
```
```java
Map<String, Attribute> nodeMapping = new HashMap<>();
nodeMapping.put("title", new Attribute("title"));
nodeMapping.put("studio", new Attribute("studio"));
new XMLRecordReader("id", nodeMapping).loadFromXML(sourceFile, "/movies/movie", dataset);
```
# User-specified Model
## Records
The [movie record](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/Movie.java) class represents movie records in a dataset. It extends the abstract class `AbstractRecord`.
To hold a movie's information, the [movie record](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/Movie.java) has attributes like 'title' or 'director'. When implementing these attributes, you should know whether and where they are contained in the input files. You can access these attributes via distinct 'get' and 'set' methods.

```java
public class Movie extends AbstractRecord<Attribute> implements Serializable {

    ...

	public Movie(String identifier, String provenance) {
		super(identifier, provenance);
		actors = new LinkedList<>();
	}

	private String title;
	private String director;
	private LocalDateTime date;
	private List<Actor> actors;
	private String studio;
	private String genre;
	private double budget;
	...

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}
    ...
}
```

## Loading a Dataset
WInte.r provides example implementations to load data from CSV and XML files using a user-specific data model:

### CSV
Reading a CSV file: The [CSVCityReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/cities/model/CSVCityReader.java) handles the CSV reading for the city data model.

| id | countryCode | name      | latitude | longitude   | officialName | country | population |
|-------|-------------|-----------|---------|-------------|--------------|---------|------------------|
| 0     | IL          | Ashqelon  | 31.66926 | 34.57149 | אשקלון       | Israel  | 105.995 thousand |
| 1     | IQ          | Ctesiphon | 33.08333 | 44.58333     |              | Iraq    |                  |
| 2     | FI          | Espoo     | 60.25    | 24.66667     | Espoo        | Finland | 256.76 thousand  |
| ...     | ...          | ...     | ...    | ...     | ...        | ... | ...  |

```java	
// load data
DataSet<City, Attribute> dataCity = new HashedDataSet<>();
new CSVCityReader().loadFromCSV(new File("usecase/city/input/city.csv"), dataCity);
```

Whenever a new line is read from the target csv file via `readLine()`, a new city record is generated and filled as shown below. As the format of the csv file is known upfront, you can directly pass a new value from an input line 'values' to the distinct attribute of the city record. Afterwards, the new city record is added to the data set.

```java	
@Override
protected void readLine(File file, int rowNumber, String[] values, DataSet<City, Attribute> dataset) {

	if (rowNumber == 0) {
		initialiseDataset(dataset);
	} else {
		// generate new record of type city
		City r = new City(values[0], file.getAbsolutePath());

		// set values of record
		r.setCountryCode(values[1]);
		r.setName(values[2]);
			
		...

		dataset.add(r);

	}

}

```
*Hint: As you can see in the city input table above, the column 'population' contains the unit 'thousand', thus it should be converted before processing it further. Check the wiki page on [Data Normalization](TODO:link) for more details.*

### XML
Reading an XML file: The [MovieXMLReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLReader.java) handles the XML reading for the movie data model. As the format of the xml is known upfront, no additional details have to be passed to the [MovieXMLReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLReader.java).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<movies>
...
	<movie>
		<id>movie_list_3</id>
		<title>Alice in Wonderland</title>
		<studio>Disney</studio>
	</movie>
...
```
```java
HashedDataSet<Movie, Attribute> dataAcademyAwards = new HashedDataSet<>();
new MovieXMLReader().loadFromXML(new File("usecase/movie/input/academy_awards.xml"), "/movies/movie", dataAcademyAwards);
```

The mapping information between attribute and xml tag directly implemented into the [MovieXMLReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/model/MovieXMLReader.java), which generates the movie records from the xml file.

```java
public class MovieXMLReader extends XMLMatchableReader<Movie, Attribute> implements
		FusibleFactory<Movie, Attribute> {

	...
	@Override
	public Movie createModelFromElement(Node node, String provenanceInfo) {
		String id = getValueFromChildElement(node, "id");

		// create the object with id and provenance information
		Movie movie = new Movie(id, provenanceInfo);

		// fill the attributes
		movie.setTitle(getValueFromChildElement(node, "title"));
        
		...
		return movie;
	}
}

```
