Web Tables are tables that were extracted from HTML pages. Large corpora of such tables are provided by the [Web Data Commons (WDC) Project](http://www.webdatacommons.org/webtables/index.html). WInte.r includes parsers for both formats of [Web Tables published by WDC](http://www.webdatacommons.org/webtables/2015/downloadInstructions.html) (CSV and JSON) as well as methods for header detection, data type detection and subject column detection.

## Reading / Writing Web Tables

Web Tables are represented by the `Table` class and can be loaded using a `TableParser`.

Reading/Writing a CSV table:

```java
// read the table
CsvTableParser csvParser = new CsvTableParser();
Table t = csvParser.parseTable(file);
// write the table
CSVTableWriter csvWriter = new CSVTableWriter();
file = csvWriter.write(t, file);
```

Reading/Writing a JSON table:

```java
// read the table
JsonTableParser jsonParser = new JsonTableParser();		
Table t = jsonParser.parseTable(file);
// write the table
JSONTableWriter jsonWriter = new JSONTableWriter();
file = jsonWriter.write(t, file);
```

## Detecting Data Types
The `TableParser`s internally use a type detection to determine the data types of the columns in the table. The default type detection is the `TypeGuesser`, which uses pattern-matching rules to determine the data types. Alternatively, the ` TypeClassifier` uses machine learning for the type detection (with a mixture of features from [1-3]). You can specify one of these two detectors by passing them to the constructor of your `TableParser`. By implementing the interface `TypeDetector`, you can define a custom type detector.

```java
CsvTableParser csvParser = new CsvTableParser(new TypeClassifier());
JsonTableParser jsonParser = new JsonTableParser(new TypeClassifier());
```

## Detecting Header Rows
The `TableParser`s automatically apply a header detection and use the detected header to set the attribute names (=column headers). Column headers can be accesses using the `Table`s `getSchema()` function.

```java
TableSchema schema = t.getSchema();

for(int i = 0; i < schema.getSize(); i++){
  System.out.println(schema.get(i).getHeader());
}
```

To change the header detection method, implement the `TableHeaderDetector` interface.

```java
public class TableHeaderDetectorFirstRow implements TableHeaderDetector {

	@Override
	public int[] detectTableHeader(String[][] attributeValues) {
		int[] result = {0};
		return result;
	}
}
```

Then assign it to your `TableParser` using the `setTableHeaderDetector()` method before parsing a table.

```java
CsvTableParser csvParser = new CsvTableParser();
csvParser.setTableHeaderDetector(new TableHeaderDetectorFirstRow());
```

## Detecting the Entity Label Column / Subject Column

The Entity Label Column or Subject Column is a single column that likely contains the names of the entities that are described in a table. It can be detected by calling the `identifySubjectColumn()` method, which expects a uniqueness threshold as parameter.

```java
CsvTableParser csvParser = new CsvTableParser();
Table t = csvParser.parseTable(pFilePath);

t.identifySubjectColumn(0.9);

TableColumn subjectColumn = t.getSubjectColumn();
```

## Accessing Table Data

The `Table` class gives you access to the table's data with the `getRows()` function and to the tables schema with the `getColumns()` and `getSchema()` functions.


```java
// load a table from json
JsonTableParser p = new JsonTableParser();
Table t = p.parseTable(f);

// print some metda-data: number of columns & rows, subject column
System.out.println(String.format("* # Columns: %d", t.getColumns().size()));
System.out.println(String.format("* # Rows: %d", t.getRows().size()));
System.out.println(String.format("* Subject Column: %s",
  t.getKey()==null ? "?" : t.getSubjectColumn().getHeader()));

// print the table schema (all column names including column index and data type):
for(TableColumn c : t.getColumns()) {
  System.out.println(String.format("[%d] %s (%s)",
    c.getColumnIndex(),
    c.getHeader(),
    c.getDataType()));
}

// print all rows including their row number
// (with a maximum column width of 20 characters):for(TableRow r : t.getRows()) {
  System.out.println(String.format("[%d] %s",
    r.getRowNumber(),
    r.format(20)));
}
```

## References

[1] Venetis, Petros, et al. "Recovering semantics of tables on the web." Proceedings of the VLDB Endowment 4.9 (2011): 528-538.

[2] Bakalov, Anton, et al. "Scad: Collective discovery of attribute values." Proceedings of the 20th international conference on World wide web. ACM, 2011.

[3] Kolomiyets, Oleksandr, and Marie-Francine Moens. "KUL: recognition and normalization of temporal expressions." Proceedings of the 5th International Workshop on Semantic Evaluation. Association for Computational Linguistics, 2010.
