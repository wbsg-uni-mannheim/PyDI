This page gives an overview about the data value normalization features that are provided by WInte.r. Data normalization is a pre-processing step in the data integration process that is done to harmonize data values across all datasets. For example, values like *"1.5 thousand metre"* and *"1.5 kilometre"* are both converted to *"1,500"*. Thereby, both values are represented in the respective base unit metre.

# Data Value Normalization

WInte.r supports the parsing of data values with various data types: boolean, link, coordinate, date, numeric, and string. Input data values can be parsed, converted into the respective Java type, and normalised with the [ValueNormalizer](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/preprocessing/datatypes/ValueNormalizer.java) class.

For the data type numeric, further parsing of quantities (`thousand`, `million`) and units of measurement (`km`, `squareMile`, `joule`) are supported.
During normalisation, these quantities and units of measurement are handled by multiplying the numeric value with the respective factors.
For quantities this simply means scaling the value, i.e., normalising `1.5 million` to `1,500,000`.
For units, the value is converted into an [SI base unit](https://en.wikipedia.org/wiki/International_System_of_Units), i.e., the data value `1.5 km` is converted to `1,500`, which represents the value in metres.
All supported units are listed in the table below.

All value normalisations can be executed using the [ValueNormalizer](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/preprocessing/datatypes/ValueNormalizer.java)'s `normalize()` method. The method takes as input the data value as well as the desired data type and unit category.

```java
ValueNormalizer valueNormalizer = new ValueNormalizer();
Double value = (Double) valueNormalizer.normalize("1.5 thousand km", 
                         DataType.numeric, UnitCategoryParser.getUnitCategory("Distance"));
```
In this data value normalization example, the data value `1.5 thousand km` is normalized to the value `1,500,000`, which represents the original value in the base unit `metre`. First the data value's quantity `thousand` is detected, which results in a multiplication of the numeric value by `1,000`. Then the unit `km` is recognized, which results in another multiplication by `1,000` when converting the numeric value to the base unit `metre`. Afterwards, the quantity and unit are removed from the original value and a double-precision floating point number (Java's `Double` type) is returned.

A user can define the desired unit conversion by specifying a unit category such as `Distance` or `Area`.
The `ValueNormaliser` then checks for any unit belonging to this category and converts the value to the corresponding base unit.
The following table provides an overview of the supported units and their unit categories:

| Unit Category     | Units                                                                                                                                                                                                                                                                                                                                             | Base Unit                     |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Area              |  squareMetre, squareMillimetre, squareCentimetre, squareDecimetre, squareDecametre, squareHectometre, squareKilometre, hectare, squareInch, squareFoot, squareYard, acre, squareMile, squareNauticalMile                                                                                                                                          | squareMetre                   |
| Density           |  kilogramPerCubicMetre, kilogramPerLitre, gramPerCubicCentimetre, gramPerMillilitre                                                                                                                                                                                                                                                               | kilogramPerCubicMetre         |
| ElectricCurrent   |  ampere, kiloampere, milliampere, microampere                                                                                                                                                                                                                                                                                                     | ampere                        |
| Energy            |  joule, kilojoule, erg, milliwattHour, wattHour, kilowattHour, megawattHour, gigawattHour, terawattHour, millicalorie, calorie, kilocalorie, megacalorie, inchPound, footPound                                                                                                                                                                   | joule                        |
| FlowRate          |  cubicMetrePerSecond, cubicFeetPerSecond, cubicMetrePerYear, cubicFeetPerYear                                                                                                                                                                                                                                                                     | cubicMetrePerSecond           |
| Force             |  newton, nanonewton, millinewton, kilonewton, meganewton, giganewton, tonneForce, megapond, kilogramForce, kilopond, gramForce, pond, milligramForce, millipond, poundal                                                                                                                                                                          | newton                        |
| Frequency         |  hertz, millihertz, kilohertz, megahertz, gigahertz, terahertz                                                                                                                                                                                                                                                                                    | hertz                         |
| FuelEfficiency    |  kilometresPerLitre                                                                                                                                                                                                                                                                                                                               | kilometresPerLitre            |
| InformationUnit   |  byte, bit, kilobit, megabit, kilobyte, megabyte, gigabyte, terabyte                                                                                                                                                                                                                                                                              | byte                          |
| Distance            |  metre, nanometre, micrometre, millimetre, centimetre, decimetre, decametre, hectometre, kilometre, megametre, gigametre, inch, hand, foot, yard, fathom, rod, chain, furlong, mile, nautialMile, astronomicalUnit, lightYear, kilolightYear                                                                                                      | metre                         |
| LinearMassDensity |  gramPerKilometre                                                                                                                                                                                                                                                                                                                                 | gramPerKilometre              |
| Mass              |  gram, milligram, kilogram, tonne, stone, pound, ounce, grain, carat                                                                                                                                                                                                                                                                              | gram                          |
| PopulationDensity |  inhabitantsPerSquareKilometre, inhabitantsPerSquareMile                                                                                                                                                                                                                                                                                          | inhabitantsPerSquareKilometre |
| Power             |  watt, kilowatt, milliwatt, megawatt, gigawatt, horsepower, pferdestaerke, brakeHorsepower                                                                                                                                                                                                                                                        | watt                          |
| Pressure          |  pascal, millipascal, hectopascal, kilopascal, megapascal, millibar, decibar, bar, standardAtmosphere, poundPerSquareInch                                                                                                                                                                                                                         | pascal                        |
| Speed             |  kilometrePerHour, metrePerSecond, kilometrePerSecond, milePerHour, footPerSecond, footPerMinute, knot                                                                                                                                                                                                                                            | kilometrePerHour              |
| Time              |  second, millisecond, microsecond, nanosecond, minute, hour, day                                                                                                                                                                                                                                                                                  | second                        |
| Torque            |  newtonMetre, newtonMillimetre, newtonCentimetre, poundFoot                                                                                                                                                                                                                                                                                       | newtonMetre                   |
| Voltage           |  volt, megavolt, kilovolt, millivolt, microvolt                                                                                                                                                                                                                                                                                                   | volt                          |
| Volume            |  cubicMetre, cubicMillimetre, cubicCentimetre, cubicDecimetre, cubicDecametre, cubicHectometre, cubicKilometre, microlitre, millilitre, centilitre, decilitre, litre, hectolitre, kilolitre, megalitre, gigalitre, cubicMile, cubicYard, cubicFoot, cubicInch, imperialBarrel, usBarrel, imperialBarrelOil, usBarrelOil, imperialGallon, usGallon | cubicMetre                    |


# Examples for Data Normalization

Two code examples on how to use the [ValueNormalizer](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/preprocessing/datatypes/ValueNormalizer.java) can be found in the [Country use-case](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/countries/) folder. The examples show how data normalization is apploed to a country data set during the loading of the data set.

1. The [user-defined data model approach](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/countries/DataNormalization_CountryModel.java) provides the user with an additional layer of abstraction.
2. The [default data model approach with data type detection](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/countries/DataNormalization_DefaultModel.java) denotes a light-weighted implementation suitable for rapid prototyping including a user-specific type detector.


In all examples, the following input from [Countries.csv](https://github.com/olehmberg/winter/blob/master/winter-usecases/usecase/country/input/countries.csv) is normalised:

| Index | Name                        | Population     | Area            | Speed Limit | Date Latest Constitution |
|-------|-----------------------------|----------------|-----------------|-------------|--------------------------|
| 1     | Federal Republic of Germany | 82.6 million   | 357,386 km2     |             | 23.05.1949               |
| 2     | French Republic             | 67,186,638     | 640,679 km2     | 130 km/h    | 04-Oct-1958              |
| 3     | United States of America    | 325,719,178    | 3,796,742 sq mi | 85 mph      | June 21 1788             |
| 4     | People's Republic of China  | 1.4 billion    | 9,596,961 km2   | 120 km/h    | 04-Dec-1982              |
| 5     | Russian Federation          | 144.52 million | 17,098,246 km2  | 130 km/h    | 12-Dec-1993   

## User-defined Data Model

In this scenario, a user-defined data model is used. Thus, the records in the dataset represent objects of the user-defined `Country` class, instead of the default `Record` class.
If the user defines her own data model, it is assumed that the attribute mapping as well as data type and unit of each column are already known. 
Therefore, attribute mapping, data type and unit information can be encapsulated within a dedicated [CSVCountryReader](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/countries/model/CSVCountryReader.java), such that the data can be normalized while loading it from the csv-file:

```java	
// load data
DataSet<Country, Attribute> dataCountry = new HashedDataSet<>();
new CSVCountryReader().loadFromCSV(new File("usecase/country/input/countries.csv"), dataCountry);
```

The normalisation is implemented in the `readLine()` method of the `CSVCountryReader`, which receives the values of a line in the CSV file and creates a new Country object.
If a column's value should be normalized, the [ValueNormalizer](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/preprocessing/datatypes/ValueNormalizer.java) class is used as explained above:

```java	
@Override
protected void readLine(File file, int rowNumber, String[] values, DataSet<Country, Attribute> dataset) {

	if (rowNumber == 0) {
		initialiseDataset(dataset);
	} else {
		// generate new record of type country
		Country r = new Country(values[0], file.getAbsolutePath());

		// set value without conversion (keep the original string)
		r.setName(values[1]);
			
		...
			
		// convert data type and normalise value
		Object area = valueNormalizer.normalize(values[3], DataType.numeric,
                                          UnitCategoryParser.getUnitCategory("Area"));
		if (area != null) {
			r.setArea((Double) area);
		}
			
		...
		dataset.add(r);
	}

}
```

This setup results in a dataset with normalised formats, units and quantities for the `Population`, `Area`, `SpeedLimit`, and `DateLatestConstitution` columns:

| Id | Name                        | Population  | Area        | SpeedLimit | DateLatestConstitution |
|----|-----------------------------|-------------|-------------|------------|------------------------|
| 1  | Federal Republic of Germany | 8.26E7      | 3.57386E11   |            | 1949-05-23T00:00       |
| 2  | French Republic             | 6.7186E7    | 6.40679E11    | 130.0       | 1958-10-04T00:00       |
| 3  | United States of America    | 3.2571E8 | 9.833516E12   | 136.7939    | 1788-06-21T00:00       |
| 4  | People's Republic of China  | 1.4E9    | 9.596961E12 | 120.0       | 1982-12-04T00:00       |
| 5  | Russian Federation          | 1.4452E8  | 1.7098246E13 | 130.0       | 1993-12-12T00:00       |

## Default Data Model with Type Detection

In both scenarios using the default model, the data first has to be loaded into a dataset.
Please check the [Data Model page in this wiki for a detailed explanation on how to load a dataset into WInte.r utilizing the default data model](https://github.com/olehmberg/winter/wiki/Data-Model#loading-a-dataset).


```java	
// load data
DataSet<Record, Attribute> dataCountry = new HashedDataSet<>();
new CSVRecordReader(0).loadFromCSV(new File("usecase/country/input/countries.csv"), dataCountry);
```

Normalisation for a complete data set instead of single values can be performed with the [DataSetNormalizer](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/model/defaultmodel/preprocessing/DataSetNormalizer.java) class. Additionally, WInte.r provides the [PatternbasedTypeDetector](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/webtables/detectors/PatternbasedTypeDetector.java), which can detect data types and units of measurement for a dataset using a large range of regular expressions.
To trigger the data set normalization, pass the data set and the [PatternbasedTypeDetector](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/webtables/detectors/PatternbasedTypeDetector.java) as parameters to the `normaliseDataset(...)` method:

```java	
// normalize dataset
new DataSetNormalizer<Record>().normalizeDataset(dataCountry, new PatternbasedTypeDetector());
```

This setup results in a dataset with normalised formats, units and quantities for the `Population`, `Area`, `SpeedLimit`, and `DateLatestConstitution` columns. Nevertheless, this is a default model, hence the columns headers are not detected. Please refer to the [wiki page on Data Models](https://github.com/olehmberg/winter/wiki/Data-Model) for further details:

| countries.csv_Col0 | countries.csv_Col1                        | countries.csv_Col2  | countries.csv_Col3        | countries.csv_Col4 | countries.csv_Col5 |
|----|-----------------------------|-------------|-------------|------------|------------------------|
| 1  | Federal Republic of Germany | 8.26E7      | 3.57386E11   |            | 1949-05-23T00:00       |
| 2  | French Republic             | 6.7186E7    | 6.40679E11    | 130.0       | 1958-10-04T00:00       |
| 3  | United States of America    | 3.2571E8 | 9.833516E12   | 136.7939    | 1788-06-21T00:00       |
| 4  | People's Republic of China  | 1.4E9    | 9.596961E12 | 120.0       | 1982-12-04T00:00       |
| 5  | Russian Federation          | 1.4452E8  | 1.7098246E13 | 130.0       | 1993-12-12T00:00       |
