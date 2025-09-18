The event usecase integrates XML data from the DBpedia and YAGO Knowledge Graphs using the following classes:
* `dbo:Event` for DBpedia and 
* `yago:wordnet_event_100029378` for YAGO

The following attributes where used to query the Knowledge Graphs:
* label: `rdf:label`
* date: `dbo:date` (DBpedia) and `yago:happenedOnDate` (YAGO)
* coordinates: `geo:lat` and `geo:long` (DBpedia) and `yago:hasLatitude` and `yago:hasLongitude` (YAGO)
* same (real-world entity) in other Knowledge Graphs: `owl:sameAs`
* location: `dbo:place` (DBpedia) and `yago:isLocatedIn` (YAGO)
Different or additional attributes can be used for creating the XML files.


Each event is required to have a URI and at least one label. Events are not required to have any location. But if they do, these locations are required to have a URI and at least one label as well. All other attribute (date, coordinates, same) can be used zero, one, or multiple times for events and locations.

An example XML file with one event having all possible attributes looks like the following:

    <events>
     <event uri="http://dbpedia.org/resource/Battle_of_Bouvines">
        <label>Battle of Bouvines@en</label>
        <date>1214-07-27^^http://www.w3.org/2001/XMLSchema#date</date>
        <coordinates>50.5833^^http://www.w3.org/2001/XMLSchema#float,3.225^^http://www.w3.org/2001/XMLSchema#float</coordinates>
        <same>http://wikidata.dbpedia.org/resource/Q830626</same>
        <same>http://yago-knowledge.org/resource/Battle_of_Bouvines</same>
        <locations>
          <location uri="http://dbpedia.org/resource/Bouvines">
            <label>Bouvines@en</label>
            <coordinates>50.583^^http://www.w3.org/2001/XMLSchema#float,3.183^^http://www.w3.org/2001/XMLSchema#float</coordinates>
            <same>http://yago-knowledge.org/resource/Bouvines</same>
            <same>http://www.wikidata.org/entity/Q1079414</same>
          </location>
        </locations>
      </event>
    </events>



The public SPARQL endpoints http://dbpedia.org/sparql and https://linkeddata1.calcul.u-psud.fr/sparql can be used to query the Knowledge Graphs. The results have to be converted to the XML structure as shown in the example above.