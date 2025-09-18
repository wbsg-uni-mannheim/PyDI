Welcome to the WInte.r Wiki. The pages in this Wiki introduce the most important concepts and parts of the framework. In addition, the [WInte.r Tutorial](WInte.r-Tutorial) provides a set-by-step explanation of the framework along a complete data integration project..

The WInte.r framework comprises a data model and methods for various data integration tasks. The general workflow of an end-to-end data integration process looks like the following.
1.	Load the data into dataset objects
2.	Apply schema matching to get correspondences between attributes
3.	Apply identity resolution to get correspondences between records
4.	Transform the data into a consolidated schema using the correspondences between the attributes
5.	Use the correspondences between the records to perform data fusion and create one consolidated dataset

Contents
- [Data Model](DataModel)
  - [Data Normalisation](DataNormalisation)
  - [Web Tables](WebTables)
- [Matching](Matching)
  - [Similarity Measures](SimilarityMeasures)
  - [Blocking](Blocking)
  - [Schema Matching](SchemaMatching)
  - [Duplicate Detection](DuplicateDetection)
  - [Identity Resolution](IdentityResolution)
  - [Learning Matching Rules](Learning-Matching-Rules)
    - [RapidMiner Integration](RapidMiner-Integration)
- [Data Fusion](DataFusion)
- [Evaluation](Evaluation)
- [Event and Result Logging](Event-and-Result-Logging)
- [Tutorial: Movie Data Integration](WInte.r-Tutorial)
