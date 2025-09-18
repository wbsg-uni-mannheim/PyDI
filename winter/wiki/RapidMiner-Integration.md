WInte.r provides capabilities to learn matching rules using [WEKA](http://www.cs.waikato.ac.nz/ml/weka/index.html) as described in the section [Learning Matching Rules](https://github.com/olehmberg/winter/wiki/Learning-Matching-Rules).
Additionally, numerous tools like [RapidMiner](https://rapidminer.com/) exist to learn supervised machine learning models that allow a user to learn classifiers, which can be use for data matching. In order to give the user the tools of his or her choice, WInte.r allows the user to generate and export training data to learn these matching rule models outside of WInte.r. This documentation shows how to use [RapidMiner](https://rapidminer.com/) to learn matching rule models and import them using the [WEKA](http://www.cs.waikato.ac.nz/ml/weka/index.html) or [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) format back into WInte.r.

An example [identity resolution process using RapidMiner](https://github.com/olehmberg/winter/blob/master/winter-usecases/src/main/java/de/uni_mannheim/informatik/dws/winter/usecase/movies/Movies_IdentityResolutionRapidminerMatchingRule.java) can be found in the movies usecase. Additionally, a corresponding [RapidMiner repository containing sample processes is provided](https://github.com/olehmberg/winter/tree/rapidminerintegration/winter-usecases/usecase/movie/Rapidminer), so that a user can easily run the whole process locally by adding the repository to the user's own [RapidMiner](https://rapidminer.com/) repositories.

# Prerequisites

- [RapidMiner PMML Extension](https://marketplace.rapidminer.com/UpdateServer/faces/product_details.xhtml?productId=rmx_pmml)

# Export Training Data from WInte.r

Each matching rule has an interface to export the training data, which is used to train a corresponding matching rule model. With this data, a matching rule model can be trained externally on the same training data as a matching rule model trained in WInte.r. 

Inputs to the export method are the data sets which are being integrated, a goldstandard containing matches for those two data sets, and the designated file for the training data.
Calling this method internally triggers the feature generation, which executes the matching rule's comparators to calculate similarities. Hence, comparators must be added to the matching rule using the export.
Thus, each line of the training data csv file contains the calculated similarity values for each comparator, which was added to the matching rule before.
As a matching rule may contain multiple comparators, the header of the comparator's values is pre-fixed with the comparator's position and the comparator's name. For example, the MovieDirectoryComparatorLevenshtein, which is the first comparator of the matching rule, has the header  '[0] MovieTitleComparatorEqual'. Additionally, each line has a column 'label', which describes whether the line describes a match. This information on the label is inferred from the provided goldstandard. Hence, matching pairs from the goldstandard result in a "1" whereas non-matching pairs result in a "0".

| [0] MovieTitleComparatorEqual | [1] MovieDateComparator2Years | ... | label |
| ------------- | ------------- |------------- | ------------- | 
| 1  |  0.5 | ... | 1  |
| 0  | 0.5  | ... | 1  | 
| ...  | ...  | ...| ...  |


```java
// Export Training Data
matchingRule.exportTrainingData(dataAcademyAwards, dataActors, gsTest, new File
               ("usecase/movie/output/optimisation/academy_awards_2_actors_features.csv"));
```

# Train Model in RapidMiner

The screenshot below shows an example of a [Rapidminer process](https://github.com/olehmberg/winter/tree/rapidminerintegration/winter-usecases/usecase/movie/Rapidminer/processes), which trains a matching rule model using a [decision tree](https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/trees/parallel_decision_tree.html) learner. It contains three parts. First, the csv file containing the training data is loaded, afterwards the decision tree model is learned and finally the trained model is exported in the [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) format.

![Rapidminer Integration](img/rapidminer_process_decison_tree.png)

## Import Training Data into RapidMiner

When importing the training data into RapidMiner a couple of rules have to be followed to ensure compatibility of the learned model with WInte.r.

When using the import wizard of the [Read CSV](https://docs.rapidminer.com/latest/studio/operators/data_access/files/read/read_csv.html) operator, the user selects the correct file as used for the training data export in WInte.r: *</your path to winte.r>/usecase/movie/Rapidminer/data/optimisation/academy_awards_2_actors_features.csv*

During data formatting, do *not* rename or delete any column to keep the data schema as provided by WInte.r. Change the type of the column 'label' to *binominal* and set the role of this column to *label*. Afterwards, the data import can be finished by clicking the Finish button.

![Data Formatting](img/read_training_data_formatting2.png)

## Training and exporting a model via PMML in RapidMiner

WInte.r supports the model formats of [WEKA](http://www.cs.waikato.ac.nz/ml/weka/index.html) and [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) to import matching models. Thus, any model which is exported from [RapidMiner](https://rapidminer.com/), has to be in either of these formats. As the [RapidMiner WEKA extension](https://marketplace.rapidminer.com/UpdateServer/faces/product_details.xhtml?productId=rmx_weka) contains mainly operators which are already available in WInte.r as described in [Learning Matching Rules](https://github.com/olehmberg/winter/wiki/Learning-Matching-Rules), this documentation focuses on [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) models.

To export [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) models from [RapidMiner](https://rapidminer.com/), the [RapidMiner PMML extension](https://marketplace.rapidminer.com/UpdateServer/faces/product_details.xhtml?productId=rmx_pmml) has to be installed. This extension provides the [Write PMML](https://docs.rapidminer.com/latest/studio/operators/data_access/files/write/write_pmml.html) operator to export machine learning models in the [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html) format.

The list of models, which can be exported by the [Write PMML](https://docs.rapidminer.com/latest/studio/operators/data_access/files/write/write_pmml.html) operator, is limited and can be found in the operator's [description](https://docs.rapidminer.com/latest/studio/operators/data_access/files/write/write_pmml.html). We tested the [decision tree](https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/trees/parallel_decision_tree.html) and the [linear regression](https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/functions/linear_regression.html). Both accept the training data provided by the [Read CSV](https://docs.rapidminer.com/latest/studio/operators/data_access/files/read/read_csv.html) operator as described above.

The trained model has to be passed to the [Write PMML](https://docs.rapidminer.com/latest/studio/operators/data_access/files/write/write_pmml.html) operator. Apart from the model, the user has to specify a location for the model:
*</your path to winte.r>/usecase/movie/Rapidminer/models/matchingModel.pmml*.


Afterwards, the RapidMiner process can be executed to train and export a matching rule model.

# Import Learned model into WInte.r

The trained matching rule model can be loaded via the readModel(File file) method of the [WekaMatchingRule](https://github.com/olehmberg/winter/blob/master/winter-framework/src/main/java/de/uni_mannheim/informatik/dws/winter/matching/rules/WekaMatchingRule.java).

```java
// import matching rule model
matchingRule.readModel(new File("usecase/movie/input/model/matchingRule/matchingModel.pmml"));
```

The trained matching model can be applied to any matching task, that follows the same schema as defined for the training data. 