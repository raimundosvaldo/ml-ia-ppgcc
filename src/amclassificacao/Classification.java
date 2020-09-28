package amclassificacao;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	AbstractClassifier classifier;
	Instances dataset, train, test;

	public Classification(AbstractClassifier classifier) {
		this.classifier = classifier;
	}

	public void loadDataset(String path) throws Exception {
		DataSource source = new DataSource(path);
		this.dataset = source.getDataSet();
		if (dataset.classIndex() == -1)
			dataset.setClassIndex(dataset.numAttributes() - 1);
	}

	public void loadDatasets(String training, String testing) {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(training));
			this.train = new Instances(reader);
			this.train.setClassIndex(train.numAttributes() - 1);

			reader = new BufferedReader(new FileReader(testing));
			this.test = new Instances(reader);
			this.test.setClassIndex(train.numAttributes() - 1);

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void generateModel(boolean crossvalidation) throws Exception {
		if (!crossvalidation)
			this.classifier.buildClassifier(this.train);
		else
			this.classifier.buildClassifier(this.dataset);
	}

	public void crossValidate() {
		Evaluation eval = null;
		try {
			eval = new Evaluation(this.dataset);
			eval.crossValidateModel(this.classifier, this.dataset, 10, new Random(1));
			System.out.println(eval.toSummaryString());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	public void simpleEvaluation() throws Exception {

		Evaluation eval = new Evaluation(this.train);
		eval.evaluateModel(this.classifier, this.test);

		System.out.println("Corretos % = " + eval.pctCorrect());
		System.out.println("Incorretos % = " + eval.pctIncorrect());
		System.out.println("Precisão = " + eval.precision(1));
		System.out.println("Recall = " + eval.recall(1));
		System.out.println("fMeasure = " + eval.fMeasure(1));
		System.out.println(eval.toMatrixString("Matriz de Confusão"));

	}

	public void selectFeaturesWithClassifiers() {
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		classifier.setClassifier(this.classifier);
		classifier.setEvaluator(eval);
		classifier.setSearch(search);
		Evaluation evaluation;
		try {
			evaluation = new Evaluation(this.dataset);
			evaluation.crossValidateModel(classifier, this.dataset, 10, new Random(1));
			System.out.println(evaluation.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void selectFeatures() {
		AttributeSelection attSelection = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		attSelection.setEvaluator(eval);
		attSelection.setSearch(search);
		try {
			attSelection.SelectAttributes(this.train);
			int[] attIndex = attSelection.selectedAttributes();
			System.out.println(Utils.arrayToString(attIndex));
		} catch (Exception e) {
		}
	}

}