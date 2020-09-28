package amclassificacao;

import weka.classifiers.trees.J48;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;

public class ClassificationClient {

	public static void main(String[] args) throws Exception {

		AbstractClassifier[] algorit = new AbstractClassifier[2];
		algorit[0] = new J48();
		algorit[1] = new NaiveBayes();

		for (int k = 0; k < 2; k++) {
			Classification cl = new Classification(algorit[k]);
			
			cl.loadDataset("dados/dataset.arff");
			System.out.println("--- CROSSVALIDATION | 10 folds---");
			cl.generateModel(true);
			cl.crossValidate();
			
			System.out.println("--- Com Seleção de Atributos ---");
			cl.selectFeatures();
			cl.selectFeaturesWithClassifiers();

			for (int i = 0; i < 3; i++) {

				String path1 = "dados/train" + i + ".arff";
				String path2 = "dados/test" + i + ".arff";

				cl.loadDatasets(path1, path2);
				cl.generateModel(false);

				System.out.println("--- Avaliação ---");
				cl.simpleEvaluation();

			}

		}
	}
}