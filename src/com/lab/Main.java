package com.lab;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.jvm.JVMAudioInputStream;
import com.lab.processors.MFCC;
import org.apache.commons.math3.stat.correlation.Covariance;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Random;

public class Main {

    private static final int CEP_SIZE = 15;
    public static final String COMPRESSED_OUT_ARFF = "neuronalOut.arff";
    public static final String FULL_OUT_ARFF = "out.arff";
    public static final String COMPRESSED_ARFF = "compressed.arff";
    public static final String FULL_ARFF = "full.arff";

    public static void main(String[] args) throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        SMO smo = new SMO();
        mlp.setOptions(Utils.splitOptions("-L 0.4 -M 0.4 -N 500 -V 0 -S 0 -E 20 -H a"));
        smo.setOptions(Utils.splitOptions("-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));

        String[] newArgs = Arrays.copyOfRange(args, 1, args.length);

        if (args[0].equals("-train")) {
            generateTrainingArffs(newArgs, newArgs, FULL_OUT_ARFF, COMPRESSED_OUT_ARFF);
            FileReader trainReader = new FileReader(COMPRESSED_OUT_ARFF);
//            FileReader trainReader = new FileReader(FULL_OUT_ARFF);
            Instances trainingSet = new Instances(trainReader);
            trainingSet.randomize(new Random(0));

            int trainSize = (int) Math.round(trainingSet.numInstances() * 0.66);
            int testSize = trainingSet.numInstances() - trainSize;
            Instances train = new Instances(trainingSet, 0, trainSize);
            Instances test = new Instances(trainingSet, trainSize, testSize);
            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            mlp.buildClassifier(train);
            smo.buildClassifier(train);

            Evaluation mlpEval = new Evaluation(train);
            Evaluation smoEval = new Evaluation(train);
            mlpEval.evaluateModel(mlp, test);
            smoEval.evaluateModel(smo, test);

            printEvaluationResults(mlpEval, mlp.getClass().getSimpleName());
            printEvaluationResults(smoEval, smo.getClass().getSimpleName());

            SerializationHelper.write("mlp.model", mlp);
            SerializationHelper.write("smo.model", smo);
        } else {
            String[] folders = new String[1];
            folders[0] = "genres/?";


            //open the folder with data and get all files inside
            File folder = new File(folders[0]);
            File[] audioFiles = folder.listFiles();

            int wrongMLP = 0;
            int wrongSMO = 0;


//            generateTrainingArffs(folders, args, FULL_ARFF, COMPRESSED_ARFF);

            mlp = (MultilayerPerceptron) weka.core.SerializationHelper.read("mlp.model");
            smo = (SMO) SerializationHelper.read("smo.model");
            for (int k = 0; k < audioFiles.length; k++) {
                generatePredictionArffs(audioFiles[k], args, FULL_ARFF, COMPRESSED_ARFF);
//                Instances dataPredictMLP = new Instances( new BufferedReader( new FileReader(COMPRESSED_ARFF)));
                Instances dataPredictMLP = new Instances( new BufferedReader( new FileReader(FULL_ARFF)));
                dataPredictMLP.setClassIndex(dataPredictMLP.numAttributes() - 1);
                Instances predictedDataMLP = new Instances(dataPredictMLP);

//                Instances dataPredictSMO = new Instances( new BufferedReader( new FileReader(COMPRESSED_ARFF)));
                Instances dataPredictSMO = new Instances( new BufferedReader( new FileReader(FULL_ARFF)));
                dataPredictSMO.setClassIndex(dataPredictSMO.numAttributes() - 1);
                Instances predictedDataSMO = new Instances(dataPredictSMO);

                System.out.println("Multilayer Perceptron:");
                for (int i = 0; i < dataPredictMLP.numInstances(); i++) {
                    double clsLabel = mlp.classifyInstance(dataPredictMLP.instance(i));
                    predictedDataMLP.instance(i).setClassValue(clsLabel);
                    System.out.print(audioFiles[k].getName() + " ------ ");
                    System.out.print(predictedDataMLP.classAttribute().value((int) clsLabel));

                    String[] filenameSplit = audioFiles[k].getName().split("\\.");
                    if (!filenameSplit[0].equals(predictedDataMLP.classAttribute().value((int) clsLabel))){
                        wrongMLP++;
                        System.out.print(" X");
                    } else {
                        System.out.print(" O");
                    }

                    System.out.println();
                }

                System.out.println("SMO:");
                for (int i = 0; i < dataPredictSMO.numInstances(); i++) {
                    double clsLabel = smo.classifyInstance(dataPredictSMO.instance(i));
                    predictedDataSMO.instance(i).setClassValue(clsLabel);
                    System.out.print(audioFiles[k].getName() + " ------ ");
                    System.out.print(predictedDataSMO.classAttribute().value((int) clsLabel));

                    String[] filenameSplit = audioFiles[k].getName().split("\\.");
                    if (!filenameSplit[0].equals(predictedDataSMO.classAttribute().value((int) clsLabel))){
                        wrongSMO++;
                        System.out.print(" X");
                    } else {
                        System.out.print(" O");
                    }

                    System.out.println();
                }
            }

            System.out.println("MLP:");
            System.out.println("Total: " + audioFiles.length + "/Wrong: " + wrongMLP);
            System.out.println("SMO:");
            System.out.println("Total: " + audioFiles.length + "/Wrong: " + wrongSMO);


//            System.out.println(predicteddata.toString());
        }

    }

    private static void printEvaluationResults(Evaluation evaluation, String evaluatorClassName) throws Exception {
        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));


        System.out.println("training performance results of: " + evaluatorClassName
                + "\n---------------------------------");
        System.out.println(evaluation.toSummaryString("\nResults",true));
        System.out.println("fmeasure: " +evaluation.fMeasure(1) + " Precision: " + evaluation.precision(1)+ " Recall: "+ evaluation.recall(1));
        System.out.println(evaluation.toMatrixString());
        System.out.println(evaluation.toClassDetailsString());
        System.out.println("AUC = " +evaluation.areaUnderROC(1));
        System.out.println("Training complete, please validate trained model");
    }

    private static void generatePredictionArffs(File song, String[] classFolders, String fullOutput, String compressedOutput) throws IOException, UnsupportedAudioFileException {
        //write in csv file
        File fout = new File(fullOutput);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        File neuronalFout = new File(compressedOutput);
        FileOutputStream nfos = new FileOutputStream(neuronalFout);
        BufferedWriter nbw = new BufferedWriter(new OutputStreamWriter(nfos));

        bw.write("@RELATION music_recognition");
        nbw.write("@RELATION music_recognition_compressed");
        bw.newLine();
        nbw.newLine();

        for (int k = 0; k < 240; k++){
            bw.write("@ATTRIBUTE sample_" + k + " NUMERIC");
            bw.newLine();
        }

        for (int k = 0; k < 135; k++) {
            nbw.write("@ATTRIBUTE sample_" + k + " NUMERIC");
            nbw.newLine();
        }

        String[] classes = new String[classFolders.length];
        String arfClasses = "{";
        for (int k = 0; k < classFolders.length; k++) {
            classes[k] = classFolders[k].split("/")[1];
            arfClasses += classes[k] + ",";
        }

        arfClasses = arfClasses.substring(0, arfClasses.length()-1);
        arfClasses += "}";

        bw.write("@ATTRIBUTE class " + arfClasses);
        bw.newLine();
        nbw.write("@ATTRIBUTE class " + arfClasses);
        nbw.newLine();

        bw.write("@DATA");
        bw.newLine();
        nbw.write("@DATA");
        nbw.newLine();

        //prepare the matrix to store frame data
        double[][] songData = new double[638][CEP_SIZE];

        AudioInputStream is = AudioSystem.getAudioInputStream(song);

        JVMAudioInputStream jis = new JVMAudioInputStream(is);
        AudioDispatcher dispatcher = new AudioDispatcher(jis, 1024, 512);

        AudioProcessor customProcessor = new AudioProcessor() {

            MFCC customMfcc = new MFCC(1024,
                    jis.getFormat().getSampleRate(),
                    20,
                    1024,
                    false,
                    1,
                    false);
            int stop = 0;

            @Override
            public boolean process(AudioEvent audioEvent) {
                if (stop == 638) return true; //Process around 50% of the song

                float[] audioFloatBuffer = audioEvent.getFloatBuffer().clone();
                double[] newBuffer = customMfcc.getParameters(convertFloatsToDoubles(audioFloatBuffer));
                double[] relevantCepstralFeatures = new double[CEP_SIZE];

                for (int j = 0; j < CEP_SIZE; j++) {
                    relevantCepstralFeatures[j] = (float) newBuffer[j];
                }

                songData[stop] = relevantCepstralFeatures;
                stop++;

                return true;
            }

            @Override
            public void processingFinished() {

            }
        };

        dispatcher.addAudioProcessor(customProcessor);
        dispatcher.run();


        double[] meanVector = getMeanVector(songData);
        double[][] covarianceMatrix = new Covariance(songData).getCovarianceMatrix().getData();

        for (double element :
                meanVector) {
            try {

                bw.write(Double.toString(round(element, 2)) + ",");
//                        bw.write(Double.toString(element) + ",");
//                        nbw.write(Double.toString(element) + ",");
                nbw.write(Double.toString(round(element, 2)) + ",");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        for (double[] elements :
                covarianceMatrix) {
            for (double element:
                    elements) {
                try {
//                            bw.write(Double.toString(element) + ",");
                    bw.write(Double.toString(round(element, 2)) + ",");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        for (int j = 0; j < (covarianceMatrix.length / 2) + 1; j++) {
            for (int l = 0; l < covarianceMatrix[j].length; l++) {
//                        nbw.write(Double.toString(covarianceMatrix[j][l]) + ",");
                nbw.write(Double.toString(round(covarianceMatrix[j][l], 2)) + ",");
            }
        }

//                bw.write(Integer.toString(k + 1));
        bw.write("?");
//                nbw.write(Integer.toString(k + 1));
        nbw.write("?");
        bw.newLine();
        nbw.newLine();


        bw.close();
        nbw.close();
    }

    private static void generateTrainingArffs(String[] args, String[] classFolders, String fullOutput, String compressedOutput) throws IOException, UnsupportedAudioFileException {
        //write in csv file
        File fout = new File(fullOutput);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        File neuronalFout = new File(compressedOutput);
        FileOutputStream nfos = new FileOutputStream(neuronalFout);
        BufferedWriter nbw = new BufferedWriter(new OutputStreamWriter(nfos));

        bw.write("@RELATION music_recognition");
        nbw.write("@RELATION music_recognition_compressed");
        bw.newLine();
        nbw.newLine();

        for (int k = 0; k < 240; k++){
            bw.write("@ATTRIBUTE sample_" + k + " NUMERIC");
            bw.newLine();
        }

        for (int k = 0; k < 135; k++) {
            nbw.write("@ATTRIBUTE sample_" + k + " NUMERIC");
            nbw.newLine();
        }

        String[] classes = new String[classFolders.length];
        String arfClasses = "{";
        for (int k = 0; k < classFolders.length; k++) {
            classes[k] = classFolders[k].split("/")[1];
            arfClasses += classes[k] + ",";
        }

        arfClasses = arfClasses.substring(0, arfClasses.length()-1);
        arfClasses += "}";

        bw.write("@ATTRIBUTE class " + arfClasses);
        bw.newLine();
        nbw.write("@ATTRIBUTE class " + arfClasses);
        nbw.newLine();

        bw.write("@DATA");
        bw.newLine();
        nbw.write("@DATA");
        nbw.newLine();

        for (int k = 0; k < args.length; k++) {

            //open the folder with data and get all files inside
            File folder = new File(args[k]);
            File[] audioFiles = folder.listFiles();

            for (int i = 0; i < audioFiles.length; i++) {

                //prepare the matrix to store frame data
                double[][] songData = new double[638][CEP_SIZE];

                assert audioFiles != null;
                AudioInputStream is = AudioSystem.getAudioInputStream(audioFiles[i]);

                JVMAudioInputStream jis = new JVMAudioInputStream(is);
                AudioDispatcher dispatcher = new AudioDispatcher(jis, 1024, 512);

                AudioProcessor customProcessor = new AudioProcessor() {

                    MFCC customMfcc = new MFCC(1024,
                            jis.getFormat().getSampleRate(),
                            20,
                            1024,
                            false,
                            1,
                            false);
                    int stop = 0;

                    @Override
                    public boolean process(AudioEvent audioEvent) {
                        if (stop == 638) return true; //Process around 50% of the song

                        float[] audioFloatBuffer = audioEvent.getFloatBuffer().clone();
                        double[] newBuffer = customMfcc.getParameters(convertFloatsToDoubles(audioFloatBuffer));
                        double[] relevantCepstralFeatures = new double[CEP_SIZE];

                        for (int j = 0; j < CEP_SIZE; j++) {
                            relevantCepstralFeatures[j] = (float) newBuffer[j];
                        }

                        songData[stop] = relevantCepstralFeatures;
                        stop++;

                        return true;
                    }

                    @Override
                    public void processingFinished() {

                    }
                };

                dispatcher.addAudioProcessor(customProcessor);
                dispatcher.run();


                double[] meanVector = getMeanVector(songData);
                double[][] covarianceMatrix = new Covariance(songData).getCovarianceMatrix().getData();

                for (double element :
                        meanVector) {
                    try {

                        bw.write(Double.toString(round(element, 2)) + ",");
//                        bw.write(Double.toString(element) + ",");
//                        nbw.write(Double.toString(element) + ",");
                        nbw.write(Double.toString(round(element, 2)) + ",");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                for (double[] elements :
                        covarianceMatrix) {
                    for (double element:
                            elements) {
                        try {
//                            bw.write(Double.toString(element) + ",");
                            bw.write(Double.toString(round(element, 2)) + ",");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }

                for (int j = 0; j < (covarianceMatrix.length / 2) + 1; j++) {
                    for (int l = 0; l < covarianceMatrix[j].length; l++) {
//                        nbw.write(Double.toString(covarianceMatrix[j][l]) + ",");
                        nbw.write(Double.toString(round(covarianceMatrix[j][l], 2)) + ",");
                    }
                }

//                bw.write(Integer.toString(k + 1));
                bw.write(classes[k]);
//                nbw.write(Integer.toString(k + 1));
                nbw.write(classes[k]);
                bw.newLine();
                nbw.newLine();
            }
        }

        bw.close();
        nbw.close();
    }

    private static double[] convertFloatsToDoubles(float[] input)
    {
        if (input == null)
        {
            return null; // Or throw an exception - your choice
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i];
        }
        return output;
    }

    private static double[] getMeanVector(double[][] matrix){
        double[] result = new double[matrix[0].length];

        for (int i = 0; i < matrix[0].length; i++) {
            double sum = 0;

            for (double[] aMatrix : matrix) {
                sum += aMatrix[i];
            }

            result[i] = sum/matrix.length;
        }

        return result;
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}


