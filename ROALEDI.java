package moa.classifiers.active;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.DoubleVector;
import moa.core.Measurement; // should be added due to unimplemented method?
import moa.core.Utils;
import moa.options.ClassOption;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

public class ROALEDI extends AbstractClassifier {
    public class ClassInfo {
        public double[] initsampleNum;
        public Vector<Instance> newlabelingSamples;
        public double imbRatio;
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.",
            Classifier.class, "trees.HoeffdingTree -e 2000000 -g 100 -c 0.01");

    public IntOption memberCountOption = new IntOption("memberCount", 'n',
            "Maximum number of classifiers in the ensemble.", 10, 1, Integer.MAX_VALUE);

    public FloatOption fixedThresholdOption = new FloatOption("fixedThreshold",
            'u', "Fixed threshold.",
            0.5, 0.00, 0.5);

    public FloatOption uncertaintyThresholdOption = new FloatOption("uncertaintyThreshold", 'u',
            "Threshold for training on uncertain instances.", 0.3, 0.0, 1.0);

    public FloatOption StepOption = new FloatOption("StepOption",
            's', "Threshold adjustment step.",
            0.01, 0.00, 1.00);

    public FloatOption StableWeight = new FloatOption("StableWeight",
            'w', "Weight of stable classifier.",
            0.5, 0.00, 1.00);

    public FloatOption PerofRandOption = new FloatOption("PerofRandOption",
            'p', "Percentenge of random strategy labled .",
            0.01, 0.00, 1.00);

    public IntOption chunkSizeOption = new IntOption("chunkSize",
            'c', "The chunk size used for classifier creation and evaluation.", 500, 1, Integer.MAX_VALUE);

    protected DoubleVector observedClassDistribution;
    protected double newThreshold;
    protected double imbalancethreshold;
    protected double[] ensembleWeights;
    protected int numberOfClasses;
    protected int currentBaseLearnerNo = -1;
    protected int selectionSize; // every time we add selectionSize to train
    protected int classLowerlimit;
    protected int costLabeling = 0;
    protected Classifier StableClassifier;
    protected Classifier[] ensemble;
    protected ClassInfo[] classinformation;

    public boolean[] selected;
    public int iterationControl = 0;
    public int sumBaseLearners = 0;
    public int processedInstances = 0;
    public Instance[] currentChunk;

    public ROALEDI() {
        super();
        ensemble = new Classifier[this.memberCountOption.getValue()];
    }

    public void createNewBaseLearner() {
        int labeled = 0;
        int initSelectednum = 0;
        newThreshold = (fixedThresholdOption.getValue() * 2) / (double) numberOfClasses;
        imbalancethreshold = this.PerofRandOption.getValue();
        Random rd2 = new Random();
        int[] selNumbers = new int[numberOfClasses];
        for (int i = 0; i < numberOfClasses; i++) {
            selNumbers[i] = 0;
        }
        if (sumBaseLearners < this.memberCountOption.getValue())
            currentBaseLearnerNo = sumBaseLearners;
        else
            currentBaseLearnerNo = Utils.minIndex(ensembleWeights);

        sumBaseLearners++;
        // classifier replacement
        int numEnsembledBaseLearners = Math.min(this.memberCountOption.getValue(), sumBaseLearners);

        // learn all instances in the first data block
        if (sumBaseLearners == 1)
            initSelectednum = selectionSize;
        else
            initSelectednum = selectionSize;

        for (int i = 0; i < this.chunkSizeOption.getValue(); i++) {
            selected[i] = false;
        }
        double weightfactor = 1;

        Vector<Instance> allinitRandSamples = new Vector<Instance>();
        // Stage1
        while (labeled < initSelectednum) {
            int no = (int) (rd2.nextFloat() * this.chunkSizeOption.getValue());
            if (sumBaseLearners == 1)
                no = labeled;
            if (!selected[no]) {
                int classlable = (int) currentChunk[no].classValue();
                selNumbers[classlable]++;
                allinitRandSamples.addElement(currentChunk[no]);
                if (correctlyClassifies(currentChunk[no]))
                    newThreshold = newThreshold * ((double) 1 - StepOption.getValue());
                else if (classinformation[classlable].imbRatio < 1 / (double) (numberOfClasses))
                    for (int j = 0; j < numEnsembledBaseLearners; j++)
                        if (!ensemble[j].correctlyClassifies(currentChunk[no]))
                            ensembleWeights[j] *= (1 - 1 / (double) (1 + numEnsembledBaseLearners));
                        else
                            ensembleWeights[j] *= (1 + 1 / (double) (1 + numEnsembledBaseLearners));
                selected[no] = true;
                labeled++;
            }
        }

        // Stage2
        ensemble[currentBaseLearnerNo].resetLearning();
        for (int i = 0; i < numberOfClasses; i++) {
            classinformation[i].imbRatio = 0;
            classinformation[i].initsampleNum[currentBaseLearnerNo] = selNumbers[i];
            if (selNumbers[i] < this.classLowerlimit) {
                int numNeedmore = this.classLowerlimit - selNumbers[i];
                int numSavesample = classinformation[i].newlabelingSamples.size();
                int minNum = Math.min(numNeedmore, numSavesample);

                for (int j = minNum; j > 0; j--) {
                    Instance sample = classinformation[i].newlabelingSamples.get(numSavesample - j);
                    ensemble[currentBaseLearnerNo].trainOnInstance(sample);
                }
            }
        }

        for (Instance inst : allinitRandSamples) {
            int classlable = (int) inst.classValue();
            classinformation[classlable].newlabelingSamples.addElement(inst);
            StableClassifier.trainOnInstance(inst);
            for (int j = 0; j < numEnsembledBaseLearners; j++)
                ensemble[j].trainOnInstance(inst);
        }

        double sumWeight = 0;
        ensembleWeights[currentBaseLearnerNo] = 0;

        for (int i = 0; i < numEnsembledBaseLearners; i++) {
            ensembleWeights[i] = ensembleWeights[i] * (1 - 1 / (double) numEnsembledBaseLearners);
            sumWeight += ensembleWeights[i];
        }

        ensembleWeights[currentBaseLearnerNo] = weightfactor * 1 / (double) (numEnsembledBaseLearners);
        sumWeight += ensembleWeights[currentBaseLearnerNo];

        double sumweightNum = 0;
        for (int i = 0; i < numEnsembledBaseLearners; i++) {
            ensembleWeights[i] = ensembleWeights[i] / sumWeight;
            for (int j = 0; j < numberOfClasses; j++) {
                double dNum = classinformation[j].initsampleNum[i] * ensembleWeights[i];
                classinformation[j].imbRatio = classinformation[j].imbRatio + dNum;
                sumweightNum = sumweightNum + dNum;
            }
        }
        costLabeling += labeled;
        for (int i = 0; i < numberOfClasses; i++) {
            int leng = classinformation[i].newlabelingSamples.size() - classLowerlimit;
            for (int j = 0; j < leng; j++)
                classinformation[i].newlabelingSamples.remove(0);
            if (sumweightNum > 0)
                classinformation[i].imbRatio = (classinformation[i].imbRatio / sumweightNum);
            else
                classinformation[i].imbRatio = 1;
        }

    }

    public boolean UncertaintyStrategy(Instance instance) {
        double[] count = getVotesForInstance(instance);
        int maxIndex = Utils.maxIndex(count);
        double maxDistr = count[maxIndex];
        count[maxIndex] = 0;
        int secondMaxIndex = Utils.maxIndex(count);
        double margin = maxDistr - count[secondMaxIndex];
        if (margin <= newThreshold)
            return true;
        else
            return false;
    }

    public boolean RandomStrategy(Instance instance) {
        double[] count = getVotesForInstance(instance);
        int maxIndex = Utils.maxIndex(count);
        double tmpRanthreshold = this.imbalancethreshold;
        if (classinformation[maxIndex].imbRatio > 0)
            tmpRanthreshold = Math.max(tmpRanthreshold,
                    this.imbalancethreshold / (double) (numberOfClasses * classinformation[maxIndex].imbRatio));
        else
            tmpRanthreshold = 1;
        Random rd1 = new Random();
        if (rd1.nextDouble() < tmpRanthreshold) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    public void resetLearningImpl() {
        StableClassifier.resetLearning();
        for (int i = 0; i < ensemble.length; i++) {
            ensemble[i] = (Classifier) getPreparedClassOption(learnerOption);
            ensemble[i].resetLearning();
        }
        currentChunk = new Instance[this.chunkSizeOption.getValue()];
        currentBaseLearnerNo = -1;
        costLabeling = 0;
        sumBaseLearners = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        Instance instance = currentChunk[iterationControl]; // get instance from the buffer
        int numEnsembledBaseLearners = Math.min(this.memberCountOption.getValue(), sumBaseLearners);
        // Uncertainty strategy
        if (UncertaintyStrategy(instance)) {
            // labeled instance to train on it
            if (correctlyClassifies(instance))
                newThreshold = newThreshold * ((double) 1 - StepOption.getValue());
            StableClassifier.trainOnInstance(instance);
            for (int j = 0; j < numEnsembledBaseLearners; j++) {
                ensemble[j].trainOnInstance(instance);
            }
            costLabeling++;
            newThreshold = newThreshold * ((double) 1 - StepOption.getValue());
            classinformation[(int) instance.classValue()].newlabelingSamples.addElement(instance);
        } else {
            // Random strategy
            if (!selected[iterationControl] && RandomStrategy(instance)) {
                // labeled instance to train on it
                if (!correctlyClassifies(instance)) {
                    int classlable = (int) instance.classValue();
                    if (classinformation[classlable].imbRatio < 1 / ((double) numberOfClasses))
                        for (int j = 0; j < numEnsembledBaseLearners; j++)
                            if (!ensemble[j].correctlyClassifies(instance))
                                ensembleWeights[j] *= (1 - 1 / (1 + (double) numEnsembledBaseLearners));
                            else
                                ensembleWeights[j] *= (1 + 1 / (1 + (double) numEnsembledBaseLearners));
                }
                StableClassifier.trainOnInstance(instance);
                for (int j = 0; j < numEnsembledBaseLearners; j++)
                    ensemble[j].trainOnInstance(instance);
                costLabeling++;
                classinformation[(int) instance.classValue()].newlabelingSamples.addElement(instance);
            }
        }
        currentChunk[iterationControl] = inst;
        // new inst replaced with instance in buffer
        iterationControl = (iterationControl + 1) % this.chunkSizeOption.getValue();
        if (iterationControl == 0)
            // new instances fulfill the buffer again
            createNewBaseLearner();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {

        DoubleVector combinedVote = new DoubleVector();
        for (Classifier classifier : ensemble) {
            DoubleVector vote = new DoubleVector(classifier.getVotesForInstance(inst));
            combinedVote.addValues(vote);
        }
        if (combinedVote.sumOfValues() > 0.0) {
            combinedVote.normalize();
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    // Add labeling cost and new threshold
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<Measurement>();
        double labelingcost = 0;
        if (processedInstances > 0) {
            labelingcost = 1.0 * this.costLabeling / (double) processedInstances;
        }
        measurementList.add(new Measurement("labeling cost", labelingcost));
        measurementList.add(new Measurement("labels number", this.costLabeling));
        measurementList.add(new Measurement("newThreshold", this.newThreshold));
        measurementList.add(new Measurement("limit", this.classLowerlimit));
        measurementList.add(new Measurement("weight", this.ensembleWeights[0]));
        for (int j = 0; j < numberOfClasses; j++) {
            String s = String.valueOf(j);
            measurementList.add(new Measurement("imbRatio_" + s, this.classinformation[j].imbRatio));
        }

        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
}
