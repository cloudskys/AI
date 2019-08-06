package com.ai;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class AiJava {
	public static void main(String[] args) {
		//构建神经网络
		BasicNetwork basicNetWrok = new BasicNetwork();
		//相当于2个输入神经元
		basicNetWrok.addLayer(new BasicLayer(null, true, 2));
		//隐藏层   有2个神经元 激活函数为Sigmoid 有偏执神经元
		basicNetWrok.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
		//输出层   有1个输出神经元 激活函数为Sigmoid 无偏执神经元
		basicNetWrok.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		//构建完成
		basicNetWrok.getStructure().finalizeStructure();
		//初始化参数
		basicNetWrok.reset();
		//输入
		double[][] input = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
		//期望的输出
		double[][] ideal = { { 0 }, { 1 }, { 1 }, { 0 } };
		//构建MLDataSet对象训练集
		MLDataSet trainingset = new BasicMLDataSet(input, ideal);
		//构建弹性传播训练
		MLTrain mltrain = new ResilientPropagation(basicNetWrok, trainingset);
		//训练神经网络
		//迭代次数
		int epoch = 1;
		do {
			mltrain.iteration();
			System.out.println("迭代次数:" + epoch + ",错误率:" + mltrain.getError());
			epoch++;
		} while (mltrain.getError() > 0.01);
		//测试神经网络
		MLData output1 = basicNetWrok.compute(new BasicMLData(new double[] { 0, 0 }));
		System.out.println("0,0计算的结果为:" + output1.getData()[0]);
		MLData output2 = basicNetWrok.compute(new BasicMLData(new double[] { 1, 0 }));
		System.out.println("1,0计算的结果为:" + output2.getData()[0]);
		MLData output3 = basicNetWrok.compute(new BasicMLData(new double[] { 0, 1 }));
		System.out.println("0,1计算的结果为:" + output3.getData()[0]);
		MLData output4 = basicNetWrok.compute(new BasicMLData(new double[] { 1, 1 }));
		System.out.println("1,1计算的结果为:" + output4.getData()[0]);

	}
}
