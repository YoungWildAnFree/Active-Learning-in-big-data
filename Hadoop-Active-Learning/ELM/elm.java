package elm;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;

public class elm {
	private DenseMatrix train_set;   //训锟斤拷锟斤拷锟捷撅拷锟斤拷
	private DenseMatrix test_set;	 //锟斤拷锟斤拷锟斤拷锟捷撅拷锟斤拷
	private int numTrainData;		//训锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
	private int numTestData;		//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
	private DenseMatrix InputWeight;	//锟斤拷锟斤拷权锟截撅拷锟斤拷锟斤拷锟斤拷锟斤拷桑锟�
	private float TrainingTime;			//训锟斤拷时锟斤拷
	private float TestingTime;			//锟斤拷锟斤拷时锟斤拷
	private double TrainingAccuracy, TestingAccuracy; //训锟斤拷锟酵诧拷锟斤拷准确锟斤拷
	private int Elm_Type;		//锟斤拷锟斤拷锟斤拷锟酵ｏ拷0-锟截归，1-锟斤拷锟洁）
	private int NumberofHiddenNeurons;		//锟斤拷锟截节碉拷锟斤拷锟斤拷锟皆硷拷锟斤拷锟矫ｏ拷
	private int NumberofOutputNeurons;						//锟斤拷锟斤拷锟节碉拷锟斤拷锟斤拷也锟斤拷锟斤拷锟斤拷锟�
	private int NumberofInputNeurons;						//锟斤拷锟斤拷锟节碉拷锟斤拷锟斤拷也锟斤拷锟斤拷锟斤拷锟斤拷
	private String func;	//锟斤拷锟筋函锟斤拷
	private int []label;		//锟斤拷锟斤拷锟较拷锟斤拷锟斤拷锟斤拷锟�
	//this class label employ a lazy and easy method,any class must written in 0,1,2...so the preprocessing is required
	//the blow variables in both train() and test()
	private DenseMatrix  BiasofHiddenNeurons;		//偏锟斤拷锟斤拷锟�
	private DenseMatrix  OutputWeight;				//锟斤拷锟斤拷锟斤拷锟窖碉拷锟斤拷锟斤拷锟斤拷锟斤拷锟�
	private DenseMatrix  testP;				//锟斤拷锟皆撅拷锟斤拷
	private DenseMatrix  testT;				//锟斤拷锟斤拷锟斤拷
	private DenseMatrix  Y;				//预锟斤拷值锟斤拷锟斤拷
	private DenseMatrix  T;				//目锟斤拷值锟斤拷锟斤拷
	
	private DenseMatrix OutMatrix;
	
	private String splitmark=",";
	/**
     * Construct an ELM
     * @param
     * elm_type              - 0 for regression; 1 for (both binary and multi-classes) classification
     * @param
     * numberofHiddenNeurons - Number of hidden neurons assigned to the ELM
     * @param
     * ActivationFunction    - Type of activation function:
     *                      'sig' for Sigmoidal function
     *                      'sin' for Sine function
     *                      'hardlim' for Hardlim function
     *                      'tribas' for Triangular basis function
     *                      'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
     * @throws NotConvergedException
     */

	//锟斤拷锟届函锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟酵ｏ拷0-锟截归，1-锟斤拷锟洁，锟斤拷锟截节碉拷锟斤拷锟斤拷锟斤拷锟筋函锟斤拷锟斤拷
	public elm(int elm_type, int numberofHiddenNeurons, String ActivationFunction){
		
		
		
		Elm_Type = elm_type;
		NumberofHiddenNeurons = numberofHiddenNeurons;
		func = ActivationFunction;
		
		TrainingTime = 0;
		TestingTime = 0;
		TrainingAccuracy= 0;
		TestingAccuracy = 0;
		NumberofOutputNeurons = 1;	//锟斤拷锟斤拷锟节碉拷锟斤拷默锟斤拷为1
		
	}
	public elm(){
		
	}
	//the first line of dataset file must be the number of rows and columns,and number of classes if neccessary
	//the first column is the norminal class value 0,1,2...
	//if the class value is 1,2...,number of classes should plus 1

	//锟斤拷锟斤拷锟侥硷拷
	public DenseMatrix loadmatrix(String filename) throws IOException{
		
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));

		String firstlineString = reader.readLine();
		String []strings = firstlineString.split(splitmark);
		int m = Integer.parseInt(strings[0]);//锟斤拷锟斤拷锟斤拷锟斤拷
		int n = Integer.parseInt(strings[1]);//锟斤拷锟斤拷锟斤拷锟斤拷
		if(strings.length > 2)   //锟斤拷锟斤拷时锟斤拷取锟斤拷锟斤拷锟�
			NumberofOutputNeurons = Integer.parseInt(strings[2]);
				
		
		DenseMatrix matrix = new DenseMatrix(m, n);//锟斤拷锟斤拷m*n锟斤拷锟斤拷锟斤拷锟斤拷锟�
		
		firstlineString = reader.readLine();//一锟斤拷锟叫讹拷取锟斤拷锟斤拷
		int i = 0;
		while (i<m) {
			String []datatrings = firstlineString.split(splitmark);
			for (int j = 0; j < n; j++) {
				matrix.set(i, j, Double.parseDouble(datatrings[j]));
			}
			i++;
			firstlineString = reader.readLine();
		}
		reader.close();
		return matrix;
	}
	
	
	public void train(String TrainingData_File) throws NotConvergedException{
		try {
			train_set = loadmatrix(TrainingData_File);//锟斤拷锟斤拷锟斤拷锟捷ｏ拷锟斤拷锟斤拷锟斤拷转锟斤拷锟缴撅拷锟斤拷
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		train();//锟斤拷锟斤拷训锟斤拷锟斤拷锟斤拷锟斤拷锟矫碉拷锟斤拷锟饺拷兀锟斤拷锟酵筹拷锟街达拷锟绞憋拷锟斤拷准确锟斤拷
	}
	/*shenchu**********************/
	public void train(DenseMatrix dm ,int num) throws NotConvergedException{
		int m=dm.numRows();
		int n=dm.numColumns();
		//System.out.println(dm.get(100, 5));
		NumberofOutputNeurons=num;
		train_set=new DenseMatrix(m, n);
		train_set=dm.copy();
		//System.out.println(train_set.numRows()+" "+train_set.numColumns());
		//System.out.println(train_set.get(3, 3));
		train();
	}
	public void train(double [][]traindata) throws NotConvergedException{
	
		//classification require a the number of class
		
		train_set = new DenseMatrix(traindata);
		int m = train_set.numRows();
		if(Elm_Type == 1){
			double maxtag = traindata[0][0];
			for (int i = 0; i < m; i++) {
				if(traindata[i][0] > maxtag)
					maxtag = traindata[i][0];
			}
			NumberofOutputNeurons = (int)maxtag+1;
		}
		train();
	}
	
	
	private void train() throws NotConvergedException{
		
		numTrainData = train_set.numRows();//锟斤拷锟斤拷锟斤拷
		NumberofInputNeurons = train_set.numColumns() - 1;//锟斤拷锟斤拷锟叫ｏ拷锟斤拷一锟斤拷为锟斤拷锟斤拷锟较�
		InputWeight = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);//锟斤拷锟斤拷权锟截ｏ拷锟斤拷锟斤拷锟绞硷拷锟�
		
		DenseMatrix transT = new DenseMatrix(numTrainData, 1);//锟斤拷锟斤拷锟较�
		DenseMatrix transP = new DenseMatrix(numTrainData, NumberofInputNeurons);//锟斤拷锟斤拷锟斤拷息
//		for (int i = 0; i < numTrainData; i++) {
//			transT.set(i, 0, train_set.get(i, 0));   //  类标在第一行时选择
//			for (int j = 1; j <= NumberofInputNeurons; j++)
//				transP.set(i, j-1, train_set.get(i, j));
//		}
		for (int i = 0; i < numTrainData; i++) {
			transT.set(i, 0, train_set.get(i, NumberofInputNeurons));   //  类标在最后一行时选择
			for (int j = 0; j <NumberofInputNeurons; j++)
				transP.set(i, j, train_set.get(i, j));
		}
		
		T = new DenseMatrix(1,numTrainData);
		DenseMatrix P = new DenseMatrix(NumberofInputNeurons,numTrainData);
		transT.transpose(T);
		transP.transpose(P);
		
		if(Elm_Type != 0)	//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟阶拷锟斤拷锟斤拷锟�
		{
			label = new int[NumberofOutputNeurons];
			for (int i = 0; i < NumberofOutputNeurons; i++) {
				label[i] = i;							//锟斤拷锟斤拷锟较拷锟�0锟斤拷始
			}
			DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,numTrainData);//锟斤拷锟斤拷锟斤拷
			tempT.zero();//锟斤拷始锟斤拷为0
			for (int i = 0; i < numTrainData; i++){			//锟斤拷取锟斤拷锟斤拷锟较拷锟斤拷锟斤拷锟斤拷诟锟斤拷锟斤拷锟斤拷锟�1
					int j = 0;
			        for (j = 0; j < NumberofOutputNeurons; j++){
			            if (label[j] == T.get(0, i))
			                break; 
			        }
			        tempT.set(j, i, 1); 
			}
			
			T = new DenseMatrix(NumberofOutputNeurons,numTrainData);	// T=temp_T*2-1;锟斤拷锟节革拷锟斤拷为1锟斤拷锟斤拷锟斤拷锟斤拷为-1
			for (int i = 0; i < NumberofOutputNeurons; i++){
		        for (int j = 0; j < numTrainData; j++)
		        	T.set(i, j, tempT.get(i, j)*2-1);
			}
			
			transT = new DenseMatrix(numTrainData,NumberofOutputNeurons);//锟斤拷锟斤拷锟较�
			T.transpose(transT);
			
		} 	
		
		long start_time_train = System.currentTimeMillis();//锟斤拷始时锟斤拷
		// Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
		// InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
		
		BiasofHiddenNeurons = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);//偏锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷始锟斤拷
		
		DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
		InputWeight.mult(P, tempH);//锟斤拷锟斤拷值*权锟截ｏ拷锟斤拷锟斤拷tempH
		//DenseMatrix ind = new DenseMatrix(1, numTrainData);
		
		DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTrainData);//偏锟斤拷锟斤拷螅锟斤拷锟斤拷锟接ｏ拷锟斤拷小匹锟斤拷
		
		for (int j = 0; j < numTrainData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH.add(BiasMatrix);//锟斤拷锟斤拷偏锟斤拷
		DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainData);//锟斤拷锟斤拷锟斤拷螅锟斤拷f(x)
		//锟斤拷锟筋函锟斤拷
		if(func.startsWith("sig")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTrainData; i++) {
					double temp = tempH.get(j, i);
					temp = 1.0f/ (1 + Math.exp(-temp));
					H.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("sin")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTrainData; i++) {
					double temp = tempH.get(j, i);
					temp = Math.sin(temp);
					H.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("hardlim")){
			//If you need it ,you can absolutely complete it yourself
		}
		else if(func.startsWith("tribas")){
			//If you need it ,you can absolutely complete it yourself	
		}
		else if(func.startsWith("radbas")){
			//If you need it ,you can absolutely complete it yourself
		}

		DenseMatrix Ht = new DenseMatrix(numTrainData,NumberofHiddenNeurons);
		H.transpose(Ht);//锟斤拷锟斤拷锟斤拷锟阶拷锟�
		Inverse invers = new Inverse(Ht);
		DenseMatrix pinvHt = invers.getMPInverse();			//NumberofHiddenNeurons*numTrainData锟斤拷锟矫碉拷锟斤拷锟斤拷锟斤拷
		//DenseMatrix pinvHt = invers.getMPInverse(0.000001); //fast method, PLEASE CITE in your paper properly: 
		//Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010.
		
		OutputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
		//OutputWeight=pinv(H') * T';  
		pinvHt.mult(transT, OutputWeight);//锟斤拷锟饺拷锟�=锟斤拷锟斤拷锟斤拷锟侥癸拷锟斤拷锟斤拷*目锟斤拷锟斤拷锟�
		
		long end_time_train = System.currentTimeMillis();
		TrainingTime = (end_time_train - start_time_train)*1.0f/1000;

		DenseMatrix Yt = new DenseMatrix(numTrainData,NumberofOutputNeurons);//锟斤拷锟皆わ拷锟街�
		Ht.mult(OutputWeight,Yt);//预锟斤拷值=锟斤拷锟斤拷锟斤拷锟�*锟斤拷锟饺拷锟�
		Y = new DenseMatrix(NumberofOutputNeurons,numTrainData);
		Yt.transpose(Y);
		
		if(Elm_Type == 0){   //锟截癸拷准确锟斤拷
			double MSE = 0;
			for (int i = 0; i < numTrainData; i++) {
				MSE += (Yt.get(i, 0) - transT.get(i, 0))*(Yt.get(i, 0) - transT.get(i, 0));
			}
			TrainingAccuracy = Math.sqrt(MSE/numTrainData);
		}
		
		else if(Elm_Type == 1){  //锟斤拷锟斤拷准确锟斤拷
			float MissClassificationRate_Training=0;
		    
		    for (int i = 0; i < numTrainData; i++) {
				double maxtag1 = Y.get(0, i);
				int tag1 = 0;			//预锟斤拷锟斤拷锟斤拷签
				double maxtag2 = T.get(0, i);
				int tag2 = 0;			//目锟斤拷锟斤拷锟斤拷签
		    	for (int j = 1; j < NumberofOutputNeurons; j++) {
					if(Y.get(j, i) > maxtag1){
						maxtag1 = Y.get(j, i);
						tag1 = j;
					}
					if(T.get(j, i) > maxtag2){
						maxtag2 = T.get(j, i);
						tag2 = j;
					}
				}
		    	if(tag1 != tag2)  //锟叫讹拷预锟斤拷锟斤拷目锟斤拷锟角凤拷一锟斤拷
		    		MissClassificationRate_Training ++;
			}
		    TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/numTrainData;//训锟斤拷锟斤拷准确锟斤拷
		}
		
	}
	//--锟斤拷锟皆诧拷锟斤拷--
	public void test(DenseMatrix  d)throws IOException{
		
		test_set=d.copy();
		numTestData = test_set.numRows();   //锟斤拷锟捷硷拷锟斤拷锟斤拷
		DenseMatrix ttestT = new DenseMatrix(numTestData, 1);	//锟斤拷锟捷硷拷锟斤拷锟斤拷锟较拷锟斤拷锟�
		DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);	//锟斤拷锟捷硷拷锟斤拷锟斤拷锟斤拷息锟斤拷锟斤拷
//		for (int i = 0; i < numTestData; i++) {			//第一列为类标的
//			ttestT.set(i, 0, test_set.get(i, 0));
//			for (int j = 1; j <= NumberofInputNeurons; j++)
//				ttestP.set(i, j-1, test_set.get(i, j));
//		}
		for (int i = 0; i < numTestData; i++) {			//最后一列为类标的
			ttestT.set(i, 0, test_set.get(i,NumberofInputNeurons));
			for (int j = 0; j <NumberofInputNeurons; j++)
				ttestP.set(i, j, test_set.get(i, j));
		}
		testT = new DenseMatrix(1,numTestData);			//锟斤拷锟斤拷锟较拷锟斤拷锟�
		testP = new DenseMatrix(NumberofInputNeurons,numTestData);		//锟斤拷锟斤拷锟斤拷息锟斤拷锟斤拷
		ttestT.transpose(testT);
		ttestP.transpose(testP);
		
		long start_time_test = System.currentTimeMillis();		//锟斤拷始时锟斤拷
		DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);	//锟斤拷时锟斤拷锟斤拷锟斤拷锟斤拷*锟斤拷锟斤拷权锟斤拷+偏锟斤拷
		InputWeight.mult(testP, tempH_test);	//锟斤拷锟斤拷*锟斤拷锟斤拷权锟斤拷
		DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);	//偏锟斤拷锟斤拷螅锟窖碉拷锟绞币伙拷锟�
		
		for (int j = 0; j < numTestData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH_test.add(BiasMatrix2);//锟斤拷锟斤拷偏锟斤拷
		DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);//锟斤拷锟斤拷锟斤拷锟絝(x)
		
		if(func.startsWith("sig")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = 1.0f/ (1 + Math.exp(-temp));
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("sin")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = Math.sin(temp);
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("hardlim")){
			
		}
		else if(func.startsWith("tribas")){
	
		}
		else if(func.startsWith("radbas")){
			
		}
		
		DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
		H_test.transpose(transH_test);
		DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);//预锟斤拷值锟斤拷锟斤拷
		transH_test.mult(OutputWeight,Yout);
		
		//*****shenchu***
		OutMatrix=new DenseMatrix(numTestData, NumberofOutputNeurons);
		OutMatrix=Yout.copy();
		
		DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
		Yout.transpose(testY);
		
		long end_time_test = System.currentTimeMillis();//锟斤拷锟皆斤拷锟斤拷
		TestingTime = (end_time_test - start_time_test)*1.0f/1000;
		//writefile(Yout, "C:\\Users\\shen\\Desktop\\write1.txt");
		//锟截癸拷准确锟斤拷
		if(Elm_Type == 0){
			double MSE = 0;
			for (int i = 0; i < numTestData; i++) {
				MSE += (Yout.get(i, 0) - testT.get(0,i))*(Yout.get(i, 0) - testT.get(0,i));
			}
			TestingAccuracy = Math.sqrt(MSE/numTestData);
		}
		
		
		//锟斤拷锟斤拷准确锟斤拷
		else if(Elm_Type == 1){

			DenseMatrix temptestT = new DenseMatrix(NumberofOutputNeurons,numTestData);//锟斤拷锟斤拷锟较拷锟斤拷锟斤拷锟斤拷锟轿�1锟斤拷
			for (int i = 0; i < numTestData; i++){
					int j = 0;
			        for (j = 0; j < NumberofOutputNeurons; j++){
			            if (label[j] == testT.get(0, i))
			                break; 
			        }
			        temptestT.set(j, i, 1); 
			}
			
			testT = new DenseMatrix(NumberofOutputNeurons,numTestData);	//锟斤拷锟斤拷锟较拷锟斤拷锟斤拷锟斤拷锟轿�1锟斤拷锟斤拷锟斤拷锟斤拷为-1
			for (int i = 0; i < NumberofOutputNeurons; i++){
		        for (int j = 0; j < numTestData; j++)
		        	testT.set(i, j, temptestT.get(i, j)*2-1);
			}

		    float MissClassificationRate_Testing=0;

		    for (int i = 0; i < numTestData; i++) {
				double maxtag1 = testY.get(0, i);
				int tag1 = 0;
				double maxtag2 = testT.get(0, i);
				int tag2 = 0;
		    	for (int j = 1; j < NumberofOutputNeurons; j++) {
					if(testY.get(j, i) > maxtag1){
						maxtag1 = testY.get(j, i);
						tag1 = j;
					}
					if(testT.get(j, i) > maxtag2){
						maxtag2 = testT.get(j, i);
						tag2 = j;
					}
				}
		    	if(tag1 != tag2)
		    		MissClassificationRate_Testing ++;
			}
		    TestingAccuracy = 1 - MissClassificationRate_Testing*1.0f/numTestData;//锟斤拷锟皆碉拷准确锟斤拷
		    
		}
	}
	//shenchu**********************************
	public  void writefile(DenseMatrix dm_t,String filename) throws IOException{
		BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename, true)));
		for(int i=0;i<dm_t.numRows();i++){
			String t=new String();
			for(int j=0;j<dm_t.numColumns();j++){
				if(j==(dm_t.numColumns()-1))
					t=t+Double.toString(dm_t.get(i, j));
				else
				t=t+Double.toString(dm_t.get(i, j))+" ";
			}
            
			bw.append(t);
			if(i!=dm_t.numRows()-1)
			bw.newLine();
			
		}
		bw.close();
	}
	
	public double[] testOut(double[][] inpt){
		test_set = new DenseMatrix(inpt);
		return testOut();
	}
	public double[] testOut(double[] inpt){
		test_set = new DenseMatrix(new DenseVector(inpt));
		return testOut();
	}
	//Output	numTestData*NumberofOutputNeurons
	private double[] testOut(){
		numTestData = test_set.numRows();
		NumberofInputNeurons = test_set.numColumns()-1;
		
		DenseMatrix ttestT = new DenseMatrix(numTestData, 1);
		DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);
		for (int i = 0; i < numTestData; i++) {
			ttestT.set(i, 0, test_set.get(i, 0));
			for (int j = 1; j <= NumberofInputNeurons; j++)
				ttestP.set(i, j-1, test_set.get(i, j));
		}
		
		testT = new DenseMatrix(1,numTestData);
		testP = new DenseMatrix(NumberofInputNeurons,numTestData);
		ttestT.transpose(testT);
		ttestP.transpose(testP);
		//test_set.transpose(testP);
		
		DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		InputWeight.mult(testP, tempH_test);
		DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		
		for (int j = 0; j < numTestData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH_test.add(BiasMatrix2);
		DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		
		if(func.startsWith("sig")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = 1.0f/ (1 + Math.exp(-temp));
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("sin")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = Math.sin(temp);
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("hardlim")){
			
		}
		else if(func.startsWith("tribas")){
	
		}
		else if(func.startsWith("radbas")){
			
		}
		
		DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
		H_test.transpose(transH_test);
		DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
		transH_test.mult(OutputWeight,Yout);
		
		//DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
		//Yout.transpose(testY);
		
		double[] result = new double[numTestData];                         
		
		if(Elm_Type == 0){
			for (int i = 0; i < numTestData; i++)
				result[i] = Yout.get(i, 0);
		}
		
		else if(Elm_Type == 1){
			for (int i = 0; i < numTestData; i++) {
				int tagmax = 0;
				double tagvalue = Yout.get(i, 0);
				for (int j = 1; j < NumberofOutputNeurons; j++)
				{
					if(Yout.get(i, j) > tagvalue){
						tagvalue = Yout.get(i, j);
						tagmax = j;
					}
		
				}
				result[i] = tagmax;
			}
		}
		return result;
	}
	public DenseMatrix getTrainMatrix(){
		return train_set;
	}
	public DenseMatrix getTestMatrix(){
		return test_set;
	}
	public float getTrainingTime() {  //锟斤拷取训锟斤拷时锟斤拷
		return TrainingTime;
	}
	public double getTrainingAccuracy() {	//锟斤拷取训锟斤拷准确锟斤拷
		return TrainingAccuracy;
	}
	public float getTestingTime() {			//锟斤拷取锟斤拷锟斤拷时锟斤拷
		return TestingTime;
	}
	public double getTestingAccuracy() {	//锟斤拷取锟斤拷锟斤拷准确锟斤拷
		return TestingAccuracy;
	}
	
	public int getNumberofInputNeurons() {	//锟斤拷取锟斤拷锟斤拷锟节碉拷锟斤拷
		return NumberofInputNeurons;
	}
	public int getNumberofHiddenNeurons() {	//锟斤拷取锟斤拷锟截诧拷诘锟斤拷锟�
		return NumberofHiddenNeurons;
	}
	public int getNumberofOutputNeurons() {	//锟斤拷取锟斤拷锟斤拷锟节碉拷锟斤拷
		return NumberofOutputNeurons;
	}
	
	public DenseMatrix getInputWeight() {	//锟斤拷取锟斤拷锟斤拷权锟截撅拷锟斤拷
		return InputWeight;
	}
	
	public DenseMatrix getBiasofHiddenNeurons() {	//锟斤拷取偏锟斤拷锟斤拷锟�
		return BiasofHiddenNeurons;
	}
	
	public DenseMatrix getOutputWeight() {	//锟斤拷取锟斤拷锟饺拷鼐锟斤拷锟�
		return OutputWeight;
	}
	
	public DenseMatrix getOutMatrix(){	//锟斤拷取锟斤拷锟皆斤拷锟�
		return OutMatrix;
	}

	//for predicting a data file based on a trained model.
	public void testgetoutput(String filename) throws IOException {
		
		try {
			test_set = loadmatrix(filename);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		numTestData = test_set.numRows();
		NumberofInputNeurons = test_set.numColumns() - 1;
		
		
		double rsum = 0;
		double []actual = new double[numTestData];
		
		double [][]data = new double[numTestData][NumberofInputNeurons];
		for (int i = 0; i < numTestData; i++) {
			actual[i] = test_set.get(i, 0);
			for (int j = 0; j < NumberofInputNeurons; j++)
				data[i][j] = test_set.get(i, j+1);
		}
		
		double[] output = testOut(data);
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("Output")));
		for (int i = 0; i < numTestData; i++) {
			
			writer.write(String.valueOf(output[i]));
			writer.newLine();
			
			if(Elm_Type == 0){
					rsum += (output[i] - actual[i])*(output[i] - actual[i]);
			}
			
			if(Elm_Type == 1){
				if(output[i] == actual[i])
					rsum ++;
			}
			
		}
		writer.flush();
		writer.close();
		
		if(Elm_Type == 0)
			System.out.println("Regression GetOutPut RMSE: "+Math.sqrt(rsum*1.0f/numTestData));
		else if(Elm_Type == 1)
			System.out.println("Classfy GetOutPut Right: "+rsum*1.0f/numTestData);
	}
	
}
