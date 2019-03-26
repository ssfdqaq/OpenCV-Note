#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

char* WcharToChar(const wchar_t* wp)  
{  
	char *m_char;
	int len= WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),NULL,0,NULL,NULL);  
	m_char=new char[len+1];  
	WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),m_char,len,NULL,NULL);  
	m_char[len]='\0';  
	return m_char;  
}  

wchar_t* CharToWchar(const char* c)  
{   
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP,0,c,strlen(c),NULL,0);  
	m_wchar=new wchar_t[len+1];  
	MultiByteToWideChar(CP_ACP,0,c,strlen(c),m_wchar,len);  
	m_wchar[len]='\0';  
	return m_wchar;  
}  

wchar_t* StringToWchar(const string& s)  
{  
	const char* p=s.c_str();  
	return CharToWchar(p);  
}

int main()
{
	const string fileform = "*.png";//文件格式
	const string perfileReadPath = "charSamples";//文件前置路径

	const int sample_mun_perclass = 25;//训练字符每类数量
	const int class_mun = 10;//训练字符类数

	const int image_cols = 8;//图片分为列,可自行调整
	const int image_rows = 16;//图片分为行

    string  fileReadName,fileReadPath;

	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//每一行一个训练样本
	float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//训练样本标签


	/*-----------------------------------------读取图片------------------------------------*/
	for(int i=0;i<=class_mun-1;++i)//不同类
	{
		//读取每个类文件夹下所有图像
		int j = 0;//每一类读取图像个数计数
		sprintf(temp, "%d", i);//按顺序读图  
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;//文件读取路径
		cout<<"File"<<i<<endl;
		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//搜索失败
		}
		//do-while循环读取
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//过滤.和..
				continue;
			j++;//读取一张图
			printf("%s\n",WcharToChar(pNextInfo.cFileName));//对读入的图片进行处理

			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);//读入图像  
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage,resizeImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现

			threshold(resizeImage,trainImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);//对灰度图像进行阈值操作得到二值图像  

			for(int k = 0; k<image_rows*image_cols; ++k)//每个图片
			{
				trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.data[k];//将二值化数据拷贝到trainingData中  
			}

		} 
		while (FindNextFile(hFile,&pNextInfo) && j<sample_mun_perclass);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片

	}

    // 设置训练数据Mat输入  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout<<"trainingDataMat--OK!"<<endl;// 设置训练数据Mat  

	// 设置标签数据Mat输出
	for(int i=0;i<=class_mun-1;++i)
	{
		for(int j=0;j<=sample_mun_perclass-1;++j)
		{
			for(int k = 0;k<class_mun;++k)
			{
				if(k==i)
					labels[i*sample_mun_perclass + j][k] = 1;
				else labels[i*sample_mun_perclass + j][k] = 0;
			}
		}
	}
	Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1,labels);
	cout<<"labelsMat:"<<endl;
	cout<<labelsMat<<endl;
	cout<<"labelsMat--OK!"<<endl;//设置标签数据Mat  


	/*-----------------------------------------训练代码------------------------------------*/
	cout<<"training start...."<<endl;
	CvANN_MLP bp;//bp神经网络  

	 //设置bp神经网络的参数  
	CvANN_MLP_TrainParams params;//类,参数  1
	params.train_method=CvANN_MLP_TrainParams::BACKPROP;//训练方法:误差反向传播法  
	params.bp_dw_scale=0.001;//权值更新率  0.001
	params.bp_moment_scale=0.1; //权值更新冲量  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //设置结束条件term_crit：终止条件，10000,0.0001 
	//它包括了两项，迭代次数(CV_TERMCRIT_ITER)和误差最小值(CV_TERMCRIT_EPS)，一旦有一个达到条件就终止训练。  
	
	//设置bp神经网络  
	Mat layerSizes=(Mat_<int>(1,5) << image_rows*image_cols,128,128,128,class_mun);//1个输入，1个输出，3个隐藏层
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM,1.0,1.0);//CvANN_MLP::SIGMOID_SYM节点使用的函数  创建神经网络
	//CvANN_MLP::GAUSSIAN
	//CvANN_MLP::IDENTITY
	cout<<"training...."<<endl;
	bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);//trainingDataMat:输入矩阵,存储了所有训练样本的特征  神经网络的训练

	bp.save("../bpcharModel.xml"); //保存训练  
	cout<<"training finish...bpModel1.xml saved "<<endl;


	//测试神经网络
	cout<<"Text:"<<endl;
	Mat test_image = imread("9.png",CV_LOAD_IMAGE_GRAYSCALE);//读入测试图  
	Mat test_temp;
	resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
	threshold(test_temp,test_temp,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);//二值化  
	Mat_<float>sampleMat(1,image_rows*image_cols); 
	for(int i = 0; i<image_rows*image_cols; ++i)  
	{  
		sampleMat.at<float>(0,i) = (float)test_temp.at<uchar>(i/8,i%8);  //将test数据（unchar）copy到sampleMat(float)中图像特征   
	}  

	Mat responseMat;  
	bp.predict(sampleMat,responseMat);   //过调用predict函数，我们得到一个输出向量，它是一个1*nClass的行向量， 识别  
	                                                                      //其中每一列说明它与该类的相似程度（0-1之间），也可以说是置信度  
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat,NULL,&maxVal,NULL,&maxLoc);  //最小最大值  
	cout<<"Result:"<<maxLoc.x<<" Similarity:"<<maxVal*100<<"%"<<endl;
	imshow("test_image",test_image);  
	waitKey(0);

	return 0;
}