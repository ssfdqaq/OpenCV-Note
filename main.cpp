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
	const string fileform = "*.png";//�ļ���ʽ
	const string perfileReadPath = "charSamples";//�ļ�ǰ��·��

	const int sample_mun_perclass = 25;//ѵ���ַ�ÿ������
	const int class_mun = 10;//ѵ���ַ�����

	const int image_cols = 8;//ͼƬ��Ϊ��,�����е���
	const int image_rows = 16;//ͼƬ��Ϊ��

    string  fileReadName,fileReadPath;

	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//ÿһ��һ��ѵ������
	float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//ѵ��������ǩ


	/*-----------------------------------------��ȡͼƬ------------------------------------*/
	for(int i=0;i<=class_mun-1;++i)//��ͬ��
	{
		//��ȡÿ�����ļ���������ͼ��
		int j = 0;//ÿһ���ȡͼ���������
		sprintf(temp, "%d", i);//��˳���ͼ  
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;//�ļ���ȡ·��
		cout<<"File"<<i<<endl;
		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//����ʧ��
		}
		//do-whileѭ����ȡ
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//����.��..
				continue;
			j++;//��ȡһ��ͼ
			printf("%s\n",WcharToChar(pNextInfo.cFileName));//�Զ����ͼƬ���д���

			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);//����ͼ��  
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage,resizeImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���

			threshold(resizeImage,trainImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);//�ԻҶ�ͼ�������ֵ�����õ���ֵͼ��  

			for(int k = 0; k<image_rows*image_cols; ++k)//ÿ��ͼƬ
			{
				trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.data[k];//����ֵ�����ݿ�����trainingData��  
			}

		} 
		while (FindNextFile(hFile,&pNextInfo) && j<sample_mun_perclass);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ

	}

    // ����ѵ������Mat����  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout<<"trainingDataMat--OK!"<<endl;// ����ѵ������Mat  

	// ���ñ�ǩ����Mat���
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
	cout<<"labelsMat--OK!"<<endl;//���ñ�ǩ����Mat  


	/*-----------------------------------------ѵ������------------------------------------*/
	cout<<"training start...."<<endl;
	CvANN_MLP bp;//bp������  

	 //����bp������Ĳ���  
	CvANN_MLP_TrainParams params;//��,����  1
	params.train_method=CvANN_MLP_TrainParams::BACKPROP;//ѵ������:���򴫲���  
	params.bp_dw_scale=0.001;//Ȩֵ������  0.001
	params.bp_moment_scale=0.1; //Ȩֵ���³���  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //���ý�������term_crit����ֹ������10000,0.0001 
	//�������������������(CV_TERMCRIT_ITER)�������Сֵ(CV_TERMCRIT_EPS)��һ����һ���ﵽ��������ֹѵ����  
	
	//����bp������  
	Mat layerSizes=(Mat_<int>(1,5) << image_rows*image_cols,128,128,128,class_mun);//1�����룬1�������3�����ز�
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM,1.0,1.0);//CvANN_MLP::SIGMOID_SYM�ڵ�ʹ�õĺ���  ����������
	//CvANN_MLP::GAUSSIAN
	//CvANN_MLP::IDENTITY
	cout<<"training...."<<endl;
	bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);//trainingDataMat:�������,�洢������ѵ������������  �������ѵ��

	bp.save("../bpcharModel.xml"); //����ѵ��  
	cout<<"training finish...bpModel1.xml saved "<<endl;


	//����������
	cout<<"Text:"<<endl;
	Mat test_image = imread("9.png",CV_LOAD_IMAGE_GRAYSCALE);//�������ͼ  
	Mat test_temp;
	resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
	threshold(test_temp,test_temp,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);//��ֵ��  
	Mat_<float>sampleMat(1,image_rows*image_cols); 
	for(int i = 0; i<image_rows*image_cols; ++i)  
	{  
		sampleMat.at<float>(0,i) = (float)test_temp.at<uchar>(i/8,i%8);  //��test���ݣ�unchar��copy��sampleMat(float)��ͼ������   
	}  

	Mat responseMat;  
	bp.predict(sampleMat,responseMat);   //������predict���������ǵõ�һ���������������һ��1*nClass���������� ʶ��  
	                                                                      //����ÿһ��˵�������������Ƴ̶ȣ�0-1֮�䣩��Ҳ����˵�����Ŷ�  
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat,NULL,&maxVal,NULL,&maxLoc);  //��С���ֵ  
	cout<<"Result:"<<maxLoc.x<<" Similarity:"<<maxVal*100<<"%"<<endl;
	imshow("test_image",test_image);  
	waitKey(0);

	return 0;
}