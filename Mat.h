#pragma once

class Mat
{
private:
	double* p_Date;
	int H, W;

public:
	Mat();
	~Mat();
	Mat(double *pDate,int h,int w);
	Mat(double** ppDate, int h, int w);
	Mat(const Mat& m);
	Mat(int h, int w);
	Mat(const Mat& m, int y1, int x1, int y2, int x2);
	void create(int h, int w);
	void load(const Mat& m);
	void load(double* pDate, int h, int w);
	void load(double** ppDate, int h, int w);
	void distroy();
	void setCol(double* pDate, int x);
	void setRow(double* pDate, int y);
	void copyTo(Mat& m);
	void copyTo(Mat& m, int y1, int x1, int y2, int x2);
	void set(double data, int y, int x);
	bool isAllocated();
	bool same(const Mat& m);
	double* convertTo();
	Mat clone();
	Mat getCol(int x);
	Mat getRow(int y);
	Mat getSubMat(int y1, int x1, int y2, int x2);
	Mat diag();
	Mat eye(int n);
	Mat ones(int h, int w);
	Mat inverse();
	Mat plus(const Mat& m);
	Mat minus(const Mat& m);
	Mat mult(const Mat& m);
	Mat div(const Mat& m);
	Mat matchInRow(const Mat m);
	Mat matchInCol(const Mat m);
	int Cols();
	int Rows();
	double sum();
	double sum(int y1, int x1, int y2, int x2);
	double get(int y, int x);
	double* sumInRow();
	double* sumInCol();
	Mat& operator=(const Mat& m);
};

