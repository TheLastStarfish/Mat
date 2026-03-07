#include "Mat.h"
#include <cstring>
#include <exception>
#include <cmath>
#include <algorithm>

Mat::Mat()
{
	p_Data = nullptr;
	H = 0;
	W = 0;
}

Mat::Mat(double* pDate, int h, int w) :p_Data(nullptr), H(0), W(0)
{
	load(pDate, h, w);
}

Mat::Mat(double** ppDate, int h, int w) :p_Data(nullptr), H(0), W(0)
{
	load(ppDate, h, w);
}

Mat::Mat(const Mat& m) :p_Data(nullptr), H(0), W(0)
{
	load(m);
}

Mat::Mat(int h, int w) :p_Data(nullptr), H(0), W(0)
{
	create(h, w);
}

Mat::Mat(const Mat& m, int y1, int x1, int y2, int x2) :p_Data(nullptr), H(0), W(0)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0
		|| y1 >= m.H || x1 >= m.W || y2 >= m.H || x2 >= m.W)
		throw std::exception("子矩阵坐标越界");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);

	int h = y2 - y1 + 1;
	int w = x2 - x1 + 1;
	create(h, w);

	for (int i = 0; i < h; ++i)
		memcpy(p_Data + i * w, m.p_Data + (y1 + i) * m.W + x1, sizeof(double) * w);
}

Mat::~Mat()
{
	distroy();
}

void Mat::create(int h, int w)
{
	if(h == 0 || w == 0)
		throw std::exception("长度不能为0");
	if (p_Data != nullptr)
		distroy();
	p_Data = new double[h * w] {};
	H = h;
	W = w;
}

void Mat::distroy()
{
	if (p_Data != nullptr)
		delete[] p_Data;
	p_Data = nullptr;
	H = 0;
	W = 0;
}

void Mat::load(double* pDate, int h, int w)
{
	create(h, w);
	memcpy(p_Data, pDate, sizeof(double) * h * w);
}

void Mat::load(double** ppDate, int h, int w)
{
	create(h, w);
	for (int i = 0; i < h; i++)
		memcpy(p_Data + i * w, ppDate[i], sizeof(double) * w);
}

void Mat::load(const Mat& m)
{
	create(m.H, m.W);
	memcpy(p_Data, m.p_Data, sizeof(double) * H * W);
}

void Mat::setCol(double* pDate, int x)
{
	if (x < 0 || x >= W)
		throw std::exception("列索引越界");
	for (int i = 0; i < H; i++)
		p_Data[i * W + x] = pDate[i];
}

void Mat::setRow(double* pDate, int y)
{
	if (y < 0 || y >= H)
		throw std::exception("行索引越界");
	memcpy(p_Data + y * W, pDate, sizeof(double) * W);
}

void Mat::copyTo(Mat& m)
{
	m.load(*this);
}

void Mat::copyTo(Mat& m, int y1, int x1, int y2, int x2)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0 
		|| y1 >= H || x1 >= W || y2 >= H || x2 >= W)
		throw std::exception("子矩阵坐标越界");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);
	int h = y2 - y1 + 1;
	int w = x2 - x1 + 1;
	m.create(h, w);
	for (int i = 0; i < h; ++i)
		memcpy(m.p_Data + i * w, p_Data + (y1 + i) * W + x1, sizeof(double) * w);
}

void Mat::set(double data, int y, int x)
{
	if (y < 0 || x < 0 || y >= H || x >= W)
		throw std::exception("索引越界");
	p_Data[y * W + x] = data;
}

bool Mat::isAllocated()
{
	return p_Data != nullptr;
}

bool Mat::same(const Mat& m)
{
	if (H != m.H || W != m.W) return false;
	return memcmp(p_Data, m.p_Data, sizeof(double) * H * W) == 0;
}

double* Mat::convertTo()
{
	return p_Data;
}

Mat Mat::clone()
{
	Mat m;
	copyTo(m);
	return m;
}

Mat Mat::getCol(int x)
{
	if (x < 0 || x >= W)
		throw std::exception("列索引越界");
	Mat m(H, 1);
	for (int i = 0; i < H; i++)
		m.p_Data[i] = p_Data[i * W + x];
	return m;
}

Mat Mat::getRow(int y)
{
	if (y < 0 || y >= H)
		throw std::exception("行索引越界");
	Mat m(1, W);
	memcpy(m.p_Data, p_Data + y * W, sizeof(double) * W);
	return m;
}

Mat Mat::getSubMat(int y1, int x1, int y2, int x2)
{
	Mat m(*this, y1, x1, y2, x2);
	return m;
}

Mat Mat::diag()
{
	int n = H < W ? H : W;
	Mat m(n, 1);
	for (int i = 0; i < n; i++)
		m.p_Data[i] = p_Data[i * W + i];
	return m;
}

Mat Mat::eye(int n)
{
	Mat m(n, n);
	for (int i = 0; i < n; i++)
		m.p_Data[i * n + i] = 1;
	return m;
}

Mat Mat::ones(int h, int w)
{
	Mat m(h, w);
	std::fill_n(m.p_Data, h * w, 1);
	return m;
}

Mat Mat::inverse()
{
	Mat m(W, H);
	for(int i = 0; i < H; i++)
		for (int j = 0; j < W; j++)
			m.p_Data[j * H + i] = p_Data[i * W + j];
	return m;
}

Mat Mat::plus(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::exception("矩阵维度不匹配");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Data[i] = p_Data[i] + m.p_Data[i];
	return r;
}

Mat Mat::minus(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::exception("矩阵维度不匹配");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Data[i] = p_Data[i] - m.p_Data[i];
	return r;
}

Mat Mat::mult(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::exception("矩阵维度不匹配");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Data[i] = p_Data[i] * m.p_Data[i];
	return r;
}

Mat Mat::div(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::exception("矩阵维度不匹配");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Data[i] = p_Data[i] / m.p_Data[i];
	return r;
}

Mat Mat::matchInRow(const Mat m)
{
	if (H != m.H)
		throw std::exception("行数不匹配");
	Mat r(H, W + m.W);
	for (int i = 0; i < H; i++) {
		memcpy(r.p_Data + i * r.W, p_Data + i * W, sizeof(double) * W);
		memcpy(r.p_Data + i * r.W + W, m.p_Data + i * m.W, sizeof(double) * m.W);
	}
	return r;
}

Mat Mat::matchInCol(const Mat m)
{
	if (W != m.W)
		throw std::exception("列数不匹配");
	Mat r(H + m.H, W);
	for(int i = 0; i < H; i++)
		memcpy(r.p_Data + i * W, p_Data + i * W, sizeof(double) * W);
	for(int i = 0; i < m.H; i++)
		memcpy(r.p_Data + (H + i) * W, m.p_Data + i * W, sizeof(double) * W);
	return r;
}

int Mat::Cols()
{
	return W;
}

int Mat::Rows()
{
	return H;
}

double Mat::sum()
{
	double s = 0;
	for (int i = 0; i < H * W; i++)
		s += p_Data[i];
	return s;
}

double Mat::sum(int y1, int x1, int y2, int x2)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0
		|| y1 >= H || x1 >= W || y2 >= H || x2 >= W)
		throw std::exception("子矩阵坐标越界");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);
	double s = 0;
	for (int i = y1; i <= y2; i++)
		for (int j = x1; j <= x2; j++)
			s += p_Data[i * W + j];
	return s;
}

double Mat::get(int y, int x)
{
	if (y < 0 || x < 0 || y >= H || x >= W)
		throw std::exception("索引越界");
	return p_Data[y * W + x];
}

double* Mat::sumInRow()
{
	double* s = new double[W];
	for(int j = 0; j < W; j++) {
		s[j] = 0;
		for (int i = 0; i < H; i++)
			s[j] += p_Data[i * W + j];
	}
	return s;
}

double* Mat::sumInCol()
{
	double* s = new double[H];
	for (int i = 0; i < H; i++) {
		s[i] = 0;
		for (int j = 0; j < W; j++)
			s[i] += p_Data[i * W + j];
	}
	return s;
}

Mat& Mat::operator=(const Mat& m)
{
	if (this != &m)
		load(m);
	return *this;
}