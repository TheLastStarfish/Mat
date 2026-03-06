#include "Mat.h"
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>

Mat::Mat()
{
	p_Date = nullptr;
	H = 0;
	W = 0;
}

Mat::Mat(double* pDate, int h, int w) :p_Date(nullptr), H(0), W(0)
{
	load(pDate, h, w);
}

Mat::Mat(double** ppDate, int h, int w) :p_Date(nullptr), H(0), W(0)
{
	load(ppDate, h, w);
}

Mat::Mat(const Mat& m) :p_Date(nullptr), H(0), W(0)
{
	load(m);
}

Mat::Mat(int h, int w) :p_Date(nullptr), H(0), W(0)
{
	create(h, w);
}

Mat::Mat(const Mat& m, int y1, int x1, int y2, int x2) :p_Date(nullptr), H(0), W(0)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0
		|| y1 >= m.H || x1 >= m.W || y2 >= m.H || x2 >= m.W)
		throw std::runtime_error("Submatrix index out of bounds");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);

	int h = y2 - y1 + 1;
	int w = x2 - x1 + 1;
	create(h, w);

	for (int i = 0; i < h; ++i)
		memcpy(p_Date + i * w, m.p_Date + (y1 + i) * m.W + x1, sizeof(double) * w);
}

Mat::~Mat()
{
	distroy();
}

void Mat::create(int h, int w)
{
	if(h == 0 || w == 0)
		throw std::runtime_error("Dimensions cannot be zero");
	if (p_Date != nullptr)
		distroy();
	p_Date = new double[h * w] {};
	H = h;
	W = w;
}

void Mat::distroy()
{
	if (p_Date != nullptr)
		delete[] p_Date;
	p_Date = nullptr;
	H = 0;
	W = 0;
}

void Mat::load(double* pDate, int h, int w)
{
	create(h, w);
	memcpy(p_Date, pDate, sizeof(double) * h * w);
}

void Mat::load(double** ppDate, int h, int w)
{
	create(h, w);
	for (int i = 0; i < h; i++)
		memcpy(p_Date + i * w, ppDate[i], sizeof(double) * w);
}

void Mat::load(const Mat& m)
{
	if (this == &m) return;
	create(m.H, m.W);
	memcpy(p_Date, m.p_Date, sizeof(double) * H * W);
}

void Mat::setCol(double* pDate, int x)
{
	if (x < 0 || x >= W)
		throw std::runtime_error("Column index out of bounds");
	for (int i = 0; i < H; i++)
		p_Date[i * W + x] = pDate[i];
}

void Mat::setRow(double* pDate, int y)
{
	if (y < 0 || y >= H)
		throw std::runtime_error("Row index out of bounds");
	memcpy(p_Date + y * W, pDate, sizeof(double) * W);
}

void Mat::copyTo(Mat& m)
{
	m.load(*this);
}

void Mat::copyTo(Mat& m, int y1, int x1, int y2, int x2)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0 
		|| y1 >= H || x1 >= W || y2 >= H || x2 >= W)
		throw std::runtime_error("Submatrix index out of bounds");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);
	int h = y2 - y1 + 1;
	int w = x2 - x1 + 1;
	m.create(h, w);
	for (int i = 0; i < h; ++i)
		memcpy(m.p_Date + i * w, p_Date + (y1 + i) * W + x1, sizeof(double) * w);
}

void Mat::set(double data, int y, int x)
{
	if (y < 0 || x < 0 || y >= H || x >= W)
		throw std::runtime_error("Index out of bounds");
	p_Date[y * W + x] = data;
}

bool Mat::isAllocated()
{
	return p_Date != nullptr;
}

bool Mat::same(const Mat& m)
{
	if (H != m.H || W != m.W) return false;
	return memcmp(p_Date, m.p_Date, sizeof(double) * H * W) == 0;
}

double* Mat::convertTo()
{
	return p_Date;
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
		throw std::runtime_error("Column index out of bounds");
	Mat m(H, 1);
	for (int i = 0; i < H; i++)
		m.p_Date[i] = p_Date[i * W + x];
	return m;
}

Mat Mat::getRow(int y)
{
	if (y < 0 || y >= H)
		throw std::runtime_error("Row index out of bounds");
	Mat m(1, W);
	memcpy(m.p_Date, p_Date + y * W, sizeof(double) * W);
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
		m.p_Date[i] = p_Date[i * W + i];
	return m;
}

Mat Mat::eye(int n)
{
	Mat m(n, n);
	for (int i = 0; i < n; i++)
		m.p_Date[i * n + i] = 1;
	return m;
}

Mat Mat::ones(int h, int w)
{
	Mat m(h, w);
	std::fill_n(m.p_Date, h * w, 1);
	return m;
}

Mat Mat::transpose()
{
	Mat m(W, H);
	for(int i = 0; i < H; i++)
		for (int j = 0; j < W; j++)
			m.p_Date[j * H + i] = p_Date[i * W + j];
	return m;
}

Mat Mat::plus(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::runtime_error("Matrix dimensions do not match");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Date[i] = p_Date[i] + m.p_Date[i];
	return r;
}

Mat Mat::minus(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::runtime_error("Matrix dimensions do not match");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Date[i] = p_Date[i] - m.p_Date[i];
	return r;
}

Mat Mat::mult(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::runtime_error("Matrix dimensions do not match");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Date[i] = p_Date[i] * m.p_Date[i];
	return r;
}

Mat Mat::div(const Mat& m)
{
	if (H != m.H || W != m.W)
		throw std::runtime_error("Matrix dimensions do not match");
	Mat r(H, W);
	for (int i = 0; i < H * W; i++)
		r.p_Date[i] = p_Date[i] / m.p_Date[i];
	return r;
}

Mat Mat::matchInRow(const Mat m)
{
	if (H != m.H)
		throw std::runtime_error("Row counts do not match");
	Mat r(H, W + m.W);
	for (int i = 0; i < H; i++) {
		memcpy(r.p_Date + i * r.W, p_Date + i * W, sizeof(double) * W);
		memcpy(r.p_Date + i * r.W + W, m.p_Date + i * m.W, sizeof(double) * m.W);
	}
	return r;
}

Mat Mat::matchInCol(const Mat m)
{
	if (W != m.W)
		throw std::runtime_error("Column counts do not match");
	Mat r(H + m.H, W);
	for(int i = 0; i < H; i++)
		memcpy(r.p_Date + i * W, p_Date + i * W, sizeof(double) * W);
	for(int i = 0; i < m.H; i++)
		memcpy(r.p_Date + (H + i) * W, m.p_Date + i * W, sizeof(double) * W);
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
		s += p_Date[i];
	return s;
}

double Mat::sum(int y1, int x1, int y2, int x2)
{
	if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0
		|| y1 >= H || x1 >= W || y2 >= H || x2 >= W)
		throw std::runtime_error("Submatrix index out of bounds");
	if (y1 > y2) std::swap(y1, y2);
	if (x1 > x2) std::swap(x1, x2);
	double s = 0;
	for (int i = y1; i <= y2; i++)
		for (int j = x1; j <= x2; j++)
			s += p_Date[i * W + j];
	return s;
}

double Mat::get(int y, int x)
{
	if (y < 0 || x < 0 || y >= H || x >= W)
		throw std::runtime_error("Index out of bounds");
	return p_Date[y * W + x];
}

// Returns an array of H values: s[i] is the sum of all elements in row i.
// Caller must delete[] the returned array.
double* Mat::sumInRow()
{
	double* s = new double[H];
	for (int i = 0; i < H; i++) {
		s[i] = 0;
		for (int j = 0; j < W; j++)
			s[i] += p_Date[i * W + j];
	}
	return s;
}

// Returns an array of W values: s[j] is the sum of all elements in column j.
// Caller must delete[] the returned array.
double* Mat::sumInCol()
{
	double* s = new double[W];
	for(int j = 0; j < W; j++) {
		s[j] = 0;
		for (int i = 0; i < H; i++)
			s[j] += p_Date[i * W + j];
	}
	return s;
}

Mat& Mat::operator=(const Mat& m)
{
	if (this != &m)
		load(m);
	return *this;
}