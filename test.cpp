#include"MatTool.h"

int main() {
	Mat m(3, 4);
	MatTool::print(m);
	double data[3][3] = { {1,4,6}, {2,3,6},{5,7,9} };
	m.load((double*)data, 3, 3);
	MatTool::print(m);
	MatTool::print(m.inverse());
	return 0;
}