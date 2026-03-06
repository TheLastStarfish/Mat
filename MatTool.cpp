#include "MatTool.h"
#include<iostream>

void MatTool::print(Mat m) {
	for (int i = 0; i < m.Rows(); i++) {
		for (int j = 0; j < m.Cols(); j++) {
			std::cout << m.get(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}