#ifndef UTILITY_H_
#define UTILITY_H_

#include <iostream>
#include <vector>
using namespace std;

namespace utils{

	int log_mult_sample(vector<double> vals);

	int mult_sample(vector<double> vals, double norm_sum);

	void normalize(vector<double> &vals, double norm_sum);

	int getIndex(vector<int> v, int K);

	vector<int> sort_indexes(const vector<double> &v);

	void save_matrix(string filename, vector<std::vector<double> > mat);

	void save_sample(string filename, vector<vector<int>> samples);

};
#endif /* UTILITY_H_ */
