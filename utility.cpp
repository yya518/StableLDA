#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>

#include "utility.h"

namespace utils{

int log_mult_sample(vector<double> vals){
	double maxval =vals[0];
	for(int vi = 0; vi < vals.size(); vi++){
		if(vals[vi] > maxval)
			maxval = vals[vi];
	}
	vector<double> newvals;
	double normsum = 0.0;
	for(int vi = 0; vi < vals.size(); vi++){
		double t = exp(vals[vi]-maxval);
		newvals.push_back(t);
		normsum += t;
	}
	return mult_sample(newvals, normsum);
}

int mult_sample(vector<double> vals, double norm_sum){

	double r = rand() / double(RAND_MAX) * norm_sum;
	double tmp_sum = 0.0;
	int j = 0;
	while(tmp_sum < r || j == 0){
		tmp_sum += vals[j];
		j++;
	}
	return j-1;
}

int getIndex(vector<int> v, int K){
    auto it = find(v.begin(), v.end(), K);

    // If element was found
    if (it != v.end())
    {

        // calculating the index of K
        int index = it - v.begin();
        return index;
    }
    else {
        // If the element is not present in the vector
        return -1;
    }
}

void normalize(vector<double> &vals, double norm_sum){
	for(int i = 0; i < vals.size(); i++)
		vals[i] = vals[i] / norm_sum;
}

vector<int> sort_indexes(const vector<double> &v) {
  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

void save_matrix(string filename, vector<vector<double> > mat) {
  ofstream file(filename.c_str());
  int row = mat.size();
  int col = mat[0].size();
  for(int i = 0; i < row; ++i) {
    for(int j = 0; j < col; ++j) {
      file << mat[i][j] << " ";
    }
    file << endl;
  }
  file.close();
}

void save_sample(string filename, vector<vector<int>> samples) {
  ofstream file(filename.c_str());

  for(int i = 0; i < samples.size(); ++i) {
    for(int j = 0; j < samples[i].size(); ++j) {
      file << samples[i][j] << " ";
    }
    file << endl;
  }
  file.close();
}

};
