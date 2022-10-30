#ifndef STABLELDA_H_
#define STABLELDA_H_

#include <vector>
#include <iostream>
#include <string>
#include <map>
#include "nodes.h"
using namespace std;

class Estimator {
public:

	double alpha;
	double beta;
	double eta;
	int num_topics;
	int num_words;
	int rand_seed;
	int num_docs;
	vector<vector<int>> docs;
	vector<vector<int>> samples;
	vector<int> doc_lens;
	vector<vector<int> > topical_clusters;
	vector<vector<int>> mustlinks;
	vector<vector<int>> cannotlinks;
	ROOT root;
	vector<int> leafmap;
	vector<string> vocab;
	map<string, int> vocab2id;
	vector<ROOT> topics;
	vector<vector<int>> nd;

	vector<vector<double>> theta;
	vector<vector<double>> phi;

	Estimator(double alpha, double beta, double eta, int num_topics, int num_words, int rand_seed);

	void load_data(string data_file, string z_file, string cluster_file, string vocab_file);

	void estimate(int epochs);

	virtual ~Estimator();


	void print_topwords(int N=10);

	void save(string output_path);

private:

	vector<vector<int>> ml_cliques; //must-link connected components
	vector<vector<int>> cl_cliques; //cannot-link connected components

	void readin_data(string data_file);
	void readin_vocab(string vocab_file);
	void readin_clusters(string cluster_file);
	void build_tree();

	void calc_theta();
	void calc_phi();


};

#endif /* STABLELDA_H_ */

