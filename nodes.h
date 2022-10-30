#ifndef NODES_H_
#define NODES_H_

#include <iostream>
#include <vector>
using namespace std;

class MultiNode;

class ROOT{
public :
	vector<double> edge_weights;
	vector<MultiNode> children;
	vector<int> maxind;
	int leafstart;
	vector<double> orig_edge_weights;
	double edgesum;
	double orig_edgesum;

	ROOT();

	ROOT(vector<double> edge_weights, vector<MultiNode> children, vector<int> maxind,
			int leafstart, double edgesum, vector<double> orig_edge_weights, double orig_edgesum);


	int num_leaves();

	void sample_node();
	void leaf_count_update(double val, int leaf);
	double wordval_update(double val, int leaf);
	double logphi_update();
	vector<MultiNode> get_multinodes();
};

class Node { //represent a node in dirichlet tree
public:

	vector<double> edge_weights;

	vector<Node> children;

	vector<int> maxind;
	int leafstart;
	vector<double> orig_edge_weights;
	double edgesum;
	double orig_edgesum;

	Node(vector<double> edge_weights, vector<Node> children,
			vector<int> maxind, int leafstart, double edgesum,
			vector<double> orig_edge_weights, double orig_edgesum);

	vector<int> words;
	Node(vector<double> edge_weights, vector<Node> children, vector<int> maxind,
					int leafstart, vector<int> words,
					double edgesum, vector<double> orig_edge_weights, double orig_edgesum);

	int num_leaves();

	void sample_node();
	void leaf_count_update(double val, int leaf);
	double wordval_update(double val, int leaf);
	double logphi_update();
};

class MultiNode{ //represent an intermediate node
public:
	vector<double> edge_weights;
	vector<int> maxind;
	int leafstart;
	vector<Node> children;
	vector<double> orig_edge_weights;
	double edgesum;
	double orig_edgesum;

	vector<int> words;
	vector<Node> variants;
	vector<double> variant_logweights;
	vector<vector<int>> fake_leafmap;
	int y;

	MultiNode(vector<double> edge_weights, vector<Node> children, vector<int> maxind,
				int leafstart, double edgesum, vector<double> orig_edge_weights, double orig_edgesum);
		MultiNode(vector<double> edge_weights, vector<Node> children, vector<int> maxind,
				int leafstart, vector<int> words, vector<Node> variants, vector<vector<int>> fake_leafmap,
				vector<double> variant_logweights);

	void leaf_count_update(double val, int leaf);
	double wordval_update(double val, int leaf);
	int num_variants();
	double var_logweight(int given_y);
	double logphi_update(int given_y);
	int num_leaves();
	double logphi_update();

};



#endif /* NODES_H_ */
