#include "nodes.h"
#include "utility.h"

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace utils;

ROOT::ROOT(){}

ROOT::ROOT(vector<double> edge_weights,
		vector<MultiNode> children, vector<int> maxind,
		int leafstart, double edgesum, vector<double> orig_edge_weights,
		double orig_edgesum):edge_weights(edge_weights),children(children),
				maxind(maxind),
				leafstart(leafstart), edgesum(edgesum), orig_edge_weights(orig_edge_weights),
				orig_edgesum(orig_edgesum){}

int ROOT::num_leaves(){
	int n = 0;
	for(int i = 0; i < children.size(); i++)
		n += children[i].num_leaves();
	return n+edge_weights.size()-children.size();
}

void ROOT::sample_node(){
	vector<MultiNode> multi = get_multinodes();
	for(int mi = 0; mi < multi.size(); mi++){
		MultiNode* mu = &children[mi];
		vector<double> vals;
		int numvar = mu->num_variants();

		for(int vi = 0; vi < numvar; vi++){
			double v = mu->logphi_update(vi) + mu->var_logweight(vi);
			vals.push_back(v);
		}
		int y = log_mult_sample(vals);
		mu->y = y;
	}
}

void ROOT::leaf_count_update(double val, int leaf){
	for(int i = 0 ; i < children.size(); i++){
		if (leaf <= maxind[i]){
			edge_weights[i] += val;
			edgesum += val;
			children[i].leaf_count_update(val, leaf);
			return;
		}
	}
	int ei = children.size() + leaf - leafstart;
	edge_weights[ei] += val;
	edgesum += val;
	return;
}

vector<MultiNode> ROOT::get_multinodes(){
	vector<MultiNode> multinodes;
	for(int i = 0; i < children.size(); i++){
		if( string( typeid(children[i]).name() ).find("MultiNode") != string::npos ){
			multinodes.push_back(children[i]);
		}
	}
	return multinodes;
}


double ROOT::wordval_update(double val, int leaf){
	double newval;
	for(int i =0; i < children.size(); i++){
		if(leaf <= maxind[i]){
			newval = edge_weights[i] / edgesum;

			return children[i].wordval_update(newval*val, leaf);
		}
	}
	int ei = children.size() + leaf - leafstart;
	newval = edge_weights[ei] / edgesum;

	return val * newval;
}

double ROOT::logphi_update(){
	double logpwz = lgamma(orig_edgesum) -lgamma(edgesum);

	for(int ei =0; ei < edge_weights.size(); ei++)
		logpwz += lgamma(edge_weights[ei]) - lgamma(orig_edge_weights[ei]);

	for(int i = 0; i < children.size(); i++){
		logpwz += children[i].logphi_update();
	}
	return logpwz;
}

Node::Node(vector<double> edge_weights,
		vector<Node> children, vector<int> maxind,
		int leafstart, double edgesum, vector<double> orig_edge_weights,
		double orig_edgesum):edge_weights(edge_weights),children(children),
				maxind(maxind),
				leafstart(leafstart), edgesum(edgesum), orig_edge_weights(orig_edge_weights),
				orig_edgesum(orig_edgesum){}

Node::Node(vector<double> edge_weights, vector<Node> children,
			vector<int> maxind, int leafstart, vector<int> words,
			double edgesum, vector<double> orig_edge_weights,
			double orig_edgesum):edge_weights(edge_weights),children(children),
					maxind(maxind),leafstart(leafstart),words(words),
					edgesum(edgesum), orig_edge_weights(orig_edge_weights),
					orig_edgesum(orig_edgesum){}

int Node::num_leaves(){
	int n = 0;
	for(int i = 0; i < children.size(); i++)
		n += children[i].num_leaves();
	return n+edge_weights.size()-children.size();
}



void Node::leaf_count_update(double val, int leaf){
	for(int i = 0 ; i < children.size(); i++){
		if (leaf <= maxind[i]){
			edge_weights[i] += val;
			edgesum += val;
			children[i].leaf_count_update(val, leaf);
			return;
		}
	}
	int ei = children.size() + leaf - leafstart;
	edge_weights[ei] += val;
	edgesum += val;
	return;
}


double Node::wordval_update(double val, int leaf){
	double newval;
	for(int i =0; i < children.size(); i++){
		if(leaf <= maxind[i]){
			newval = edge_weights[i] / edgesum;

			return children[i].wordval_update(newval*val, leaf);
		}
	}
	int ei = children.size() + leaf - leafstart;
	newval = edge_weights[ei] / edgesum;

	return val * newval;
}

double Node::logphi_update(){
	double logpwz = lgamma(orig_edgesum) -lgamma(edgesum);

	for(int ei =0; ei < edge_weights.size(); ei++)
		logpwz += lgamma(edge_weights[ei]) - lgamma(orig_edge_weights[ei]);

	for(int i = 0; i < children.size(); i++){
		logpwz += children[i].logphi_update();
	}
	return logpwz;
}

MultiNode::MultiNode(vector<double> edge_weights, vector<Node> children, vector<int> maxind,
		int leafstart, double edgesum, vector<double> orig_edge_weights, double orig_edgesum):edge_weights(edge_weights),children(children),maxind(maxind),
				leafstart(leafstart), edgesum(edgesum), orig_edge_weights(orig_edge_weights),
				orig_edgesum(orig_edgesum){}

MultiNode::MultiNode(vector<double> edge_weights, vector<Node> children,
			vector<int> maxind, int leafstart, vector<int> words,
			vector<Node> variants, vector<vector<int>> fake_leafmap,
			vector<double> variant_logweights):edge_weights(edge_weights),children(children),
					maxind(maxind),leafstart(leafstart),words(words),
					variants(variants), fake_leafmap(fake_leafmap),
					variant_logweights(variant_logweights){
		y= -1;
}

double MultiNode::logphi_update(){
	double logpwz = lgamma(orig_edgesum) -lgamma(edgesum);

	for(int ei =0; ei < edge_weights.size(); ei++)
		logpwz += lgamma(edge_weights[ei]) - lgamma(orig_edge_weights[ei]);

	for(int i = 0; i < children.size(); i++){
		logpwz += children[i].logphi_update();
	}
	return logpwz;
}

int MultiNode::num_leaves(){
	int n =0;
	for(int i = 0; i < children.size(); i++)
		n += children[i].num_leaves();
	return n+words.size();
}


void MultiNode::leaf_count_update(double val, int leaf){
	for(int i = 0; i < children.size(); i++){

		if (leaf <= maxind[i]){
			for(int v = 0; v < variants.size(); v++){
				variants[v].leaf_count_update(val, fake_leafmap[v][i]);
			}
			children[i].leaf_count_update(val, leaf);
			return;
		}
	}
	int ei = children.size() + leaf - leafstart;
	for(int v = 0; v < variants.size(); v++)
		variants[v].leaf_count_update(val, fake_leafmap[v][ei]);
	return;
}

double MultiNode::wordval_update(double val, int leaf){
	double newval;
	for(int i = 0; i < children.size(); i++){

		if (leaf <= maxind[i]){
			newval = variants[y].wordval_update(val, fake_leafmap[y][i]);
			return children[i].wordval_update(newval, leaf);
		}
	}

	int ei = children.size() + leaf - leafstart;
	return variants[y].wordval_update(val, fake_leafmap[y][ei]);
}

int MultiNode::num_variants(){
	return variants.size();
}
double MultiNode::var_logweight(int given_y){
	return variant_logweights[given_y];
}

double MultiNode::logphi_update(int given_y){
	return variants[given_y].logphi_update();
}




