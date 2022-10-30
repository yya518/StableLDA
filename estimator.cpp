#include "estimator.h"
#include "nodes.h"
#include "utility.h"

#include<iostream>
#include<cmath>
#include <string>

#include <fstream>
#include<sstream>
#include<cstdlib>
#include <numeric>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <algorithm>

using namespace std;
using namespace utils;



Estimator::Estimator(double alpha, double beta, double eta,
		int num_topics, int num_words, int rand_seed):alpha(alpha),beta(beta),
				eta(eta),num_topics(num_topics),num_words(num_words),
				rand_seed(rand_seed){
		srand(rand_seed);

}

void Estimator::readin_vocab(string vocab_file){
	ifstream file(vocab_file);
	if(file.fail()){
		cerr<< "vocab file does not exist" <<endl;
		exit(1);
	} else{
		string line;
		int count = 0;
		while(getline(file, line)){
			vocab.push_back(line);
			vocab2id.insert(pair<string,int>(line, count));
			count += 1;
		}
	}
}

void Estimator::readin_data(string data_file){

	ifstream file(data_file);

	if(file.fail()){
		cerr<< "data file does not exist" <<endl;
		exit(1);
	} else
	{
		string line;
		std::map<string,int> vocab_iter;
		while(getline(file, line)){
			vector<int> temp_doc;
			stringstream linestream(line);
			string word;
			while(linestream >> word){
				//vocab_iter = vocab2id.find(word);
				int tok =  vocab2id.find(word)->second;
				temp_doc.push_back(tok);
			}
				//temp_doc.push_back(stoi(word));
				//temp_doc.push_back(vocab2id.   .(word));
			doc_lens.push_back(temp_doc.size());
			docs.push_back(temp_doc);
		}
		num_docs = docs.size();
		file.close();
		//cout<<"readin_data methods. number of docs: " << docs.size() <<endl;
	}
}

void Estimator::readin_clusters(string cluster_file){
	ifstream file(cluster_file);
	if(file.fail()){
		cerr<< "data file does not exist" <<endl;
		exit(1);
	}else{
		string line;
		int wordcount = 0;
		while(getline(file, line)){ //each line is a mustlink
			vector<int> temp;
			stringstream linestream(line);
			string token;
			while(getline(linestream, token, ','))
				temp.push_back( vocab2id[token]);
			ml_cliques.push_back(temp);
			wordcount += temp.size();
		}
		assert(wordcount == num_words);
		int num_cliques = ml_cliques.size(); //each ml-link is a clique
		vector<int> temp;
		for(int i = 0; i < num_cliques; i++)
			temp.push_back(i+wordcount);
		cl_cliques.push_back(temp);

	}
}

void Estimator::build_tree(){
	//given ml_clique, cl_clique, beta, W, eta, create a dirichlet tree
	vector<Node> ml_nodes;

	//build MLnodes for each ml clique will only have leaf children
	for(int i = 0; i < ml_cliques.size(); i++){//for each ml clique
		double edgesum = 0;
		vector<double> edge_weights;
		vector<double> orig_edge_weights;
		for(int j = 0; j < ml_cliques[i].size(); j++){
			edge_weights.push_back(eta * beta);
			orig_edge_weights.push_back(eta * beta);
			edgesum += eta * beta;
		}
		vector<Node> children;
		vector<int> maxind;
		Node ml(edge_weights, children, maxind,0, ml_cliques[i], edgesum, orig_edge_weights, edgesum);
		ml_nodes.push_back(ml);
	}

	//build multinodes for each cl_clique
	vector<MultiNode> multinodes;
	for(int i = 0 ; i < cl_cliques.size(); i++){ //each ml-clique is cannot-link with other ml-cliques
		vector<int> icids;
		vector<int> fake_words;
		for(int z= 0; z < cl_cliques[i].size(); z++){
			int key = cl_cliques[i][z];
			if (key >= num_words)  //those intermediate nodes must have index greater than num_words
				icids.push_back(key);
			else
				cerr<< "intermediate nodes id error" << endl;
		}

		for(int z = 0; z < icids.size(); z++)
			fake_words.push_back(icids[z]);

		vector<Node> variations;
		vector<double> variant_logweights;
		vector<vector<int>> fake_leafmap;


		vector<vector<int>> temp_allow;
		for(int i = 0; i < ml_cliques.size(); i++){
				vector<int> temp;
				temp.push_back(i+num_words);
				temp_allow.push_back(temp);
		}

		for(int j = 0; j < temp_allow.size(); j++){
			vector<int> good = temp_allow[j];
			vector<int> bad;
			for(int z =0 ; z < fake_words.size(); z++){
				if( find(good.begin(), good.end(), fake_words[z]) == good.end() )
					bad.push_back(fake_words[z]);
			}

			vector<double> aedges;
			vector<double> orig_aedges;
			for(int z =0; z < good.size(); z++){
				if(good[z] >= num_words){
					aedges.push_back( beta * ml_nodes[good[z]-num_words].num_leaves());
					orig_aedges.push_back(beta * ml_nodes[good[z]-num_words].num_leaves());
				}else{
					aedges.push_back(beta);
					orig_aedges.push_back(beta);
				}
			}
			vector<double> fedges;
			vector<double> orig_fedges;
			double aedgesum = 0;
			for(int z = 0; z<aedges.size(); z++)
				aedgesum += aedges[z];
			fedges.push_back(eta * aedgesum);
			orig_fedges.push_back(eta * aedgesum);

			for(int z= 0 ; z < bad.size(); z++){
				if(bad[z] >= num_words){
					fedges.push_back( beta * ml_nodes[bad[z]-num_words].num_leaves());
					orig_fedges.push_back( beta * ml_nodes[bad[z]-num_words].num_leaves());
				}else{
					fedges.push_back(beta);
					orig_fedges.push_back(beta);
				}
			}
			vector<Node> children;
			vector<int> maxind;

			vector<int> temp_words;
			vector<Node> temp_variants;
			vector<vector<int>> temp_fake_leafmap;
			vector<double> logweights;
			Node likely_internal(aedges, children, maxind, 0, aedgesum, orig_aedges, aedgesum);

			vector<Node> likely_internal_list;
			likely_internal_list.push_back(likely_internal);
			vector<int> maxindN;
			maxindN.push_back(good.size()-1);
			double fedgesum = 0;
			for(int z = 0; z < fedges.size(); z++)
				fedgesum += fedges[z];

			Node fakeroot(fedges, likely_internal_list, maxindN, good.size(), fedgesum, orig_fedges, fedgesum);

			vector<int> fake_wordmap;
			for(int z = 0; z < good.size(); z++)
				fake_wordmap.push_back(good[z]);
			for(int z=0; z < bad.size(); z++)
				fake_wordmap.push_back(bad[z]);

			vector<int> fake_leaf;
			for(int z= 0; z < fake_words.size(); z++){
				int wi = fake_words[z];
				int index = getIndex(fake_wordmap, wi);
				fake_leaf.push_back(index);
			}

			variations.push_back(fakeroot);
			fake_leafmap.push_back(fake_leaf);
			variant_logweights.push_back(log(aedgesum));
		}

		vector<Node> ichildrenM;
		vector<int> lchildren;
		for(int z = 0; z < cl_cliques[i].size(); z++){
			int key = cl_cliques[i][z];
			if (key >= num_words)
				ichildrenM.push_back(ml_nodes[key-num_words]);
			else
				lchildren.push_back(key);
		}

		vector<double> edge_weights;
		vector<int> maxind;

		MultiNode multi(edge_weights,ichildrenM,maxind,0,lchildren,variations, fake_leafmap, variant_logweights);
		multinodes.push_back(multi);
	}
	int cur_ind = 0;
	vector<int> wordmap;
	double edgesum = 0;

	root = ROOT();
	for(int i= 0; i < multinodes.size(); i++){
		MultiNode* multi = &(multinodes[i]);
		for(int j = 0; j < multi->children.size(); j++){
			Node* ml_child = &(multi->children[j]);
			vector<int> words = ml_child->words;
			for(int w = 0; w < words.size(); w++)
				wordmap.push_back(words[w]);
			ml_child->leafstart = cur_ind;
			cur_ind += ml_child->words.size();
			multi->maxind.push_back(cur_ind-1);
		}
		if(multi->words.size() > 0 ){
			multi->leafstart = cur_ind;
			cur_ind += multi->words.size();
			for(int w =0 ; w < multi->words.size(); w++)
				wordmap.push_back(multi->words[w]);
		}
		edgesum += beta * multi->num_leaves();
		root.edge_weights.push_back(beta * multi->num_leaves());
		root.orig_edge_weights.push_back(beta * multi->num_leaves());
		root.children.push_back(*multi);
		root.maxind.push_back(cur_ind -1);
	}


	for(int i = 0 ; i < num_words-cur_ind; i++){
		edgesum += beta;
		root.edge_weights.push_back(beta);
		root.orig_edge_weights.push_back(beta);
	}
	root.leafstart = cur_ind;

	root.edgesum = edgesum;
	root.orig_edgesum = edgesum;

	for(int wi = 0 ; wi < num_words; wi++)
		if( find(wordmap.begin(), wordmap.end(), wi) == wordmap.end() )
			wordmap.push_back(wi);

	for(int wi = 0; wi < num_words; wi++)
		leafmap.push_back( getIndex(wordmap, wi) );
}

void Estimator::estimate(int epochs){

	//sampling
	for(int epoch = 0; epoch < epochs; epoch++){ //for each epoch
		//cout<<"running epoch " <<epoch <<endl;
		for(int di = 0; di < num_docs; di++){
			for(int wi = 0; wi < doc_lens[di]; wi++){
				int z = samples[di][wi];
				int word=  docs[di][wi];
				topics[z].leaf_count_update(-1, leafmap[word]);
				nd[di][z]--;

				vector<double> probs(num_topics, 0.0);
				double probs_sum = 0.0;
				for(int ti = 0; ti < num_topics; ti++){
					double wordterm = topics[ti].wordval_update(1, leafmap[word]);
					probs[ti] = wordterm * (nd[di][ti]+alpha);
					probs_sum += probs[ti];
				}
				int newz = mult_sample(probs, probs_sum);
				samples[di][wi] = newz;
				nd[di][newz]++;
				topics[newz].leaf_count_update(1, leafmap[word]);
			}
		}

	}
	//cout << "sampling is over "<<endl;

	//cout<< "calc_theta" <<endl;
	calc_theta();
	//cout<< "calc_phi" <<endl;
	calc_phi();
	print_topwords();
}
void Estimator::load_data(string data_file, string z_file, string cluster_file, string vocab_file){

	//1. read in data
	readin_vocab(vocab_file); //vocab, vocab2id

	readin_data(data_file); //num_docs, docs, doc_lens

	//2. read in topical clusters
	readin_clusters(cluster_file); //ml_clique, cl_clique

	//3. create tree for each topic
	build_tree();  //root

	//4. initialize counts
	// Build Dirichlet Tree for each topic
	vector<double> cedges = root.edge_weights;
	vector<MultiNode> cchildren = root.children;
	vector<int> cmaxind = root.maxind;
	int cleafstart = root.leafstart;
	double edgesum = root.edgesum;
	double orig_edgesum = root.orig_edgesum;

	vector<double> orig_edge_weights = root.orig_edge_weights;
	for(int ti = 0; ti < num_topics; ti++){
		ROOT newtopic(cedges, cchildren,
				cmaxind, cleafstart, edgesum, orig_edge_weights, orig_edgesum);
		topics.push_back(newtopic);
		topics[ti].sample_node();

	}
	vector<int> temp(num_topics,0);
	nd.assign(num_docs, temp);

	vector<double> temp1(num_topics,0);
	theta.assign(num_docs, temp1);
	vector<double> temp2(num_words,0);
	phi.assign(num_topics, temp2);

	for(int di = 0; di < num_docs; di++){
		vector<int> temp_sample(doc_lens[di], 0);
		samples.push_back(temp_sample);
	}

	ifstream zfile(z_file);
	if(zfile.fail()){
		//cout<< "z file does not exist, initialize randomly" <<endl;
		for(int di = 0; di < num_docs; di++){
			for(int wi = 0; wi < doc_lens[di]; wi++){
				int word = docs[di][wi];
				int new_z = rand() % num_topics;
				samples[di][wi] = new_z;
				nd[di][new_z] +=1;
				topics[new_z].leaf_count_update(1, leafmap[word]);
			}
		}
	} else{
		//cout<< "z file exists, initialize from z file" <<endl;
		string line;
		vector<vector<int>> temp_samples;
		while(getline(zfile, line)){
			vector<int> temp_z;
			stringstream linestream(line);
			string tok;
			while(linestream >> tok){
				temp_z.push_back(stoi(tok));
			}
			temp_samples.push_back(temp_z);
		}
		samples = temp_samples;

		//cout<<"num of documents " << num_docs<<endl;
		//cout<<"samples size " << samples.size()<<endl;
		assert(samples.size() == num_docs);


		for(int di = 0; di < num_docs; di++){
			for(int wi = 0; wi < doc_lens[di]; wi++){
				int word = docs[di][wi];
				int new_z = samples[di][wi];
				nd[di][new_z] +=1;
				topics[new_z].leaf_count_update(1, leafmap[word]);
			}
		}

	}
	zfile.close();
}



void Estimator::calc_theta(){
	for(int di = 0; di < num_docs; di++){
		vector<double> probs;
		double sum_prob = 0.0;
		for(int ti = 0; ti < num_topics; ti++){
			double p = nd[di][ti] + alpha;
			sum_prob += p;
			probs.push_back(p);
		}
		normalize(probs, sum_prob);

		for(int ti = 0; ti < num_topics; ti++)
			theta[di][ti] = probs[ti];

	}
}

void Estimator::calc_phi(){
	for(int ti = 0; ti < num_topics; ti++){
		vector<double> probs;
		double sum_prob = 0.0;
		for(int wi = 0; wi < num_words; wi++){
			phi[ti][wi] = topics[ti].wordval_update(1, leafmap[wi]);
			sum_prob += phi[ti][wi];
		}
	}
}

void Estimator::print_topwords(int N){
	calc_phi();

	if(num_words < N)
		N = num_words;

	for(int ti = 0; ti < num_topics; ti++){
		cout<< "Topic " << ti << ": ";
		vector<int> idx = sort_indexes(phi[ti]);

		for(int n = 0; n < N; n++){
			//cout<< vocab[idx[n]] << " " << phi[ti][idx[n]] <<endl;
			cout<< vocab[idx[n]] << " " ;
		}
		cout <<endl;
	}
}

void Estimator::save(string output_path){
	//cout<< "saving parameters " <<endl;
	calc_theta();
	calc_phi();

	string theta_file = output_path + "theta.dat";
	string phi_file = output_path + "phi.dat";
	string sample_file = output_path + "z.final.dat";
	save_matrix(theta_file, theta);
	save_matrix(phi_file, phi);
	save_sample(sample_file, samples);
}

Estimator::~Estimator() {
	// TODO Auto-generated destructor stub
}

