/**
 * filename: edgeListTest.cpp
 * contents: this file contains a simple test
 *      for Kronecker edge list creation.
 *
 * The scalable data generator will construct a list of edge tuples
 *      containing vertex identifiers. Each edge is undirected with
 *      its endpoints given in the tuple as StartVertex and EndVertex.
 */

#include "edgeList.h"

#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
	// get scale and edge factor
    int scale = atoi(argv[1]);
    int edgeFactor = atoi(argv[2]);

    // compute num nodes & edges
    int numNodes = pow(2, scale);
    int numEdges = edgeFactor * numNodes;

    // allocate buffer 
    EdgeList edges(numEdges);

    // create edges
	edges.create(numNodes, scale);

	// prints Kronecker generated graph start and end vertices
    cout << "List of Kronecker edges:" << endl;
    cout << "Start End" << endl;
    for (int i = 0; i < numEdges; i++) {
    	cout << edges.edges()[i] << " " << edges.edges()[i + edges.size()] << endl;
    }

	return 0;
}