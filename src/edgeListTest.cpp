/**
 * filename: edgeListTest.cpp
 * contents: this file contains a simple test
 *      for Kronecker edge list creation.
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

	// prints Kronecker generated graph from and to vertices
    edge *edgeList = edges.edges();
    for (int i = 0; i < numEdges; i++) {
    	cout << edgeList[i].from << " " << edgeList[i].to << endl;
    }

	return 0;
}