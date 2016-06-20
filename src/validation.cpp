#include "validation.h"
#include <vector>
#include <algorithm>

/**
 * Simple implementation of Graph to validate parent_arrays against.
 */
class Graph
{
public:
	Graph();

	bool has_edge(VertexId v1, VertexId v2)
	{
		return false;
	}

	std::vector<VertexId> connected_component(VertexId v1)
	{
		std::vector<VertexId> component;
		return component;
	}

	std::vector<VertexId> graph_levels(VertexId root_id)
	{
		std::vector<VertexId> levels(VERTICIES, -1);
		return levels;	
	}
};

// Ensures there are no cycles in the parent_array
RetType validate_cycles(VertexId* parent_array, VertexId root_id)
{
	VertexId vertex_id;
	std::vector<bool> visited(VERTICIES);

	// Cycle check
	for (int i = 0; i < VERTICIES; i++)
	{
		// If a node was already visited, it was part of another node's path
		// which was not cyclic
		if (visited[i]) continue;

		std::vector<bool> vertex_bits(VERTICIES); //< Tracks which verticies
		                                          // are visited when
		                                          // traversing path i
		vertex_id = i; //< Tracks which vertex we are on in traversal
		do
		{
			//visiting a vertex twice in one path traversal implies a cycle
			if (vertex_bits[vertex_id])
			{
				return ERR_HAS_CYCLE;
			}

			// Track traversal and iterate
			vertex_bits[vertex_id] = visited[vertex_id] = True;
			vertex_id = parent_array[vertex_id];
		}
		//Do until we hit root
		while (vertex_id != root_id);
	}
}

// Checks that levels implied in parent_array are correct
RetType validate_levels(VertexId* parent_array, VertexId root_id, Graph graph)
{
	//Level check
	std::vector<int> expected_levels(VERTICIES, -1); //< Stores the levels implied by parent_array
	expected_levels[root_id] = 0; //< root_id level always is zero
	std::deque<unsigned int> to_visit; 

	//Inital population of to_visit
	for (int i = 0; i < VERTICIES; i++)
	{
		//Level 1 can be assigned to here
		if (parent_array[i] == root_id)
			expected_levels[i] = 1;
		// If parent_array[i] < 0, not part of bfs tree
		else if (parent_array[i] >= 0)
			to_visit.push_back(i);
	}

	// Brute force is best force
	while (!to_visit.empty())
	{
		int id = to_visit.front();
		if (expected_levels[parent_array[id]] >= 0)
		{
			expected_levels[id] = expected_levels[parent_array[id]] + 1;
		}
	}

	std::vector<int> graph_levels = graph.bfs_levels(root_id);

	for(int i = 0; i < VERTICIES; i++)
	{
		if (expected_levels[i] != graph_levels[i])
		{
			return ERR_INVALID_LEVEL;
		}
	}

	return VALIDATION_SUCCESS;
}


// Checks that all edges in an input list differ by, at most, one level
RetType validate_graph(VertexId* parent_array, )


// Checks that parent_array traverses entire connected component
RetType validate_span(VertexId* parent_array, VertexId root_id, Graph graph)
{
	//Gets all vertex_ids in the connected component
	std::vector<int> vertices = graph.connected_component(root_id);

	//Iterate through connected component vertecies
	for (auto it = vertices.begin(); it != vertices.end(); it++)
		//root_id is taken to always be visited, other -1's imply not in connected component
		if (!(*it == root_id or parent_array[*it] >= 0))
			return ERR_DOESNT_SPAN;

	return VALIDATION_SUCCESS;

}

//Checks that all edges in parent_array exist in graph
RetType validate_edges(unsigned int* parent_array, Graph graph)
{
	for (int i = 0; i < VERTICIES; i++)
	{
		// -1 denotes no edge exists
		if (parent_array[i] >= 0 && !graph.edge_exists(i, parent_array[i]))
				return ERR_UNREAL_EDGE;
	}
	return VALIDATION_SUCCESS;
}

// Ors together all validation flags to return 
RetType validate(VertexId* parent_array, VertexId root_id, EdgeList edges)
{
	return validate_cycles(parent_array, root_id)
	     | validate_levels(parent_array, root_id, Graph graph)
	     | validate_graph(parent_array, edge_list)
	     | validate_span(parent_array, root_id, graph)
	     | validate_edges(parent_array, graph)
}