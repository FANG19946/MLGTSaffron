#ifndef A7F3C921_5D84_4B6E_9A12_E8C4F07B3D56
#define A7F3C921_5D84_4B6E_9A12_E8C4F07B3D56

/**
 * @brief Hierarchical Navigable Small World implementation.
 * 
 * Uses an angular range based stopping conditions and a hierarchical graph structure to perform approximate nearest neighbor search.
 */



class HNSW{
private:
    struct Node{
        uint max_layer;
        vector<uint> degrees;
        vector<uint> offsets;
        vector<uint> neighbors;
    };
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data_eigen_;


public:
    uint dimension_;
    double num_degrees_;
    uint num_features_;
    uint debug_;
    uint M_;
    uint M_max_;
    uint M0_max_;
    uint efConstruction_;
    uint ef_;
    uint mL_;
    uint ep_;
    uint max_layer_;
    bool extendCandidates_;
    bool keepPrunedConnections_;
    vector<float> dot_products_;
    vector<uint> reset_dot_products_;
    vector<bool> visited_;
    vector<uint> reset_visited_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
    vector<Node> hnsw_graph_;

    HNSW( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data, uint num_features, double num_degrees = 10, uint M = 16, uint efConstruction = 200, uint ef = 50, uint mL = 0, uint ep = 0, int debug = 0):
        data_eigen_(data),
        dimension_(data.cols()),
        num_degrees_(num_degrees),
        num_features_(num_features),
        debug_(debug),
        M_(M),
        M_max_(M * 2),
        M0_max_(M * 4),
        efConstruction_(efConstruction),
        ef_(ef),
        mL_(mL),
        rng_(42),
        dist_(0.0, 1.0),
        ep_(ep),
        max_layer_(0),
        extendCandidates_(false),
        keepPrunedConnections_(false)
    {
        // UPDATE: ep_ needs special handling for empty graph.
        hnsw_graph_.resize(num_features_);
        dot_products_.assign(num_features_, 2.0f);
        visited_.assign(num_features_, false);
        // Insert the points sequentially and build the hnsw_graph_
        for(uint i = 0; i < num_features_; ++i){
            insert(i);
        }


    }
    /**
     * @brief Inserts a point in the HNSW graph.
     * @param item_id The ID of the item to insert.
     */
    void insert(uint item_id){
        double x = dist(rng_);
        hnsw_graph_[item_id].max_layer = static_cast<uint>(-std::log(x) * mL_);
        if(hnsw_graph_[item_id].max_layer > max_layer_){
            max_layer_ = hnsw_graph_[item_id].max_layer;
        }
        hnsw_graph_[item_id].degrees.resize(hnsw_graph_[item_id].max_layer + 1);
        hnsw_graph_[item_id].offsets.resize(hnsw_graph_[item_id].max_layer + 1);

        vector<uint> W;
        uint ep = ep_; 
        // Finding Entry Points till the layer in which the item appears is reached.
        for(uint layer = max_layer_; layer > hnsw_graph_[item_id].max_layer; --layer){
            // Search for neighbors in the current layer
            W = search_layer(item_id, ep, 1, layer);
            // Get closest point to query item_id.
            ep = get_closest_point(item_id, W);
            
        }
        for(int layer = hnsw_graph_[item_id].max_layer; layer >=0; --layer){
            W = search_layer(item_id, ep, efConstruction_, layer);
            // Select M closest neighbors from W
            vector<uint> selected_neighbors = select_neighbors(item_id, W, M_);
            // Update the graph with the new neighbors
            update_graph(item_id, selected_neighbors, layer);
            if(layer > 0){
                ep = get_closest_point(item_id, selected_neighbors);
            }
        }


        
    }

    // UPDATE: This part needs to be reconfigured since we should rather pass the query vector than the item_id.
    /**
     * @brief Gives the closest point's item_id from the query item_id from the list of candidates W and deletes it from W.
     * @param query_id The item_id of the query point.
     * @param W The list of candidate points.
     * @return The item_id of the closest point.
     */
    uint get_closest_point(uint query_id,  vector<uint>& W) {
        float best_similarity = -1.0f;
        uint best_item_id = 0;
        uint best_index = 0;

        for (size_t i = 0; i < W.size(); ++i) {
            uint candidate_id = W[i];
            if(dot_products_[candidate_id] == 2.0f){
                dot_products_[candidate_id] = data_eigen_.row(query_id).dot(data_eigen_.row(candidate_id));
                reset_dot_products_.push_back(candidate_id);
            } 
            float similarity = dot_products_[candidate_id];

            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_item_id = candidate_id;
                best_index = i;
            }
        }
        // Remove the selected point from W
        W[best_index] = W.back();
        W.pop_back();
        return best_item_id;
    }

    /**
     * @brief Gives the farthest point's item_id from the query item_id from the list of candidates W and deletes it from W.
     * @param query_id The item_id of the query point.
     * @param W The list of candidate points.
     * @return The item_id of the farthest point.
     */
    uint get_farthest_point(uint query_id,  vector<uint>& W) {
        float worst_similarity = 1.0f;
        uint worst_item_id = 0;
        uint worst_index = 0;

        for (size_t i = 0; i < W.size(); ++i) {
            uint candidate_id = W[i];
            if(dot_products_[candidate_id] == 2.0f){
                dot_products_[candidate_id] = data_eigen_.row(query_id).dot(data_eigen_.row(candidate_id));
                reset_dot_products_.push_back(candidate_id);
            } 
            float similarity = dot_products_[candidate_id];

            if (similarity < worst_similarity) {
                worst_similarity = similarity;
                worst_item_id = candidate_id;
                worst_index = i;
            }
        }
        return worst_item_id;
    }

    /**
     * @brief Selects neighbors based on the heuristic technique described in the HNSW paper. Basically minimize the distance between the neighbors and the query point while maximizing the distance between the neighbors themselves.
     * @param query The item_id of the query point.
     * @param C The list of candidate points.
     * @param M The maximum number of neighbors to select.
     * @param layer The layer in which the neighbors are being selected.
     * @param extendCandidates Whether to extend the candidate list with neighbors of the candidates.
     * @param keepPrunedConnections Whether to keep pruned connections in the graph.

     * @return M elements selected from C based on the heuristic technique.
     */
     vector<uint> select_neighbors(uint query, const vector<uint>& C, uint M, uint layer, bool extendCandidates = false, bool keepPrunedConnections = false){
        vector<uint> W = C;
        unordered_set<uint> W_set(W.begin(), W.end());
        
        if(extendCandidates){
            // Extend the candidate list with neighbors of the candidates
            for(const uint& candidate : C){
                const Node& candidate_node = hnsw_graph_[candidate];
                if(layer <= candidate_node.max_layer){
                    uint offset = candidate_node.offsets[layer];
                    uint degree = candidate_node.degrees[layer];
                    for(uint i = 0; i < degree; ++i){
                        if(W_set.find(candidate_node.neighbors[offset + i]) == W_set.end()){
                            W.push_back(candidate_node.neighbors[offset + i]);
                            W_set.insert(candidate_node.neighbors[offset + i]);
                        }
                    }
                }
                        
            }
        }
        vector<uint> W_d;
        vector<uint> selected_neighbors;

        while(W.size()!=0 && selected_neighbors.size() < M){
            
            // Find the closest point to the query in W
            uint closest_point = get_closest_point(query, W);
            float similarity_W = dot_products_[closest_point];
            bool add_to_selected = true;
            for(const uint& selected : selected_neighbors){
                float similarity_selected_neighbors = data_eigen_.row(closest_point).dot(data_eigen_.row(selected));
                if(similarity_selected_neighbors > similarity_W){
                    add_to_selected = false;
                    break;
                }
            }
            if(add_to_selected){
                selected_neighbors.push_back(closest_point);
            }
            else{
                W_d.push_back(closest_point);
            }
        }
        if(keepPrunedConnections){
            // Add some of the pruned connections to the selected neighbors
            while(selected_neighbors.size() < M && W_d.size() > 0){
                uint closest_point = get_closest_point(query, W_d);
                selected_neighbors.push_back(closest_point);
            }
        }
        return selected_neighbors;
    }

    /**
     * @brief Search a layer of the HNSW graph for neighbors of a query point.
     * @param query The item_id of the query point.
     * @param ep The item_id of the entry point for the search.
     * @param ef The number of candidates to consider during the search.
     * @param layer The layer in which the neighbors are being selected.

     * @return Elements that are closest to the query point according to the similarity threshold.
     */
    // UPDATE: This part needs to be reconfigured since we should rather pass the query vector than the item_id.
     vector<uint> search_layer(uint query, uint ep, uint ef, uint layer){
        visited_[ep] = true;
        reset_visited_.push_back(ep);
        vector<uint> Found_Neighbors;
        vector<uint> Candidates;
        Candidates.push_back(ep);
        Found_Neighbors.push_back(ep);
        while(!Candidates.empty()){
            uint closest_candidate = get_closest_point(query, Candidates);
            uint farthest_neighbor = get_farthest_point(query, Found_Neighbors);
            if(dot_products_[closest_candidate] < dot_products_[farthest_neighbor]){
                break;
            }
            const Node& closest_candidate_node = hnsw_graph_[closest_candidate];
            if(layer <= closest_candidate_node.max_layer){
                uint offset = closest_candidate_node.offsets[layer];
                uint degree = closest_candidate_node.degrees[layer];
                for(uint i = 0; i < degree; ++i){
                    uint neighbor = closest_candidate_node.neighbors[offset + i];
                    if(!visited_[neighbor]){
                        visited_[neighbor] = true;
                        reset_visited_.push_back(neighbor);
                        farthest_neighbor = get_farthest_point(query, Found_Neighbors);
                        
                        if(dot_products_[neighbor] == 2.0f){
                            dot_products_[neighbor] = data_eigen_.row(query).dot(data_eigen_.row(neighbor));
                            reset_dot_products_.push_back(neighbor);
                        }
                        if(dot_products_[neighbor] > dot_products_[farthest_neighbor] || Found_Neighbors.size() < ef){
                            Candidates.push_back(neighbor);
                            Found_Neighbors.push_back(neighbor);
                            if(Found_Neighbors.size() > ef){
                                uint worst_neighbor = get_farthest_point(query, Found_Neighbors);
                                // Remove the worst neighbor from Found_Neighbors
                                for (size_t i = 0; i < Found_Neighbors.size(); ++i) {
                                    if (Found_Neighbors[i] == worst_neighbor) {
                                        Found_Neighbors[i] = Found_Neighbors.back();
                                        Found_Neighbors.pop_back();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for(const uint& id : reset_visited_){
            visited_[id] = false;
        }
        reset_visited_.clear();
        return Found_Neighbors;
    }


    vector<uint> search(uint query_id, uint ef){
        vector<uint> Found_Neighbors;
        uint ep = ep_;
        for(uint layer = max_layer_; layer > 0; --layer){
            Found_Neighbors = search_layer(query_id, ep, 1, layer);
            ep = get_closest_point(query_id, Found_Neighbors);
        }
        Found_Neighbors = search_layer(query_id, ep, ef, 0);
        return Found_Neighbors;
    }



};


#endif // A7F3C921_5D84_4B6E_9A12_E8C4F07B3D56