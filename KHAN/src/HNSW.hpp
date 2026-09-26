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
        max_layer_(0)
    {
        hnsw_graph_.resize(num_features_);
        // Insert the points sequentially and build the hnsw_graph_
        for(uint i = 0; i < num_features_; ++i){
            insert(i);
        }


    }

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
            W = search_layer(item_id, ep, ef=1, layer);
            // Get closest point to query item_id.
            ep = get_closest_point(item_id, W);
            
        }
        for(uint layer = hnsw_graph_[item_id].max_layer; layer >=0; --layer){
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


    uint get_closest_point(uint item_id, const vector<uint>& W) {
        float best_similarity = -1.0f;
        uint best_item_id = 0;

        for (const uint& candidate_id : W) {
            float similarity =
                data_eigen_.row(item_id).dot(data_eigen_.row(candidate_id));

            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_item_id = candidate_id;
            }
        }
        return best_item_id;
    }


};


#endif // A7F3C921_5D84_4B6E_9A12_E8C4F07B3D56