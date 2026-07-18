#ifndef H_9F8C2A17_6D41_4E9B_B5F3_8C71D2A4E9F6
#define H_9F8C2A17_6D41_4E9B_B5F3_8C71D2A4E9F6

#include "headers.hpp"

/**
 * @brief Represents the mapping between items and pools for the SAFFRON algorithm.
 */
struct NovaMatrix {
    vector<vector<uint>> pools_to_items; ///< List of vectors, where each vector contains the item indices in that pool.
    vector<vector<uint>> items_to_pools; ///< List of vectors, where each vector contains the pool indices the item belongs to.
    uint num_features; ///< Total number of features/items (n).
    uint num_pools; ///< Total number of pools (m).
};

/**
 * @brief Randomized pooling matrix generation for SAFFRON.
 * 
 * Standard Left regular graph.
 * Each Right node is a test bundle
 *
 * @param num_features Total number of features (n).
 * @param sparsity Expected sparsity level (k).
 * @param debug Debug verbosity level.
 * @return NovaMatrix The mapping between items and pools.
 */
inline NovaMatrix computeNovaPools(uint num_features, uint sparsity, int debug = 0) {
    uint num_pools = sparsity * C_epsilon;
    // left degree d
    uint pools_per_item = POOLS_PER_ITEM;
    
    NovaMatrix pooling_matrix;
    pooling_matrix.num_features = num_features;
    pooling_matrix.num_pools = num_pools;
    pooling_matrix.pools_to_items.resize(num_pools);
    pooling_matrix.items_to_pools.resize(num_features);

    // fixed random seed 42
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint> dis(0, num_pools - 1);
    
    for (uint item_idx = 0; item_idx < num_features; ++item_idx) {
        unordered_set<uint> chosen_pools;
        while (chosen_pools.size() < pools_per_item) {
            chosen_pools.insert(dis(gen));
        }
        for (uint pool_idx : chosen_pools) {
            pooling_matrix.pools_to_items[pool_idx].push_back(item_idx);
            pooling_matrix.items_to_pools[item_idx].push_back(pool_idx);
        }
    }
    
    if (debug > 0) {
        cout << "[Compute Pools] num_features=" << num_features
             << ", sparsity=" << sparsity
             << ", num_pools=" << num_pools
             << ", pools_per_item=" << pools_per_item << endl;
    }
    
    return pooling_matrix;
}


#endif // H_9F8C2A17_6D41_4E9B_B5F3_8C71D2A4E9F6
