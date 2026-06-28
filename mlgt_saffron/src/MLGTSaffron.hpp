#ifndef B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D
#define B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D

#include <tuple>
#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"


/**
 * @brief Multi-Label Group Testing (MLGT) Saffron implementation.
 * 
 * Uses Multi-Label Group Testing principles combined with a per-pool inverted index 
 * (Bloom Filtering) to identify candidate items in each pool for fast recovery.
 */
class MLGTSaffron : public Saffron {
protected:
    BloomHashFunction shared_hasher_; 
    vector<GlobalInvertedIndex> pool_indices_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen_;
    vector<vector<bool>> item_signatures_;
    uint num_hashes_;
    uint hash_bits_;
    uint threshold_;
    uint dimension_;
    bool normalize_; 

public:
    /**
     * @brief Initializes the MLGTSaffron index.
     */
    // To store number of tests
     uint total_tests_;

    MLGTSaffron(
        pybind11::array_t<float> data_points_arr,
        uint num_neighbors = 100,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint hash_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : Saffron(data_points_arr.shape(0), num_neighbors, debug),
        shared_hasher_(data_points_arr.shape(1), num_hashes, hash_bits, threshold, debug),
        num_hashes_(num_hashes),
        hash_bits_(hash_bits),
        threshold_(threshold),
        dimension_(data_points_arr.shape(1)),
        normalize_(normalize)
    {
        // Pre-calculate signatures for fast search
        item_signatures_.resize(num_features_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_features_; ++i) {
            item_signatures_[i] = getSignature(i, signature_length_);
        }

        // Convert to Eigen Matrix
        auto r = data_points_arr.unchecked<2>();
        data_eigen_.resize(r.shape(0), r.shape(1));
        #pragma omp parallel for
        for (int i = 0; i < (int)r.shape(0); ++i) {
            float norm_sq = 0;
            for (int j = 0; j < (int)r.shape(1); ++j) {
                float val = r(i, j);
                data_eigen_(i, j) = val;
                norm_sq += val * val;
            }
            if (normalize_) {
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-9) data_eigen_.row(i) /= norm;
            }
        }

        // Pre-calculate hashes for all items
        vector<vector<uint>> all_hashes(num_features_);
        #pragma omp parallel for
        for (int item_idx = 0; item_idx < (int)num_features_; ++item_idx) {
            all_hashes[item_idx] = shared_hasher_(data_eigen_.row(item_idx));
        }

        // Build one index PER POOL
        pool_indices_.resize(num_pools_);
        #pragma omp parallel for
        for (int p = 0; p < (int)num_pools_; ++p) {
            pool_indices_[p] = GlobalInvertedIndex(num_hashes_, threshold_);
            
            vector<vector<uint>> pool_hashes;
            pool_hashes.reserve(pools_.pools_to_items[p].size());
            for (uint global_idx : pools_.pools_to_items[p]) {
                pool_hashes.push_back(all_hashes[global_idx]);
            }
            pool_indices_[p].build(pool_hashes, pools_.pools_to_items[p]);
        }

        // Logging memory of All Poolwise Indexes created
        if (debug_ > 0) {
            size_t GI_memory=0;
            for (int p = 0; p < (int)num_pools_; ++p) {
                
                GI_memory+=pool_indices_[p].memoryUsage();
            }
            double gb = GI_memory / (1024.0 * 1024.0 * 1024.0);
            cout<<"Total Memory used in Poolwise Global Inverted Index  "<<gb<<"GB \n";
        }

        // Adding number of tests logging
        total_tests_ = num_pools_ * signature_length_;
        if (debug_ > 0) {
            cout << "[MLGTSaffron] Number of tests = "
                << total_tests_
                << " (pools=" << num_pools_
                << ", bits=" << signature_length_ << ")"
                << endl;
        }
        
        if (debug_ > 0) {
            cout << "[MLGTSaffron] Built " << num_pools_ << " pool indices." << endl;
        }
    }

    ~MLGTSaffron() = default;

protected:
    /**
     * @brief Computes test residuals by identifying candidate items in each pool.
     * 
     * Uses the per-pool inversion indices to find items likely to be similar to the 
     * query, then combines their pre-computed signatures to form the pool residual.
     * 
     * @param query_vec Normalized query vector (Eigen).
     * @return vector<vector<bool>> The num_pools x signature_bits residual matrix.
     */
    // Changed getResiduals to return pair {residuals, hashing_time}
    inline std::pair<vector<vector<bool>>, double> getResiduals(const Eigen::VectorXf& query_vec) const {

        // Hashing Time
        auto t_hash_start = std::chrono::high_resolution_clock::now();
        vector<uint> query_hashes = shared_hasher_(query_vec);
        auto t_hash_end = std::chrono::high_resolution_clock::now();
        double hashing_time = std::chrono::duration<double>(t_hash_end - t_hash_start).count();
        
        
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        
        #pragma omp parallel for
        for (int p = 0; p < (int)num_pools_; ++p) {
            vector<uint> matched_items = pool_indices_[p].get_matches(query_hashes);
            for (uint global_item_idx : matched_items) {
                const vector<bool>& sig = item_signatures_[global_item_idx];
                for (uint b = 0; b < signature_length_; ++b) {
                    if (sig[b]) {
                        residuals[p][b] = !residuals[p][b]; // XOR modulo 2
                    }
                }
            }
        }
        return {residuals, hashing_time};
    }

public:
    /**
     * @brief Performs a nearest neighbor search.
     * 
     * @param query_arr The query vector (numpy array).
     * @return vector<uint> Top K item indices.
     */
    // Changed search to return (topK, hashing_time, decoding_time)
    inline std::tuple<std::vector<uint>, double, double> search(pybind11::array_t<float> query_arr) {
        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), dimension_);
        Eigen::VectorXf query = q_raw;
        if (normalize_) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        // Updated return of getResiduals()
        auto [residuals, hashing_time] = getResiduals(query);
        
        // Decoding Time
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        set<uint> identified = peelingAlgorithm(residuals);
        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decoding_time = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
        

        return {getTopKEigen(query, data_eigen_, identified, sparsity_), hashing_time, decoding_time};
    }

    /**
     * @brief Performs a search using a callable interface.
     * @param query_arr The query vector.
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return std::get<0>(search(query_arr));
    }
};



#endif /* B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D */
