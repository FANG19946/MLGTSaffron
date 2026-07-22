#ifndef OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3
#define OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3


#include <atomic>
#include <tuple>
#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "PulsarSaffron.hpp"
#include "OrionIndex.hpp"


/**
 * @brief Multi-Label Group Testing (MLGT) Saffron implementation.
 * 
 * Uses Multi-Label Group Testing principles combined with a Inverted Index OrionIndex.
 * (Bloom Filtering) to identify candidate items in each pool for fast recovery.
 */
class OdysseyMLGT : public PulsarSaffron {
protected:
    BloomHashFunction shared_hasher_; 
    vector<GlobalInvertedIndex> pool_indices_; // This is probably no longer needed.
    OrionIndex heliosIndex_; // Haven't thought too much of initialization will comeback to this later.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen_;
    vector<vector<bool>> item_signatures_; // I already created the same item in PulsarSafforn called signature_matrix_ so we will just remove or refactor this  
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

    OdysseyMLGT(
        pybind11::array_t<float> data_points_arr,
        uint num_neighbors = 100,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint hash_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : PulsarSaffron(data_points_arr.shape(0), num_neighbors, debug),
        shared_hasher_(data_points_arr.shape(1), num_hashes, hash_bits, threshold, debug),
        num_hashes_(num_hashes),
        hash_bits_(hash_bits),
        threshold_(threshold),
        dimension_(data_points_arr.shape(1)),
        normalize_(normalize)
    {


        // Convert to Eigen Matrix
        cout<<"Convert to Eigen Matrix"<<endl;
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

        std::atomic<uint64_t> processed = 0;
        // Pre-calculate hashes for all items
        cout<<"Pre-calculate hashes for all items"<<endl;
        vector<vector<uint>> all_hashes(num_features_);
        #pragma omp parallel for
        for (int item_idx = 0; item_idx < (int)num_features_; ++item_idx) {
            all_hashes[item_idx] = shared_hasher_(data_eigen_.row(item_idx));

            // Loader
            uint64_t cnt = ++processed;
            if (cnt % 10000 == 0 || cnt == num_features_) {
                #pragma omp critical
                {
                    std::cout << "\rProcessed " << cnt << " / " << num_features_
                            << " (" << std::fixed << std::setprecision(1)
                            << (100.0 * cnt / num_features_) << "%)"
                            << std::flush;
                }
            }
        }
        std::cout << std::endl;

        // cout << "Loading precomputed hashes..." << endl;
        // vector<vector<uint>> all_hashes(num_features_);

        // // TODO: replace with your filename
        // std::ifstream file("../results/all_hashes.csv");

        // std::string line;
        // for (uint i = 0; i < num_features_ && std::getline(file, line); i++) {
        //     std::stringstream ss(line);
        //     std::string value;

        //     while (std::getline(ss, value, ',')) {
        //         all_hashes[i].push_back(std::stoul(value));
        //     }
        // }
        // cout << "Finished loading hashes." << endl;
        


        // Dump hashes to CSV
        // {
        //     std::ofstream out("results/all_hashes.csv");

        //     for (const auto& hashes : all_hashes) {
        //         for (size_t i = 0; i < hashes.size(); i++) {
        //             if (i) out << ",";
        //             out << hashes[i];
        //         }
        //         out << "\n";
        //     }
        // }

        // cout << "Saved all_hashes to results/all_hashes_2.csv" << endl;
        

        // Build heliosIndex_
        cout<<"Building OrionIndex"<<endl;
        uint hash_range = 1u << hash_bits_;
        heliosIndex_ = OrionIndex(hash_range, num_hashes_, threshold_);
        // Index uses the extended pooling matrix
        heliosIndex_.build(all_hashes, extended_pooling_matrix_.pools_to_items);
        cout<<"OrionIndex Built"<<endl;
                


        // Adding number of tests logging
        total_tests_ = num_pools_ * signature_length_;
        if (debug_ > 0) {
            cout << "[OdysseyMLGT] Number of tests = "
                << total_tests_
                << " (pools=" << num_pools_
                << ", bits=" << signature_length_ << ")"
                << endl;
        }
        
        if (debug_ > 0) {
            cout << "[OdysseyMLGT] Built " << num_pools_ << " pool indices." << endl;
        }
    }

    ~OdysseyMLGT() = default;

protected:
    /**
     * @brief Computes test residuals by identifying candidate items in each pool.
     * 
     * Uses the per-pool inversion indices to find items likely to be similar to the 
     * query, then combines their pre-computed signatures to form the pool residual.
     * 
     * @param query_vec Normalized query vector (Eigen).
     * @return vector<vector<bool>> The num_pools x signature_bits residual matrix, set<uint> identified_defectives contains the set of items that are defectives, double hashing_time Time taken to hash th query vector.
     */
    // Changed getResiduals to return tuple {residuals, identified_defectives, hashing_time}
    inline std::tuple<vector<vector<bool>>, set<uint>,  double, double> getResiduals(const Eigen::VectorXf& query_vec) const {

        // Hashing Time
        auto t_hash_start = std::chrono::high_resolution_clock::now();
        vector<uint> query_hashes = shared_hasher_(query_vec);
        auto t_hash_end = std::chrono::high_resolution_clock::now();
        double hashing_time = std::chrono::duration<double>(t_hash_end - t_hash_start).count();
        
        
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        set<uint> identified_defectives;

        // Check if parallel threading works
        auto t_test_evaluation_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(uint pool_id = 0; pool_id < num_pools_; pool_id++){
            uint extended_base = pool_id * signature_length_;
            for(uint j = 0; j < signature_length_; j++){
                uint extended_pool_id = extended_base + j;
                auto [pool_status, global_id] = heliosIndex_.get_matches(query_hashes, extended_pool_id);
                residuals[pool_id][j] = pool_status;
                if(pool_status){
                    identified_defectives.insert(global_id);
                }
            }
        }
        auto t_test_evaluation_end = std::chrono::high_resolution_clock::now();
        double test_evaluation_time = std::chrono::duration<double>(t_test_evaluation_end - t_test_evaluation_start).count();


        return {residuals, identified_defectives, hashing_time, test_evaluation_time};
    }

public:
    /**
     * @brief Performs a nearest neighbor search.
     * 
     * @param query_arr The query vector (numpy array).
     * @return vector<uint> Top K item indices.
     */
    // Changed search to return (topK, hashing_time, decoding_time)
    inline std::tuple<std::vector<uint>, double, double, double> search(pybind11::array_t<float> query_arr) {
        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), dimension_);
        Eigen::VectorXf query = q_raw;
        if (normalize_) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        // Updated return of getResiduals()
        auto [residuals, identified_defectives, hashing_time, test_evaluation_time] = getResiduals(query);
        
        // Decoding Time
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        set<uint> identified = peelingAlgorithm(residuals, identified_defectives);
        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decoding_time = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
        

        return {getTopKEigen(query, data_eigen_, identified, sparsity_), hashing_time, decoding_time, test_evaluation_time };
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



#endif // OE6D4A82_31B5_4F0C_A8E1_7C9D2B5F64A3
